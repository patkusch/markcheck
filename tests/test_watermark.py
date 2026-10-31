"""Hidden-mark checks: Stable Diffusion (invisible-watermark) and Adobe TrustMark.

Fully offline. TrustMark tests that need the model files skip with a reason
when the files are not on disk.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from markcheck.cli import check_file
from markcheck.detectors import sd_watermark, trustmark_check

FIX_ROOT = Path(__file__).parent / "fixtures"
WM = FIX_ROOT / "watermark"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MANIFEST = json.loads((WM / "manifest.json").read_text())["files"]

pytest.importorskip("imwatermark")


def _make_module():
    spec = importlib.util.spec_from_file_location("make_fixtures_watermark", SCRIPTS / "make_fixtures_watermark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _finding(path, name):
    return next(f for f in check_file(path).findings if f.detector == name)


def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


# ---------- Stable Diffusion ----------

SD_MARKED = [e["file"] for e in MANIFEST if "sd_watermark" in e["expect"]]
UNMARKED = [e["file"] for e in MANIFEST if not e["expect"]]


def test_fixture_set_is_complete():
    assert set(SD_MARKED) >= {"sd_v1_marked.png", "sd_v2_marked.png", "sdxl_marked.png"}
    assert UNMARKED


def test_base_pictures_are_reproducible():
    mk = _make_module()
    for scene in mk.SCENES:
        a = np.array(mk.base_image(7, (300, 260), scene))
        b = np.array(mk.base_image(7, (300, 260), scene))
        assert np.array_equal(a, b)


@pytest.mark.parametrize("name", SD_MARKED)
def test_sd_marked_fixtures_are_found(name):
    f = _finding(WM / name, "sd_watermark")
    assert f.status == "found", f.summary
    assert "bits match" in f.summary
    assert f.does_not_prove


@pytest.mark.parametrize("name,payload", [("sd_v1_marked.png", "SD v1"), ("sd_v2_marked.png", "SD v2"),
                                          ("sdxl_marked.png", "SDXL")])
def test_sd_reports_the_right_payload(name, payload):
    f = _finding(WM / name, "sd_watermark")
    assert f"{payload} payload" in f.summary


@pytest.mark.parametrize("name", UNMARKED)
def test_unmarked_fixtures_are_not_found(name):
    assert _finding(WM / name, "sd_watermark").status == "not_found"


def test_many_unmarked_pictures_never_reach_the_bar():
    """60 drawn pictures and 20 noise pictures, none marked: no false alarm, and the
    best reading stays well under the bar (measured worst so far: about 69% of bits)."""
    mk = _make_module()
    rng = np.random.default_rng(0)
    pics = [np.array(mk.base_image(seed, size=(512 + seed % 5 * 64, 512)))[:, :, ::-1].copy()
            for seed in range(100, 160)]
    pics += [rng.integers(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(20)]
    worst = 0.0
    for bgr in pics:
        for name, (m, n) in sd_watermark.match_rates(bgr).items():
            assert m < sd_watermark.NEEDED[name]
            worst = max(worst, m / n)
    assert worst < 0.8


def test_numpy_reader_matches_the_library_decoder():
    """Our fast reader must give the same bits as imwatermark's own WatermarkDecoder."""
    import cv2
    from imwatermark import WatermarkDecoder

    for name in SD_MARKED + UNMARKED:
        bgr = cv2.imread(str(WM / name))
        for bits in sd_watermark.PAYLOADS.values():
            n = len(bits)
            ours = sd_watermark.vote(sd_watermark.block_bits(bgr), n)
            theirs = WatermarkDecoder("bits", n).decode(bgr, "dwtDct")
            assert np.array_equal(ours, np.array(theirs, dtype=bool)), name


def test_sdxl_needs_the_float_read_mode():
    """diffusers marks a float picture; read the uint8 way, the SDXL bits come out mixed."""
    import cv2

    bgr = cv2.imread(str(WM / "sdxl_marked.png"))
    want = np.array(sd_watermark.PAYLOADS["SDXL"], dtype=bool)
    as_float = sd_watermark.vote(sd_watermark.block_bits(bgr, True), 48)
    assert int(np.sum(as_float == want)) >= sd_watermark.NEEDED["SDXL"]


def test_thresholds_follow_the_chance_rule():
    assert sd_watermark.NEEDED == {"SDXL": 42, "SD v1": 98, "SD v2": 30}


def test_sd_payload_definitions():
    assert hex(sd_watermark.SDXL_MESSAGE) == "0xb3ec907bb19e"
    assert len(sd_watermark.PAYLOADS["SDXL"]) == 48
    assert np.packbits(sd_watermark.PAYLOADS["SD v1"]).tobytes() == b"StableDiffusionV1"
    assert np.packbits(sd_watermark.PAYLOADS["SD v2"]).tobytes() == b"SDV2"


def test_sd_skips_non_images(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    f = _finding(p, "sd_watermark")
    assert f.status == "skipped"


def test_sd_skips_small_pictures():
    data = _png_bytes(np.zeros((100, 100, 3), dtype=np.uint8))
    [f] = sd_watermark.check(Path("x.png"), data, "image", {})
    assert f.status == "skipped" and "too small" in f.summary


def test_sd_skips_unreadable_picture():
    [f] = sd_watermark.check(Path("x.png"), b"\x89PNG not really", "image", {})
    assert f.status == "skipped"


# ---------- TrustMark ----------

TM_MARKED = [e["file"] for e in MANIFEST if "trustmark" in e["expect"]]
needs_weights = pytest.mark.skipif(
    importlib.util.find_spec("trustmark") is None or not trustmark_check.available(),
    reason="TrustMark model files are not on this computer (run: python -m markcheck.fetch_models)",
)


@needs_weights
@pytest.mark.parametrize("name", TM_MARKED)
def test_trustmark_marked_fixture_is_found(name):
    f = _finding(WM / name, "trustmark")
    assert f.status == "found", f.summary
    entry = next(e for e in MANIFEST if e["file"] == name)
    assert f.details["payload_text"] == entry.get("payload", "markchk")
    assert f.details["schema"].startswith("BCH_5")
    for word in ("authentic", "human-made", "not AI"):
        assert word not in f.summary


@needs_weights
@pytest.mark.parametrize("name", UNMARKED + SD_MARKED)
def test_trustmark_not_found_on_other_pictures(name):
    assert _finding(WM / name, "trustmark").status == "not_found"


def test_trustmark_fixture_listed():
    assert "trustmark_q_marked.png" in TM_MARKED and len(TM_MARKED) >= 5


def test_sd_fixture_set_is_varied():
    entries = [e for e in MANIFEST if "sd_watermark" in e["expect"]]
    assert len(entries) >= 6
    assert {e["payload"] for e in entries} == {"SD v1", "SD v2", "SDXL"}
    assert any(e["file"].endswith(".jpg") for e in entries)


def test_trustmark_chance_rule():
    """Only readings a random picture would match less than once in a million count."""
    assert {v: trustmark_check.max_fixed_bits(v) for v in range(4)} == {0: 8, 1: 2, 2: 1, 3: 0}
    assert trustmark_check.chance_by_luck(3, 3) > 1e-3  # what the library alone accepts for BCH_3


@needs_weights
def test_trustmark_lucky_weak_reading_is_not_counted():
    """Round 2 finding: this ordinary control passes the library's BCH_3 check by luck.
    The library alone says 'present'; markcheck must not say found."""
    from PIL import Image

    path = FIX_ROOT / "controls" / "caption_mentions_ai.png"
    tm = trustmark_check.get_model()
    rgb = Image.open(path).convert("RGB")
    _, present, version = tm.decode(rgb, MODE="binary")
    assert present and version == 3
    f = _finding(path, "trustmark")
    assert f.status == "not_found"
    assert "weak error check" in f.summary


@needs_weights
def test_trustmark_reader_matches_library_decode():
    """Our raw-bit reading must give the same answer as TrustMark.decode()."""
    from PIL import Image

    tm = trustmark_check.get_model()
    paths = [WM / e["file"] for e in MANIFEST] + [FIX_ROOT / "controls" / "caption_mentions_ai.png"]
    for p in paths:
        rgb = Image.open(p).convert("RGB")
        ours = trustmark_check.read(tm, rgb)
        secret, present, version = tm.decode(rgb, MODE="binary")
        assert ours["passed_library_check"] == present, p.name
        if present:
            assert (ours["secret"], ours["version"]) == (secret, version), p.name


@needs_weights
def test_trustmark_many_unmarked_pictures_never_found():
    """150 unmarked pictures of three kinds. Before the round 2 fix about 1 in 25 'passed'."""
    mk = _make_module()
    spec = importlib.util.spec_from_file_location("make_fixtures_controls", SCRIPTS / "make_fixtures_controls.py")
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)
    pics = [mk.base_image(s, (384 + s % 3 * 128, 384), mk.SCENES[s % 3]) for s in range(200, 260)]
    pics += [ctl.photo_like(s, (320 + s % 4 * 64, 320)) for s in range(300, 360)]
    pics += [ctl.flat_graphic(s, (400, 300)) for s in range(500, 530)]
    tm = trustmark_check.get_model()
    lucky = 0
    for im in pics:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        f = trustmark_check.check("x.png", buf.getvalue(), "image", {})[0]
        assert f.status == "not_found"
        lucky += trustmark_check.read(tm, im.convert("RGB"))["passed_library_check"]
    assert lucky >= 1  # the library alone would have raised false alarms here


def test_trustmark_skips_without_model_files(monkeypatch):
    if importlib.util.find_spec("trustmark") is None:
        pytest.skip("trustmark not installed")
    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant="Q": ["decoder_Q.ckpt"])
    [f] = trustmark_check.check(Path("x.png"), _png_bytes(np.zeros((300, 300, 3), np.uint8)), "image", {})
    assert f.status == "skipped"
    assert "markcheck.fetch_models" in f.summary


def test_trustmark_never_downloads(monkeypatch):
    """Even with the files missing, a check must not reach the network."""
    if importlib.util.find_spec("trustmark") is None:
        pytest.skip("trustmark not installed")
    import urllib.request

    def boom(*a, **k):
        raise AssertionError("tried to download")

    monkeypatch.setattr(urllib.request, "urlretrieve", boom)
    monkeypatch.setattr(trustmark_check, "_VERIFIED", set())
    monkeypatch.setattr(trustmark_check, "_models_dir", lambda: Path("/nonexistent"))
    monkeypatch.setattr(trustmark_check, "_MODEL", None)
    [f] = trustmark_check.check(Path("x.png"), _png_bytes(np.zeros((300, 300, 3), np.uint8)), "image", {})
    assert f.status == "skipped"


def test_trustmark_skips_when_library_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "trustmark", None)
    [f] = trustmark_check.check(Path("x.png"), b"", "image", {})
    assert f.status == "skipped" and "markcheck[trustmark]" in f.summary


def test_trustmark_skips_non_images():
    [f] = trustmark_check.check(Path("a.txt"), b"hello", "text", {})
    assert f.status == "skipped"


def test_trustmark_with_fixed_bits_is_not_called_error_free(monkeypatch):
    """BCH_SUPER may accept a reading after fixing up to 8 bits. The output must then
    say bits were fixed, not that the message 'read back without errors'."""
    if importlib.util.find_spec("trustmark") is None:
        pytest.skip("trustmark not installed")
    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant="Q": [])
    monkeypatch.setattr(trustmark_check, "get_model", lambda variant="Q": None)
    monkeypatch.setattr(trustmark_check, "read", lambda tm, rgb: {
        "passed_library_check": True, "version": 0, "secret": "1" * 40, "text": "",
        "bits_fixed": 5, "unused_bits_zero": True})
    [f] = trustmark_check.check(Path("x.png"), _png_bytes(np.zeros((300, 300, 3), np.uint8)), "image", {})
    assert f.status == "found"
    words = " ".join([f.summary, *f.proves])
    assert "without errors" not in words
    assert "5 bits" in words


def test_sd_comment_numbers_match_measurements():
    """The numbers written in sd_watermark's comment must hold on the committed fixtures
    and the generated unmarked pictures."""
    import re

    src = Path(sd_watermark.__file__).read_text()
    block = src[src.index("# How many bits must match"):src.index("FALSE_ALARM_BUDGET =")]
    flat = " ".join(block.replace("#", " ").split())
    ranges = {name: (int(lo), int(hi)) for lo, hi, _, name in
              re.findall(r"(\d+)-(\d+) of (\d+) \((SDXL|SD v2|SD v1)\)", flat)}
    assert set(ranges) == {"SDXL", "SD v2", "SD v1"}
    for e in MANIFEST:
        if "sd_watermark" in e["expect"]:
            m, _ = sd_watermark.match_rates(sd_watermark.load_bgr((WM / e["file"]).read_bytes()))[e["payload"]]
            lo, hi = ranges[e["payload"]]
            assert lo <= m <= hi, (e["file"], m, ranges[e["payload"]])
    stated_worst = int(re.search(r"(\d+)% of bits", flat).group(1)) / 100
    mk = _make_module()
    rng = np.random.default_rng(0)
    pics = [np.array(mk.base_image(seed, size=(512 + seed % 5 * 64, 512)))[:, :, ::-1].copy()
            for seed in range(100, 160)]
    pics += [rng.integers(0, 256, (512, 512, 3), dtype=np.uint8) for _ in range(20)]
    pics += [sd_watermark.load_bgr((WM / f).read_bytes()) for f in UNMARKED]
    worst = max(m / n for bgr in pics for m, n in sd_watermark.match_rates(bgr).values())
    assert worst <= stated_worst, (worst, stated_worst)
