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

WM = Path(__file__).parent / "fixtures" / "watermark"
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
    assert f.details["payload_text"] == "markchk"
    assert f.details["schema"].startswith("BCH_5")
    for word in ("authentic", "human-made", "not AI"):
        assert word not in f.summary


@needs_weights
@pytest.mark.parametrize("name", UNMARKED + SD_MARKED)
def test_trustmark_not_found_on_other_pictures(name):
    assert _finding(WM / name, "trustmark").status == "not_found"


def test_trustmark_fixture_listed():
    assert TM_MARKED == ["trustmark_q_marked.png"]


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
