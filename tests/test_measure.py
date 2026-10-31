import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
WM = FIX / "watermark"
C2PA = FIX / "c2pa"
META = FIX / "metadata"


def _measure_module():
    spec = importlib.util.spec_from_file_location("measure", ROOT / "scripts" / "measure.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure"] = mod  # dataclasses look the module up while it loads
    spec.loader.exec_module(mod)
    return mod


m = _measure_module()

NEW_COLUMNS = ["JPEG q75, metadata kept", "JPEG q75, metadata dropped",
               "50% resize, metadata kept", "50% resize, metadata dropped", "screenshot-like"]


def test_columns_split_pixels_from_metadata():
    assert m.VARIANTS == ["original"] + NEW_COLUMNS


def test_measure_counts_hits_alarms_and_survival(tmp_path):
    d = tmp_path / "set"
    d.mkdir()
    shutil.copy(WM / "sd_v2_marked.png", d)
    shutil.copy(WM / "unmarked_sd_v1.jpg", d)
    (d / "manifest.json").write_text(json.dumps({"files": [
        {"file": "sd_v2_marked.png", "expect": ["sd_watermark"], "note": "marked"},
        {"file": "unmarked_sd_v1.jpg", "expect": [], "note": "control"},
    ]}))
    res = m.measure(tmp_path)
    assert res["files"] == 2 and res["controls"] == 1
    sd = res["detectors"]["sd_watermark"]
    assert (sd["hits"], sd["labelled"], sd["false_alarms"]) == (1, 1, 0)
    surv = res["survival"]["sd_watermark"]
    assert surv["original"] == {"found": 1, "says_changed": 0, "of": 1}
    assert set(surv) == set(m.VARIANTS)
    table = m.tables(res)
    assert "| sd_watermark |" in table
    for col in NEW_COLUMNS:
        assert col in table
    assert "How each copy was made" in table


def test_metadata_kept_versus_dropped(tmp_path):
    """Keeping metadata keeps an XMP tag through a JPEG re-save; dropping it loses it."""
    src = META / "iptc_trained_uri_attr.jpg"
    kept = m.jpeg_q75(src, tmp_path, keep=True)
    dropped = m.jpeg_q75(src, tmp_path, keep=False)
    assert "metadata" in m.found_detectors(kept)
    assert "metadata" not in m.found_detectors(dropped)
    assert m.read_meta(dropped).xmp is None


@pytest.mark.parametrize("name", ["ai_generated.jpg", "ai_generated.png", "ai_generated.webp"])
def test_c2pa_carried_along_says_changed(tmp_path, name):
    """A signed label copied onto re-compressed pixels must report the change, not pass as intact."""
    copy = m.jpeg_q75(C2PA / name, tmp_path, keep=True)
    f = m.findings(copy)["c2pa"]
    assert m.says_changed(f)
    half = m.half_size(C2PA / name, tmp_path, keep=True)
    assert m.says_changed(m.findings(half)["c2pa"])
    assert "c2pa" not in m.found_detectors(m.jpeg_q75(C2PA / name, tmp_path, keep=False))


def test_jumbf_round_trip():
    box = (C2PA / "ai_generated.png").read_bytes()
    jumbf = next(b for t, b in m._png_chunks(box) if t == b"caBX")
    segs = m._app11_from_jumbf(jumbf)
    assert len(segs) > 1  # bigger than one JPEG segment, so the continuation path is used
    assert m._jumbf_from_app11(segs) == jumbf


def test_gif_and_tiff_left_out_of_kept_columns(tmp_path):
    assert m.jpeg_q75(C2PA / "ai_generated.gif", tmp_path, keep=True) is None
    assert m.half_size(C2PA / "ai_generated.tif", tmp_path, keep=True) is None


@pytest.fixture(scope="module")
def full():
    return m.measure(FIX)


def test_no_false_alarms_on_any_control(full):
    assert full["controls"] >= 30
    assert full["false_alarm_files"] == []
    for d, r in full["detectors"].items():
        assert r["false_alarms"] == 0, d


def test_every_labelled_file_is_found(full):
    for d, r in full["detectors"].items():
        assert r["hits"] == r["labelled"], d


def test_labelled_set_sizes(full):
    det = full["detectors"]
    assert det["sd_watermark"]["labelled"] >= 6
    assert det["c2pa"]["labelled"] >= 8
    assert det["metadata"]["labelled"] >= 7
    if "trustmark" in det:
        assert det["trustmark"]["labelled"] >= 5


def test_c2pa_survival_is_honest(full):
    s = full["survival"]["c2pa"]
    for col in ("JPEG q75, metadata kept", "50% resize, metadata kept"):
        assert s[col]["found"] == 0  # changed pixels never pass as intact
        assert s[col]["says_changed"] == s[col]["of"] > 0
    for col in ("JPEG q75, metadata dropped", "50% resize, metadata dropped", "screenshot-like"):
        assert s[col]["found"] == s[col]["says_changed"] == 0
