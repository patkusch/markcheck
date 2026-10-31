import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WM = ROOT / "tests" / "fixtures" / "watermark"


def _measure_module():
    spec = importlib.util.spec_from_file_location("measure", ROOT / "scripts" / "measure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_measure_counts_hits_alarms_and_survival(tmp_path):
    d = tmp_path / "set"
    d.mkdir()
    shutil.copy(WM / "sd_v2_marked.png", d)
    shutil.copy(WM / "unmarked_1.png", d)
    (d / "manifest.json").write_text(json.dumps({"files": [
        {"file": "sd_v2_marked.png", "expect": ["sd_watermark"], "note": "marked"},
        {"file": "unmarked_1.png", "expect": [], "note": "control"},
    ]}))
    m = _measure_module()
    res = m.measure(tmp_path)
    assert res["files"] == 2 and res["controls"] == 1
    sd = res["detectors"]["sd_watermark"]
    assert (sd["hits"], sd["labelled"], sd["false_alarms"]) == (1, 1, 0)
    surv = res["survival"]["sd_watermark"]
    assert surv["original"] == {"found": 1, "of": 1}
    assert set(surv) == set(m.VARIANTS)
    table = m.tables(res)
    assert "| sd_watermark |" in table and "JPEG q75" in table
