"""scripts/measure_text.py helpers: fast, offline (no data, no models)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
mt = pytest.importorskip("measure_text")


def row(**k):
    r = {"split": "x", "label": 1, "text": "t", "model": "m", "kind": "news", "attack": "none",
         "doc_id": "d1", "source": "raid"}
    r.update(k)
    return r


def test_same_model_one_name():
    assert mt.norm_model("gpt4") == mt.norm_model("gpt-4") == "gpt-4"
    assert mt.norm_model(None) == "unknown"


def test_reworded_sets_are_named_plainly():
    out = mt.assign([row(attack="paraphrase", doc_id=str(i)) for i in range(20)], ("hc3", "mage", "raid"))
    assert {r["set"] for r in out} == {"raid_reworded"}
    # reworded texts are never fitted on unless asked
    assert all(r["side"] in ("test", "unused") for r in out)


def test_newer_models_held_out_unless_asked():
    rows = [row(source="detectrlx", split="newmodels", doc_id=str(i)) for i in range(40)]
    assert all(r["side"] == "test" for r in mt.assign(rows, ("hc3", "mage", "raid")))
    sides = {r["side"] for r in mt.assign(rows, ("hc3", "mage", "raid", "newer"))}
    assert sides == {"train", "test"}


def test_raid_split_keeps_a_document_on_one_side():
    rows = [row(doc_id=str(i // 3), label=i % 2) for i in range(90)]
    out = mt.assign(rows, ("hc3", "mage", "raid"))
    side = {}
    for r in out:
        assert side.setdefault(r["doc_id"], r["side"]) == r["side"]


def _res(overall, kinds, sets=None):
    return {"overall": {"people_above_band": overall, "ai_above_band": 0.5},
            "by_kind": {k: {"people_above_band": v, "n_people": 100} for k, v in kinds.items()},
            "people_only": {k: {"people_above_band": v, "n_people": 50} for k, v in (sets or {}).items()},
            "sets": {}, "by_model": {}, "reworded": {}}


def test_ship_rule_passes_when_nothing_gets_worse():
    old = _res(0.008, {"news": 0.02, "qa": 0.01}, {"toefl": 0.03})
    new = _res(0.006, {"news": 0.02, "qa": 0.0}, {"toefl": 0.01})
    assert mt.compare(old, new)["ship"]


def test_ship_rule_fails_on_one_worse_kind_or_high_overall():
    old = _res(0.008, {"news": 0.02, "qa": 0.01})
    assert mt.compare(old, _res(0.005, {"news": 0.03, "qa": 0.0}))["worse"] == ["kind: news"]
    assert not mt.compare(old, _res(0.02, {"news": 0.0, "qa": 0.0}))["ship"]
