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
