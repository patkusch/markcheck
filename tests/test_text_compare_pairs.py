"""scripts/text_compare_pairs.py helpers: fast, offline (no data, no models)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
cp = pytest.importorskip("text_compare_pairs")


def test_auroc_known_values():
    assert cp.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert cp.auroc([0, 1], [1, 0]) == 0.0
    assert cp.auroc([1, 1], [1, 0]) == 0.5


def test_auroc_needs_both_kinds_of_text():
    assert math.isnan(cp.auroc([1, 2], [1, 1]))
    assert math.isnan(cp.auroc([1, 2], [0, 0]))


def _write(tmp_path, rows):
    with open(tmp_path / "splits.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_sample_keeps_only_held_out_unreworded_long_texts(tmp_path, monkeypatch):
    long = " ".join(["word"] * 120)
    base = {"source": "beemo", "attack": "none", "split": "newmodels", "text": long, "label": 1}
    _write(tmp_path, [
        {**base, "id": 0},
        {**base, "id": 1, "split": "train"},
        {**base, "id": 2, "attack": "paraphrase"},
        {**base, "id": 3, "text": "too short"},
        {**base, "id": 4, "source": "hc3"},
    ])
    monkeypatch.setattr(cp, "DATA", tmp_path)
    assert [r["id"] for r in cp.sample({"beemo"}, 10)] == [0]


def test_sample_is_evenly_spread_and_repeatable(tmp_path, monkeypatch):
    long = " ".join(["word"] * 120)
    _write(tmp_path, [{"id": i, "source": "beemo", "attack": "none", "split": "newmodels",
                       "text": long, "label": i % 2} for i in range(100)])
    monkeypatch.setattr(cp, "DATA", tmp_path)
    ids = [r["id"] for r in cp.sample({"beemo"}, 10)]
    assert ids == list(range(0, 100, 10))
    assert ids == [r["id"] for r in cp.sample({"beemo"}, 10)]


def test_pair_file_uses_short_name_when_given():
    from types import SimpleNamespace
    assert cp.pair_file(SimpleNamespace(name="qwen1.5b", performer="/a/b")).name == "pair_qwen1.5b.jsonl"
    assert cp.pair_file(SimpleNamespace(name=None, performer="Qwen/X")).name == "pair_Qwen__X.jsonl"
