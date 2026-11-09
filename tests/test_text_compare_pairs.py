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
    # AI texts first, then people's, the way the data file is laid out.
    _write(tmp_path, [{"id": i, "source": "beemo", "attack": "none", "split": "newmodels",
                       "text": long, "label": 1 if i < 50 else 0} for i in range(100)])
    monkeypatch.setattr(cp, "DATA", tmp_path)
    got = cp.sample({"beemo"}, 10)
    assert [r["label"] for r in got] == [1, 0] * 5  # alternates, so a partial run has both
    assert [r["id"] for r in got] == [r["id"] for r in cp.sample({"beemo"}, 10)]
    assert len({r["id"] for r in got}) == 10


def test_pair_file_uses_short_name_when_given():
    from types import SimpleNamespace
    assert cp.pair_file(SimpleNamespace(name="qwen1.5b", performer="/a/b")).name == "pair_qwen1.5b.jsonl"
    assert cp.pair_file(SimpleNamespace(name=None, performer="Qwen/X")).name == "pair_Qwen__X.jsonl"


def test_plain_rows_file_gets_ids_and_is_held_out(tmp_path):
    long = " ".join(["word"] * 120)
    p = tmp_path / "rows.jsonl"
    with open(p, "w") as fh:
        for i in range(6):
            fh.write(json.dumps({"source": "detectrlx", "model": "gpt-4o" if i % 2 else "human",
                                 "label": i % 2, "doc_id": f"d{i // 2}", "text": long}) + "\n")
    got = cp.sample({"detectrlx"}, 6, path=p)
    assert len(got) == 6 and len({r["id"] for r in got}) == 6
    assert got[0]["id"] == "d0:1:gpt-4o" or got[0]["label"] == 1


def test_report_against_another_pair(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    long = " ".join(["word"] * 120)
    rows = tmp_path / "rows.jsonl"
    with open(rows, "w") as fh:
        for i in range(4):
            fh.write(json.dumps({"source": "detectrlx", "model": "gpt-4o" if i % 2 else "human",
                                 "label": i % 2, "doc_id": f"d{i}", "text": long}) + "\n")
    monkeypatch.setattr(cp, "DATA", tmp_path)
    ids = [r["id"] for r in cp.sample({"detectrlx"}, 4, path=rows)]
    labels = {r["id"]: r["label"] for r in cp.sample({"detectrlx"}, 4, path=rows)}
    for name, good in (("shipped", False), ("big", True)):
        with open(tmp_path / f"pair_{name}.jsonl", "w") as fh:
            for i in ids:
                # lower score = more AI-like; the "big" pair separates them perfectly
                bino = (0.5 if labels[i] else 1.5) if good else 1.0
                fh.write(json.dumps({"id": i, "bino": bino}) + "\n")
    cp.report(SimpleNamespace(name="big", performer="x", against="shipped", source="detectrlx",
                              limit=4, rows=rows, json_out=str(tmp_path / "r.json")))
    out = capsys.readouterr().out
    assert "4 texts scored by both pairs (2 AI, 2 people's)" in out
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved["groups"]["all"] == {"n": 4, "shipped_pair": 0.5, "new_pair": 1.0}
    assert "shipped pair 0.500  new pair 1.000" in out
