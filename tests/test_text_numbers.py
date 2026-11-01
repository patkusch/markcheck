"""No accuracy number shown to a reader can drift from the stored results.

The README's text-accuracy block must be exactly what markcheck.textcheck.accuracy renders
from the shipped calibration file, and the CLI must print every held-out rate verbatim.
Regenerate the README block with: python -m markcheck.textcheck.accuracy --write-readme
"""
from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import pytest

import markcheck.textcheck as tc
from markcheck.textcheck import _pcts, accuracy, load_calibration
from markcheck.textcheck import scorer as scorer_mod

from test_text import TEST_CAL, FakeScorer, long_text, run

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _text_section(readme: str) -> str:
    start = readme.index("## Text check")
    end = readme.index("\n## ", start + 1)
    return readme[start:end]


def test_readme_block_matches_calibration():
    block = accuracy.render(load_calibration())
    assert block in _readme(), ("README accuracy block is out of date; run "
                                "python -m markcheck.textcheck.accuracy --write-readme")


def test_readme_block_appears_once():
    r = _readme()
    assert r.count(accuracy.START) == 1 and r.count(accuracy.END) == 1


def test_readme_band_and_minimum_match_calibration():
    cal = load_calibration()
    sec = _text_section(_readme())
    _, slo, shi = _pcts(0.5, *cal["uncertain_band"])
    for band in re.findall(r"uncertain band (\d+%-\d+%)", sec):
        assert band == f"{slo}-{shi}"
    assert f"at least {cal['min_words']} words" in sec


def test_no_rates_outside_generated_block():
    """Rates in the text check section live only in the generated block."""
    sec = _text_section(_readme())
    a, b = sec.index(accuracy.START), sec.index(accuracy.END)
    outside = sec[:a] + sec[b:]
    assert not re.search(r"\d+\s+in\s+100", outside)
    # The only percentages outside are the example headline (a made-up chance plus the band).
    for line in outside.splitlines():
        if re.search(r"\d%", line):
            assert "Estimated chance this was AI-written" in line, line


def test_every_how_often_row_rendered_verbatim():
    cal = load_calibration()
    block = accuracy.render(cal)
    rows = accuracy.how_often_rows(cal)
    assert rows, "the shipped calibration has no held-out rates"
    for k, v in rows:
        assert f"| {k.replace('|', '/')} | {v.replace('|', '/')} |" in block


def test_cli_prints_every_rate_verbatim(monkeypatch):
    """Human output carries each how_often_wrong line from the shipped file, unchanged."""
    monkeypatch.setattr(scorer_mod, "PairScorer", lambda *a, **k: FakeScorer())
    cal = load_calibration()
    _, out, _ = run(["text", long_text()])
    for k, v in cal["heldout"]["how_often_wrong"].items():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            assert f"- {k}: {v}" in out
    _, jout, _ = run(["text", long_text(), "--json"])
    assert json.loads(jout)["how_often_wrong"] == cal["heldout"]["how_often_wrong"]


def test_render_survives_missing_and_odd_keys():
    cal = copy.deepcopy(TEST_CAL)
    assert accuracy.START in accuracy.render(cal)
    cal["heldout"] = {}
    assert "Not measured yet." in accuracy.render(cal)
    cal["heldout"] = {"how_often_wrong": {"group": {"inner": "2 in 100"}, "flag": True, "x": "1 in 100"}}
    cal["fairness"] = {"groups": {"A": {"flagged": {"k": 1, "n": 50, "rate": 0.02, "ci95": [0.004, 0.105]},
                                         "cant_tell": None}, "B": "junk"},
                       "band_tradeoff": [{"band_top": 0.9, "native_flagged": {"n": 0}}, {"oops": 1}]}
    block = accuracy.render(cal)
    assert "| group: inner | 2 in 100 |" in block and "flag" not in block
    assert "| A | 50 | 2 in 100 (likely 0.4 to 10) | - |" in block
    assert "| 90% | - | - |" in block


def test_rate_words():
    assert accuracy.rate_words({"k": 0, "n": 63, "rate": 0.0, "ci95": [0.0, 0.057]}) == "0 in 100 (likely 0 to 6)"
    assert accuracy.rate_words({"n": 0}) == "-"
    assert accuracy.rate_words(None) == "-"


def test_replace_block():
    old = f"a\n{accuracy.START}\nold\n{accuracy.END}\nb"
    assert accuracy.replace_block(old, "NEW") == "a\nNEW\nb"
    with pytest.raises(ValueError):
        accuracy.replace_block("no markers", "x")


def test_fairness_numbers_match_shipped_weights():
    """If fairness results are stored, they were worked out with the shipped weights and band."""
    cal = load_calibration()
    fair = cal.get("fairness")
    if not fair:
        pytest.skip("no fairness results stored yet")
    cw = fair["computed_with"]
    assert cw["uncertain_band"] == cal["uncertain_band"]
    assert cw["min_words"] == cal["min_words"]
    assert cw["intercept"] == pytest.approx(cal["intercept"])
    assert cw["coef"] == pytest.approx(cal["coef"])


def test_no_hard_coded_accuracy_numbers_in_source():
    """Accuracy claims come from the calibration file, never from string literals in the code."""
    pat = re.compile(r"\d+\s*in\s*100|\d+(\.\d+)?\s*%")
    for path in sorted(Path(tc.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if id(node) in docs:
                continue  # notes for developers, not shown to readers
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.replace("95% confidence", "")  # how sure an interval is, not an accuracy claim
                assert not pat.search(value), f"{path.name}: {node.value!r}"
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        assert not re.search(r"\d+\s*in\s*100", part.value), f"{path.name}: {part.value!r}"
