"""`markcheck text` wording: an estimate, never a verdict about a person.

Every output branch (too short, each band, missing models) in text and JSON,
every string literal in src/markcheck/textcheck/*.py, and the calibration
file are scanned for banned words and for tips on changing a text's score.
"""
from __future__ import annotations

import ast
import copy
import json
import re
from importlib import resources
from pathlib import Path

import pytest

import markcheck.textcheck as tc
from markcheck.textcheck import CAVEAT
from markcheck.textcheck import scorer as scorer_mod
from markcheck.textcheck.scorer import MissingModels

from test_text import SHORT, TEST_CAL, FakeScorer, long_text, run
from test_wording import BANNED as FILE_BANNED
from test_wording import _all_strings

TEXT_BANNED = re.compile(
    r"(human[- ]written|written by a human|\bproof\b|\bproves?\b|plagiari|cheat)", re.I)

# Advice on how to make a text score differently. The check must never coach anyone.
TIPS = re.compile(
    r"\b(tips?|try|trying|rewrit\w*|rephras\w*|paraphras\w*|humani[sz]\w*|bypass\w*|evad\w*|evasion|"
    r"fool\w*|trick\w*|avoid\w*|to get a lower|to lower the|to reduce the (estimate|score))\b",
    re.I)

SRC = Path(tc.__file__).parent


def hits(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if FILE_BANNED.search(line) or TEXT_BANNED.search(line) or TIPS.search(line)]


def json_hits(doc) -> list[str]:
    return hits("\n".join(_all_strings(doc)))


def test_regex_catches_the_obvious():
    assert hits("this is human-written")
    assert hits("Written by a human, clearly")
    assert hits("proof of cheating")
    assert hits("This proves it")
    assert hits("plagiarism found")
    assert hits("Not AI")
    assert hits("authentic voice")
    assert hits("a real person")
    assert hits("Tip: rewrite the long sentences")
    assert hits("try adding typos")
    assert not hits("Estimated chance this was AI-written: 40%")
    assert not hits("people's text")


def test_caveat_says_what_it_must():
    c = CAVEAT.lower()
    assert "estimate" in c
    assert "wrong sometimes" in c
    assert re.search(r"not (be )?the only basis", c)
    assert "not their first" in c or "non-native" in c
    assert hits(CAVEAT) == []


@pytest.fixture
def fake(monkeypatch):
    holder = {"scorer": FakeScorer()}
    monkeypatch.setattr(tc, "load_calibration", lambda: copy.deepcopy(TEST_CAL))
    monkeypatch.setattr(scorer_mod, "PairScorer", lambda *a, **k: holder["scorer"])
    return holder


def _both(args):
    _, out, err = run(args)
    _, jout, jerr = run(args + ["--json"])
    return out + "\n" + err + "\n" + jerr, json.loads(jout)


def test_too_short_wording(fake):
    text, doc = _both(["text", SHORT])
    assert hits(text) == [] and json_hits(doc) == []
    assert CAVEAT in text


@pytest.mark.parametrize("bino", [0.55, 0.80, 1.05])
def test_each_band_wording(fake, bino):
    fake["scorer"] = FakeScorer(bino=bino, top10={0.55: 0.85, 0.80: 0.7, 1.05: 0.5}[bino])
    text, doc = _both(["text", long_text()])
    assert hits(text) == [] and json_hits(doc) == []


def test_shipped_calibration_output_wording(monkeypatch):
    monkeypatch.setattr(scorer_mod, "PairScorer", lambda *a, **k: FakeScorer())
    text, doc = _both(["text", long_text()])
    assert hits(text) == [] and json_hits(doc) == []


def test_missing_model_wording(fake, monkeypatch):
    def boom(*a, **k):
        # Same text the real PairScorer raises.
        raise MissingModels("The language models for the text check are not on this computer yet. "
                            "Fetch them once with: python -m markcheck.fetch_models --text")
    monkeypatch.setattr(scorer_mod, "PairScorer", boom)
    text, doc = _both(["text", long_text()])
    assert hits(text) == [] and json_hits(doc) == []


def test_error_branches_wording(fake, tmp_path):
    for args in (["text", "--file", str(tmp_path / "missing.txt")], ["text", "  "]):
        text, doc = _both(args)
        assert hits(text) == [] and json_hits(doc) == []


def _string_literals(path: Path) -> list[str]:
    return [n.value for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_source_string_literals(path):
    assert hits("\n".join(_string_literals(path))) == []


def test_source_scan_finds_files():
    names = {p.name for p in SRC.glob("*.py")}
    assert {"__init__.py", "cli.py", "features.py", "scorer.py"} <= names


def test_calibration_file_wording():
    raw = resources.files("markcheck").joinpath("data/text_calibration.json").read_text(encoding="utf-8")
    assert json_hits(json.loads(raw)) == []


def test_help_wording(capsys):
    from markcheck.cli import main
    with pytest.raises(SystemExit):
        main(["text", "--help"])
    assert hits(capsys.readouterr().out) == []
