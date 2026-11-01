"""`markcheck text`: fast, offline tests with a fake scorer (no language model is loaded)."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import sys

import pytest

import markcheck.textcheck as tc
from markcheck.cli import main
from markcheck.textcheck import CAVEAT, band_of, check_text, load_calibration
from markcheck.textcheck import scorer as scorer_mod
from markcheck.textcheck.features import normalise, sentences
from markcheck.textcheck.scorer import MissingModels, TokenStats
from markcheck.textcheck.scorer import PairScorer as REAL_PAIR_SCORER

HEADLINE = re.compile(r"Estimated chance this was AI-written: (\d+)% "
                      r"\(uncertain band (\d+)%-(\d+)% means we can't tell\)\.")

# A calibration we control, so the bands do not depend on the numbers the lead is still fitting.
TEST_CAL = {
    "models": ["obs/model", "perf/model"],
    "features": ["bino", "top10", "spread"],
    "mean": [0.8, 0.7, 0.5],
    "std": [0.1, 0.1, 0.2],
    "coef": [-3.0, 0.0, 0.0],
    "intercept": 0.0,
    "fill": {"spread": 0.5},
    "uncertain_band": [0.35, 0.65],
    "min_words": 100,
    "full_confidence_words": 150,
    "reference": {"top10": {"ai": 0.78, "person": 0.62},
                  "bino": {"ai": 0.62, "person": 0.92},
                  "spread": {"ai": 0.38, "person": 0.55}},
    "heldout": {"how_often_wrong": {
        "people's texts wrongly put above the band": "3 in 100",
        "AI texts put below the band": "8 in 100",
        "placeholder": False,
    }},
    "data_note": "Fitted on public example texts; checked on texts kept aside.",
}

SENTENCE_BANK = [
    "The committee reviewed the annual budget and approved most of the proposed changes.",
    "My grandmother kept chickens behind the shed and named every single one after a saint.",
    "Overall, the results demonstrate a clear and consistent improvement across all key metrics.",
    "We missed the last bus, so we walked home in the rain, laughing at nothing much.",
    "In conclusion, effective communication remains essential for successful teamwork in any setting.",
    "The kettle broke on Tuesday and nobody noticed until the guests arrived for tea.",
]


def long_text(n_sentences: int = 18) -> str:
    return " ".join(SENTENCE_BANK[i % len(SENTENCE_BANK)] for i in range(n_sentences))


SHORT = "This is far too short to say anything about."


class FakeScorer:
    """Builds TokenStats from the text itself: one 'token' per run of non-spaces, offsets aligned.

    Even-numbered sentences get predictable tokens (log-prob `ll_easy`), odd ones surprising
    tokens (`ll_hard`), so sentence effects point both ways. `bino` sets the overall
    Binoculars score by scaling the cross-surprise.
    """

    def __init__(self, bino: float = 0.8, top10: float = 0.7, ll_easy: float = -1.5, ll_hard: float = -3.5):
        self.bino, self.top10, self.ll_easy, self.ll_hard = bino, top10, ll_easy, ll_hard
        self.calls = 0

    def score(self, text: str) -> TokenStats:
        self.calls += 1
        spans = sentences(text)
        offsets, ll = [], []
        for m in re.finditer(r"\S+", text):
            idx = next((i for i, (a, b) in enumerate(spans) if a <= m.start() < b), 0)
            offsets.append((m.start(), m.end()))
            ll.append(self.ll_easy if idx % 2 == 0 else self.ll_hard)
        logppl = -sum(ll) / len(ll)
        xent_val = logppl / self.bino
        return TokenStats(n_tokens=len(ll), logppl_obs=logppl * 1.1, logppl_perf=logppl, xent=xent_val,
                          fdg=1.0, top10=self.top10, token_ll=ll, token_xent=[xent_val] * len(ll),
                          offsets=offsets)


def run(args, stdin=None):
    out, err = io.StringIO(), io.StringIO()
    old = sys.stdin
    if stdin is not None:
        sys.stdin = stdin
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(args)
    finally:
        sys.stdin = old
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def fake(monkeypatch):
    """Use TEST_CAL and a FakeScorer for every PairScorer the CLI makes."""
    holder = {"scorer": FakeScorer()}
    monkeypatch.setattr(tc, "load_calibration", lambda: copy.deepcopy(TEST_CAL))
    monkeypatch.setattr(scorer_mod, "PairScorer", lambda *a, **k: holder["scorer"])
    return holder


# ---- too short ----

def test_too_short_refuses_without_a_number(fake):
    code, out, _ = run(["text", SHORT])
    assert code == 0
    assert out.startswith("Too short to estimate: 9 words.")
    assert "at least 100 words" in out
    assert "wrong too often" in out
    assert not re.search(r"\d+%", out)
    assert out.rstrip().endswith(CAVEAT)
    assert fake["scorer"].calls == 0  # the models are never asked


def test_too_short_json_has_no_probability(fake):
    code, out, _ = run(["text", SHORT, "--json"])
    doc = json.loads(out)
    assert code == 0
    assert doc["status"] == "too_short" and doc["probability_ai"] is None and doc["band"] is None
    assert doc["caveat"] == CAVEAT and doc["kind"] == "text_estimate"


def test_too_short_works_without_any_models(monkeypatch):
    """No scorer is built for short texts, so missing libraries do not matter."""
    def boom(*a, **k):
        raise MissingModels("not here")
    monkeypatch.setattr(scorer_mod, "PairScorer", boom)
    code, out, _ = run(["text", SHORT])
    assert code == 0 and "Too short" in out


# ---- scored output ----

def test_scored_output_layout(fake):
    code, out, _ = run(["text", long_text()])
    assert code == 0
    first = out.splitlines()[0]
    m = HEADLINE.match(first)
    assert m, first
    assert (m.group(2), m.group(3)) == ("35", "65")
    for heading in ("Why:", "Sentences that moved the estimate most:",
                    "How often this check is right and wrong (held-out tests):"):
        assert heading in out
    assert "- people's texts wrongly put above the band: 3 in 100" in out
    assert "- AI texts put below the band: 8 in 100" in out
    assert "placeholder" not in out
    assert "Note: Fitted on public example texts" in out
    assert out.rstrip().splitlines()[-1] == CAVEAT


def test_scored_output_with_shipped_calibration(monkeypatch):
    """Whatever numbers the shipped calibration holds, the layout and caveat stay the same."""
    monkeypatch.setattr(scorer_mod, "PairScorer", lambda *a, **k: FakeScorer())
    code, out, _ = run(["text", long_text()])
    assert code == 0
    assert HEADLINE.match(out.splitlines()[0])
    assert "How often this check is right and wrong (held-out tests):" in out
    assert out.rstrip().splitlines()[-1] == CAVEAT


@pytest.mark.parametrize("bino,band,tail", [
    (0.55, "likely_ai", "above the uncertain band"),
    (0.80, "uncertain", "inside the uncertain band"),
    (1.05, "unlikely_ai", "below the uncertain band"),
])
def test_each_band(fake, bino, band, tail):
    fake["scorer"] = FakeScorer(bino=bino)
    code, out, _ = run(["text", long_text(), "--json"])
    doc = json.loads(out)
    assert code == 0 and doc["band"] == band and tail in doc["headline"]
    code, out, _ = run(["text", long_text()])
    assert tail in out.splitlines()[0]


def test_band_edges():
    cal = TEST_CAL
    assert band_of(0.65, cal) == "likely_ai"
    assert band_of(0.6499, cal) == "uncertain"
    assert band_of(0.3501, cal) == "uncertain"
    assert band_of(0.35, cal) == "unlikely_ai"
    assert band_of(0.0, cal) == "unlikely_ai" and band_of(1.0, cal) == "likely_ai"


def test_headline_numbers_follow_probability(fake):
    fake["scorer"] = FakeScorer(bino=0.8)  # z = 0 -> 50%
    _, out, _ = run(["text", long_text()])
    assert HEADLINE.match(out).group(1) == "50"


def test_reasons_come_from_the_numbers():
    import copy
    from markcheck.textcheck import check_text
    cal = copy.deepcopy(TEST_CAL)
    cal["coef"] = [-3.0, 2.0, -1.0]
    a = check_text(long_text(), scorer=FakeScorer(bino=0.55, top10=0.85), cal=cal).reasons
    b = check_text(long_text(), scorer=FakeScorer(bino=1.05, top10=0.50), cal=cal).reasons
    assert a != b
    ja, jb = "\n".join(a), "\n".join(b)
    assert "very predictable" in ja and "85%" in ja and "raised the estimate" in ja
    assert "often surprising" in jb and "50%" in jb and "lowered the estimate" in jb
    assert "scored 0.55" in ja and "scored 1.05" in jb


def test_reasons_follow_the_fitted_weights(fake):
    """If the model gave a measure no weight, the reason says it hardly moved the estimate."""
    import copy
    cal = copy.deepcopy(TEST_CAL)
    from markcheck.textcheck import check_text
    cal["coef"] = [0.0 for _ in cal["coef"]]
    r = check_text(long_text(), scorer=FakeScorer(bino=0.55, top10=0.85), cal=cal)
    assert all("hardly moved the estimate" in x or "too few sentences" in x for x in r.reasons)


def test_sentences_listed_both_ways(fake):
    code, out, _ = run(["text", long_text(), "--json"])
    sents = json.loads(out)["sentences"]
    effects = {s["effect"] for s in sents}
    assert effects == {"raised the estimate", "lowered the estimate"}
    text = long_text()
    for s in sents:
        assert s["text"] in text
    raised = [s["text"] for s in sents if s["effect"] == "raised the estimate"]
    assert all(t in SENTENCE_BANK[0::2] for t in raised)  # the predictable sentences
    _, out, _ = run(["text", long_text()])
    assert f'- "{raised[0]}"' in out
    assert "Raised the estimate: its words were more predictable" in out


# ---- input ----

def test_file_input(fake, tmp_path):
    p = tmp_path / "essay.txt"
    p.write_text(long_text(), encoding="utf-8")
    code, out, _ = run(["text", "--file", str(p)])
    assert code == 0 and HEADLINE.match(out)


def test_missing_file_is_an_error(fake, tmp_path):
    code, out, err = run(["text", "--file", str(tmp_path / "nope.txt")])
    assert code == 2 and out == ""
    assert "could not read" in err and "missing" in err and "Traceback" not in err


def test_non_utf8_file_is_an_error(fake, tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"\xff\xfe\x00bad")
    code, _, err = run(["text", "--file", str(p)])
    assert code == 2 and "UTF-8" in err


def test_stdin_dash(fake):
    code, out, _ = run(["text", "-"], stdin=io.StringIO(long_text()))
    assert code == 0 and HEADLINE.match(out)


def test_stdin_piped_without_args(fake):
    code, out, _ = run(["text", "--json"], stdin=io.StringIO(long_text()))
    assert code == 0 and json.loads(out)["status"] == "scored"


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_no_text_at_all_is_an_error(fake):
    code, out, err = run(["text"], stdin=_Tty(""))
    assert code == 2 and "no text was given" in err


@pytest.mark.parametrize("args,stdin", [(["text", "   \n "], None), (["text", "-"], io.StringIO("  "))])
def test_empty_text_is_an_error(fake, args, stdin):
    code, _, err = run(args, stdin=stdin)
    assert code == 2 and "empty" in err


def test_text_and_file_together_is_an_error(fake, tmp_path):
    code, _, err = run(["text", "hello", "--file", str(tmp_path / "x")])
    assert code == 2 and "not both" in err


# ---- JSON ----

def test_json_structure(fake):
    code, out, _ = run(["text", long_text(), "--json"])
    doc = json.loads(out)
    assert code == 0
    from markcheck import __version__
    assert doc["version"] == __version__ and doc["kind"] == "text_estimate"
    assert doc["status"] == "scored"
    assert set(doc) >= {"words", "probability_ai", "band", "headline", "reasons", "sentences",
                        "measures", "how_often_wrong", "notes", "caveat"}
    assert 0 <= doc["probability_ai"] <= 1
    assert doc["how_often_wrong"]["AI texts put below the band"] == "8 in 100"
    assert {"tokens", "binoculars", "top10_share"} <= set(doc["measures"])


# ---- missing libraries or models ----

def test_missing_models_exit_2(fake, monkeypatch):
    msg = "The language models for the text check are not on this computer yet."

    def boom(*a, **k):
        raise MissingModels(msg)
    monkeypatch.setattr(scorer_mod, "PairScorer", boom)
    code, out, err = run(["text", long_text()])
    assert code == 2 and out == ""
    assert msg in err and "Traceback" not in err
    code, out, err = run(["text", long_text(), "--json"])
    doc = json.loads(out)
    assert code == 2 and doc["status"] == "error" and doc["probability_ai"] is None
    assert doc["error"] == msg and doc["caveat"] == CAVEAT


def test_missing_libraries_give_plain_message(fake, monkeypatch):
    """The real PairScorer, with torch and transformers hidden: plain message, exit 2."""
    monkeypatch.setattr(scorer_mod, "PairScorer", REAL_PAIR_SCORER)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    code, out, err = run(["text", long_text()])
    assert code == 2 and out == ""
    assert "pip install 'markcheck[text]'" in err and "Traceback" not in err


def test_other_load_errors_give_plain_message(fake, monkeypatch):
    def boom(*a, **k):
        raise OSError("weights corrupt")
    monkeypatch.setattr(scorer_mod, "PairScorer", boom)
    code, _, err = run(["text", long_text()])
    assert code == 2 and "fetch_models --text" in err and "Traceback" not in err


# ---- main CLI and helpers ----

def test_main_help_mentions_text_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "markcheck text" in capsys.readouterr().out


def test_text_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["text", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--file" in out and "fetch_models --text" in out


def test_file_cli_unchanged(fixtures):
    code, out, _ = run([str(fixtures)])
    assert code in (0, 1) and "==" in out


def test_normalise_stray_spaces():
    assert normalise("Hello , world .  It  works ( mostly ) !") == "Hello, world. It works (mostly)!"
    assert normalise("a\r\nb") == "a\nb"
    assert normalise("  padded  ") == "padded"
    assert normalise("50 %") == "50%"


def test_normalised_text_scores_the_same(fake):
    a = json.loads(run(["text", long_text(), "--json"])[1])
    messy = long_text().replace(".", " .").replace(",", " ,")
    b = json.loads(run(["text", messy, "--json"])[1])
    assert a["probability_ai"] == b["probability_ai"] and a["words"] == b["words"]


def test_sentences_splitting():
    t = "First one here. Second one! Third? \"Quoted\" fourth. 4 cats.\n\nnew para"
    got = [t[a:b] for a, b in sentences(t)]
    assert got == ["First one here.", "Second one!", "Third?", "\"Quoted\" fourth.", "4 cats.", "new para"]
    u = "e.g. lower case stays together."
    assert [u[a:b] for a, b in sentences(u)] == [u]
    assert sentences("   ") == []


def test_check_text_with_scorer_argument():
    res = check_text(long_text(), scorer=FakeScorer(), cal=copy.deepcopy(TEST_CAL))
    assert res.status == "scored" and res.caveat == CAVEAT


def test_shipped_calibration_has_the_keys_the_cli_needs():
    cal = load_calibration()
    for key in ("models", "features", "mean", "std", "coef", "intercept", "uncertain_band",
                "min_words", "reference", "heldout", "data_note"):
        assert key in cal
    assert len(cal["models"]) == 2
    assert isinstance(cal["heldout"]["how_often_wrong"], dict)


# ---- fetch_models --text ----

def test_fetch_text_models(monkeypatch, capsys):
    import types

    from markcheck import fetch_models
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kw: calls.append(kw) or f"/cache/{kw['repo_id']}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert fetch_models.main(["--text"]) == 0
    assert [c["repo_id"] for c in calls] == load_calibration()["models"]
    for c in calls:
        assert set(c["allow_patterns"]) == {"*.json", "*.safetensors", "tokenizer*", "*.txt"}
        assert c["ignore_patterns"] == ["*/*"]
    assert "Done." in capsys.readouterr().out


def test_fetch_text_models_failure_and_missing_library(monkeypatch, capsys):
    import types

    from markcheck import fetch_models
    hub = types.ModuleType("huggingface_hub")

    def fail(**kw):
        raise OSError("no network")
    hub.snapshot_download = fail
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert fetch_models.main(["--text"]) == 2
    assert "could not be fetched" in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert fetch_models.main(["--text"]) == 2
    assert "pip install 'markcheck[text]'" in capsys.readouterr().out


def test_fetch_models_default_is_still_trustmark(monkeypatch, capsys):
    from markcheck import fetch_models
    monkeypatch.setitem(sys.modules, "trustmark", None)
    assert fetch_models.main([]) == 2
    assert "trustmark" in capsys.readouterr().out
