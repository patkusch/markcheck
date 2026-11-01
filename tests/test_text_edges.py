"""`markcheck text` edge cases found in review: token alignment, sentence rules, odd inputs."""
from __future__ import annotations

import copy
import io
import json
import math
import re

import pytest

import markcheck.textcheck as tc
from markcheck.textcheck import check_text, headline, pct, reasons_for, sentence_effects
from markcheck.textcheck import cli as text_cli
from markcheck.textcheck.features import normalise, sentences, words
from markcheck.textcheck.scorer import MAX_TOKENS, PairScorer, TokenStats

from test_text import TEST_CAL, FakeScorer, fake, long_text, run  # noqa: F401 (fixture)
from test_text_wording import hits, json_hits

torch = pytest.importorskip("torch")


# ---- scorer: start token, offsets and the "truncated" flag (no real model) ----

class FakeTok:
    """One word piece per run of non-spaces, carrying the space before it, like GPT-2 style tokenizers."""

    def __init__(self, bos=7, adds_bos=False):
        self.bos_token_id, self.adds_bos = bos, adds_bos

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, truncation=False,
                 max_length=None):
        spans = [m.span() for m in re.finditer(r"\s*\S+", text)]
        ids = [10 + i % 20 for i in range(len(spans))]
        if self.adds_bos:
            ids, spans = [self.bos_token_id] + ids, [(0, 0)] + spans
        if truncation and max_length:
            ids, spans = ids[:max_length], spans[:max_length]
        if return_tensors != "pt":
            return {"input_ids": ids}
        out = {"input_ids": torch.tensor([ids], dtype=torch.long).reshape(1, len(ids))}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor([spans], dtype=torch.long).reshape(1, len(spans), 2)
        return out


class FakeLM:
    def __call__(self, ids):
        class Out:
            logits = torch.zeros(1, ids.shape[1], 40)
        return Out()


def fake_pair(**tok_kw) -> PairScorer:
    sc = object.__new__(PairScorer)
    sc.torch, sc.tok, sc.obs, sc.perf = torch, FakeTok(**tok_kw), FakeLM(), FakeLM()
    return sc


def n_words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_start_token_added_once_and_offsets_line_up():
    text = n_words(20)
    for kw in ({}, {"adds_bos": True}):
        st = fake_pair(**kw).score(text)
        assert st.n_tokens == 20, kw  # every word scored, start token not doubled
        assert [text[s:e].strip() for s, e in st.offsets] == text.split()
        assert len(st.offsets) == len(st.token_ll) == len(st.token_xent) == 20


def test_no_start_token_tokenizer():
    text = n_words(20)
    st = fake_pair(bos=None).score(text)
    assert st.n_tokens == 19  # the first word has nothing before it to be predicted from
    assert [text[s:e].strip() for s, e in st.offsets] == text.split()[1:]


@pytest.mark.parametrize("kw", [{}, {"adds_bos": True}, {"bos": None}], ids=["bos", "adds-bos", "no-bos"])
def test_truncated_only_when_pieces_were_dropped(kw):
    sc = fake_pair(**kw)
    fits = MAX_TOKENS + 1 - (1 if kw.get("adds_bos") else 0)
    assert not sc.score(n_words(fits)).truncated
    long = sc.score(n_words(fits + 50))
    assert long.truncated and long.n_tokens <= MAX_TOKENS + 1


def test_nothing_to_score_is_a_plain_error():
    with pytest.raises(ValueError):
        fake_pair(bos=None).score("one")


# ---- sentences, words, normalise ----

@pytest.mark.parametrize("text, want", [
    ("Dr. Smith arrived. He sat down.", ["Dr. Smith arrived.", "He sat down."]),
    ("The U.S. Army left. Then it rained.", ["The U.S. Army left.", "Then it rained."]),
    ("Use one, e.g. The Times. Fine.", ["Use one, e.g. The Times.", "Fine."]),
    ("See Fig. 3 for this. Mr. Jones agreed.", ["See Fig. 3 for this.", "Mr. Jones agreed."]),
    ("# Title\nThe body text.", ["# Title", "The body text."]),
    ("Things to do:\n- buy milk\n- walk dog\n1. call mum", ["Things to do:", "- buy milk", "- walk dog", "1. call mum"]),
    ("Done.\n- next item", ["Done.", "- next item"]),
    ("Él vino tarde. Über alles.", ["Él vino tarde.", "Über alles."]),
    ("no punctuation here at all", ["no punctuation here at all"]),
    ("Line one wraps\nonto line two. Next.", ["Line one wraps\nonto line two.", "Next."]),
    ("Really?! Yes.", ["Really?!", "Yes."]),
])
def test_sentence_rules(text, want):
    assert [text[a:b] for a, b in sentences(text)] == want


def test_words_count_numbers_hyphens_and_other_alphabets():
    assert words("It cost 3.5 or 1,000 euros, a well-known café. Größe.") == [
        "It", "cost", "3.5", "or", "1,000", "euros", "a", "well-known", "café", "Größe"]
    assert len(words("Привет мир, как дела")) == 4
    assert words("-- x -") == ["x"]


def test_normalise_keeps_meaning():
    assert normalise("We use the .NET runtime and a .bashrc file .") == \
        "We use the .NET runtime and a .bashrc file."
    assert normalise("old\rmac\r\nwindows") == "old\nmac\nwindows"
    assert normalise("Hello , world .") == "Hello, world."


def test_windows_line_endings_score_like_unix(fake):
    unix = "\n\n".join(long_text(6) for _ in range(3))
    seen = []

    class Spy(FakeScorer):
        def score(self, text):
            seen.append(text)
            return super().score(text)

    fake["scorer"] = Spy()
    a = json.loads(run(["text", unix, "--json"])[1])
    b = json.loads(run(["text", unix.replace("\n", "\r\n"), "--json"])[1])
    assert seen[0] == seen[1] and "\r" not in seen[1]
    assert a["probability_ai"] == b["probability_ai"] and a["sentences"] == b["sentences"]


# ---- sentence effects with word pieces that carry their leading space ----

def test_sentence_effects_count_the_first_word_of_each_sentence():
    text = "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa. Lambda mu nu xi omicron."
    spans = [m.span() for m in re.finditer(r"\s*\S+", text)]  # " Zeta" starts on the space
    ll = [-1.0] * 5 + [-3.0] * 5 + [-2.0] * 5
    st = TokenStats(n_tokens=15, logppl_obs=2.0, logppl_perf=1.5, xent=2.5, fdg=0.0, top10=0.8,
                    token_ll=ll, token_xent=[2.5] * 15, offsets=spans)
    got = sentence_effects(text, st, copy.deepcopy(TEST_CAL))
    assert {g["text"] for g in got} == {s.strip() for s in re.split(r"(?<=\.) ", text)}


# ---- percents and the headline ----

def test_pct_rounds_halves_up():
    assert pct(0.645) == "65%" and pct(0.125) == "13%" and pct(0.005) == "1%" and pct(0.996) == "99%"


@pytest.mark.parametrize("p, band", [(0.8301, [0.07, 0.83]), (0.5, [0.495, 0.504]), (0.996, [0.07, 0.995])])
def test_headline_never_shows_two_different_numbers_as_equal(p, band):
    cal = dict(TEST_CAL, uncertain_band=band)
    shown = re.findall(r"([\d.]+)%", headline(p, tc.band_of(p, cal), cal))[:3]
    assert len(set(shown)) == 3, shown


# ---- reasons agree with the direction they report ----

def test_predictability_reason_never_contradicts_itself():
    cal = copy.deepcopy(TEST_CAL)
    cal["coef"] = [-3.0, -10.0, 0.0]  # top10 pulls hard the other way from bino
    st = TokenStats(n_tokens=200, logppl_obs=2.0, logppl_perf=1.5, xent=2.5, fdg=0.0, top10=0.9)  # bino 0.6
    first = reasons_for(st, 0.4, cal)[0]
    assert "lowered" in first and "very predictable" not in first.split(":")[0]


# ---- calibration without optional keys ----

def test_calibration_without_optional_keys():
    cal = copy.deepcopy(TEST_CAL)
    for key in ("data_note", "heldout", "full_confidence_words"):
        cal.pop(key)
    res = check_text(long_text(), scorer=FakeScorer(), cal=cal)
    assert res.status == "scored" and res.error_rates == {} and res.notes == []


# ---- JSON stays valid when a measure is not a number ----

def test_json_has_no_nan(fake):
    class NanFdg(FakeScorer):
        def score(self, text):
            st = super().score(text)
            st.fdg = math.nan
            return st

    fake["scorer"] = NanFdg()
    code, out, _ = run(["text", long_text(), "--json"])

    def bad(c):
        raise AssertionError(f"{c} is not valid JSON")

    doc = json.loads(out, parse_constant=bad)
    assert code == 0 and doc["measures"]["fast_detectgpt"] is None


# ---- input files and standard input ----

class FirstPart(FakeScorer):
    """Scores only the start of a long text, as the real scorer does."""

    def score(self, text):
        st = super().score(text[:4000])
        st.truncated = len(text) > 4000
        return st


def test_big_file_is_cut_with_a_note(fake, tmp_path):
    fake["scorer"] = FirstPart()
    big = tmp_path / "big.txt"
    body = (long_text() + "\n\n") * (text_cli.MAX_BYTES // len(long_text()) + 5)
    big.write_text("é" + body, encoding="utf-8")  # the 2-byte "é" puts the cut inside a character
    big.write_bytes(big.read_bytes()[: text_cli.MAX_BYTES + 1] + "ü".encode() + b"tail")
    code, out, _ = run(["text", "--file", str(big), "--json"])
    doc = json.loads(out)
    assert code == 0 and doc["status"] == "scored" and doc["notes"][0] == text_cli.CUT_NOTE
    code, out, _ = run(["text", "--file", str(big)])
    assert "Note: " + text_cli.CUT_NOTE in out and hits(out) == []


def test_big_stdin_is_cut_with_a_note(fake):
    fake["scorer"] = FirstPart()
    code, out, _ = run(["text", "-", "--json"], stdin=io.StringIO("x " * text_cli.MAX_BYTES))
    assert code == 0 and text_cli.CUT_NOTE in json.loads(out)["notes"]


def test_binary_file_is_an_error(fake, tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01\x02 abc " * 50)
    code, out, err = run(["text", "--file", str(f), "--json"])
    assert code == 2 and "not plain text" in json.loads(out)["error"]


def test_folder_is_an_error(fake, tmp_path):
    code, out, err = run(["text", "--file", str(tmp_path), "--json"])
    doc = json.loads(out)
    assert code == 2 and doc["status"] == "error" and "folder" in doc["error"]


def test_unexpected_scoring_failure_gives_json(fake):
    class Broken:
        def score(self, text):
            raise RuntimeError("out of memory")

    fake["scorer"] = Broken()
    code, out, err = run(["text", long_text(), "--json"])
    doc = json.loads(out)
    assert code == 2 and doc["status"] == "error" and "out of memory" in doc["error"]
    assert "Traceback" not in err


def test_bad_option_with_json_gives_json(fake):
    code, out, err = run(["text", "--json", "--nope", long_text()])
    doc = json.loads(out)
    assert code == 2 and doc["status"] == "error" and "--nope" in doc["error"]


def test_bad_option_without_json_is_plain_usage(fake):
    with pytest.raises(SystemExit) as exc:
        run(["text", "--nope"])
    assert exc.value.code == 2


def test_truncation_note_wording(fake):
    class Long(FakeScorer):
        def score(self, text):
            st = super().score(text)
            st.truncated = True
            return st

    fake["scorer"] = Long()
    code, out, _ = run(["text", long_text(), "--json"])
    doc = json.loads(out)
    assert any("first part" in n for n in doc["notes"]) and json_hits(doc) == []
