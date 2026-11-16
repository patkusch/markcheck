"""The pair scorer's arithmetic, checked on tiny made-up models whose answers can be worked out by hand.

No language model is downloaded or loaded: the tokenizer and the two models are stand-ins that return
fixed numbers. Only torch itself is needed, and the tests skip when it is not installed.
"""
import math
import os
import sys
import types

import pytest

from markcheck.textcheck import scorer as scorer_mod
from markcheck.textcheck.scorer import MissingModels, PairScorer, TokenStats, finite

torch = pytest.importorskip("torch")

V = 4  # a vocabulary of four word pieces


class Out:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self, logits_for):
        self.logits_for = logits_for

    def __call__(self, ids):
        n = ids.shape[1]
        return Out(self.logits_for(ids).reshape(1, n, V))


class FakeTok:
    """Turns any text into `n` pieces numbered 0..n-1 (mod V). `extra` pretends the full text was longer."""

    def __init__(self, n, bos=None, extra=0, adds_bos=False):
        self.n, self.bos_token_id, self.extra, self.adds_bos = n, bos, extra, adds_bos

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, truncation=False, max_length=None):
        n = self.n
        ids = [i % V for i in range(n)]
        if self.adds_bos:
            ids = [self.bos_token_id] + ids
        if return_tensors is None:
            return {"input_ids": ids + [0] * self.extra}
        offsets = [(i, i + 1) for i in range(len(ids))]
        return {"input_ids": torch.tensor([ids]), "offset_mapping": torch.tensor([offsets])}


def build(tok, obs_logits, perf_logits=None):
    s = PairScorer.__new__(PairScorer)
    s.torch, s.tok = torch, tok
    s.obs = FakeModel(obs_logits)
    s.perf = FakeModel(perf_logits or obs_logits)
    return s


def certain(ids):
    """Always puts almost all the probability on the piece that really comes next."""
    n = ids.shape[1]
    logits = torch.full((n, V), -50.0)
    for pos in range(n - 1):
        logits[pos, int(ids[0, pos + 1])] = 50.0
    return logits


def flat(ids):
    return torch.zeros((ids.shape[1], V))


def test_a_model_that_is_never_surprised_scores_zero_surprise_and_a_perfect_top_ten():
    s = build(FakeTok(6, bos=9), certain)
    st = s.score("anything")
    assert st.n_tokens == 6  # six pieces plus the start token gives six scored predictions
    assert st.logppl_obs == pytest.approx(0, abs=1e-4)
    assert st.logppl_perf == pytest.approx(0, abs=1e-4)
    assert st.top10 == 1.0
    assert len(st.token_ll) == len(st.token_xent) == len(st.offsets) == 6
    assert st.truncated is False


def test_a_model_with_no_idea_is_exactly_as_surprised_as_chance_and_binoculars_is_one():
    s = build(FakeTok(5, bos=9), flat)
    st = s.score("anything")
    assert st.logppl_obs == pytest.approx(math.log(V), abs=1e-5)
    assert st.logppl_perf == pytest.approx(math.log(V), abs=1e-5)
    assert st.xent == pytest.approx(math.log(V), abs=1e-5)
    assert st.binoculars == pytest.approx(1.0, abs=1e-5)
    assert st.fdg == pytest.approx(0.0, abs=1e-3), "a flat model has no curvature to find"
    assert all(v == pytest.approx(-math.log(V), abs=1e-5) for v in st.token_ll)


def test_the_start_token_is_added_so_the_first_word_is_scored_and_its_span_is_dropped():
    s = build(FakeTok(3, bos=9), flat)
    st = s.score("abc")
    assert st.offsets == [(0, 1), (1, 2), (2, 3)], "one span per real word piece, none for the start token"
    assert st.n_tokens == 3


def test_a_tokenizer_that_already_adds_a_start_token_does_not_get_a_second_one():
    s = build(FakeTok(3, bos=9, adds_bos=True), flat)
    st = s.score("abc")
    assert st.n_tokens == 3  # four ids in total, three predictions
    assert len(st.offsets) == 3


def test_a_tokenizer_with_no_start_token_leaves_the_first_word_unscored():
    s = build(FakeTok(4, bos=None), flat)
    assert s.score("abcd").n_tokens == 3


def test_a_text_with_too_few_pieces_is_refused_with_a_plain_message():
    s = build(FakeTok(1, bos=None), flat)
    with pytest.raises(ValueError, match="too few word pieces"):
        s.score("a")


def test_the_result_says_when_the_text_was_cut_to_fit():
    s = build(FakeTok(4, bos=9, extra=10), flat)
    assert s.score("long").truncated is True


def test_a_performer_that_is_sure_of_something_the_observer_does_not_expect_raises_the_cross_surprise():
    def sure_of_piece_three(ids):
        logits = torch.zeros((ids.shape[1], V))
        logits[:, 3] = 20.0
        return logits

    agree = build(FakeTok(6, bos=9), flat, flat).score("x")
    disagree = build(FakeTok(6, bos=9), flat, sure_of_piece_three).score("x")
    assert disagree.logppl_obs == pytest.approx(agree.logppl_obs, abs=1e-5), "the observer did not change"
    assert agree.xent == pytest.approx(math.log(V), abs=1e-5)
    # A flat observer expects every piece a quarter of the time; against a performer that puts nearly
    # nothing on three of them, the cross-surprise is far above chance: (0 + 3 * 20) / 4 = 15, give or take.
    assert disagree.xent == pytest.approx(15.0, abs=0.1)


def test_binoculars_is_not_a_number_when_the_cross_surprise_is_zero():
    st = TokenStats(n_tokens=1, logppl_obs=1.0, logppl_perf=2.0, xent=0.0, fdg=0.0, top10=0.0)
    assert math.isnan(st.binoculars)
    assert TokenStats(1, 1.0, 3.0, 1.5, 0.0, 0.0).binoculars == 2.0


def test_finite_accepts_only_real_floats():
    assert finite(1.5) and finite(0.0)
    assert not finite(float("nan")) and not finite(float("inf"))
    assert not finite(1) and not finite("1.0") and not finite(None)


def test_missing_libraries_are_explained_with_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(MissingModels, match=r"pip install 'markcheck\[text\]'"):
        PairScorer()


def fake_libraries(monkeypatch, *, fail_loading=False):
    calls = {"threads": None, "loaded": []}
    fake_torch = types.SimpleNamespace(
        float32="f32", bfloat16="bf16", set_num_threads=lambda n: calls.__setitem__("threads", n))
    class Auto:
        @staticmethod
        def from_pretrained(name, **kw):
            if fail_loading:
                raise OSError("not cached")
            calls["loaded"].append((name, kw))
            return types.SimpleNamespace(eval=lambda: name)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = Auto
    transformers.AutoTokenizer = Auto
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_models_that_are_not_on_this_computer_are_explained_with_the_fetch_command(monkeypatch):
    fake_libraries(monkeypatch, fail_loading=True)
    with pytest.raises(MissingModels, match=r"fetch_models --text"):
        PairScorer()


def test_loading_stays_offline_by_default_uses_the_chosen_threads_and_precision(monkeypatch):
    calls = fake_libraries(monkeypatch)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    s = PairScorer(threads=3, half=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert calls["threads"] == 3
    names = [name for name, _ in calls["loaded"]]
    assert names == [scorer_mod.DEFAULT_PAIR[0], scorer_mod.DEFAULT_PAIR[0], scorer_mod.DEFAULT_PAIR[1]]
    assert all(kw["local_files_only"] is True for _, kw in calls["loaded"])
    assert [kw.get("dtype") for _, kw in calls["loaded"][1:]] == ["bf16", "bf16"]
    assert s.names == scorer_mod.DEFAULT_PAIR


def test_the_shipped_pair_stays_at_full_precision_unless_half_is_asked_for(monkeypatch):
    calls = fake_libraries(monkeypatch)
    PairScorer()
    assert [kw.get("dtype") for _, kw in calls["loaded"][1:]] == ["f32", "f32"]
