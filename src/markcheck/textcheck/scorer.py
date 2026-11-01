"""Score a text with two small open language models (needs `markcheck[text]`).

Plain idea: a language model guesses the next word over and over. AI-written
text tends to use the words a model expects, so the model is rarely surprised.
On its own that also flags plain, predictable writing by people. Binoculars
(Hans et al. 2024, arXiv 2401.12070) divides the surprise by how surprised one
model is by the *other* model's guesses, which cancels part of that effect.

We use a base model (the "observer") and its chat-tuned twin (the
"performer"). They must share a vocabulary. Everything runs on the CPU and
never goes online: the weights must already be on this computer (fetch them
once with `python -m markcheck.fetch_models --text`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

DEFAULT_PAIR = ("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct")
MAX_TOKENS = 512


@dataclass
class TokenStats:
    """Numbers for one text. Per-token lists line up with `offsets`."""
    n_tokens: int
    logppl_obs: float        # average surprise of the observer (log perplexity)
    logppl_perf: float       # average surprise of the performer
    xent: float              # how surprised the observer is by the performer's guesses
    fdg: float               # Fast-DetectGPT curvature with the performer alone
    top10: float             # share of words in the performer's top 10 guesses
    token_ll: list[float] = field(default_factory=list)      # performer log-prob per token
    token_xent: list[float] = field(default_factory=list)    # cross-surprise per token
    offsets: list[tuple[int, int]] = field(default_factory=list)  # character span per token
    truncated: bool = False

    @property
    def binoculars(self) -> float:
        return self.logppl_perf / self.xent if self.xent > 0 else float("nan")


class MissingModels(RuntimeError):
    pass


class PairScorer:
    def __init__(self, observer: str = DEFAULT_PAIR[0], performer: str = DEFAULT_PAIR[1],
                 local_only: bool = True, threads: int | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on install
            raise MissingModels(
                "The text check needs extra libraries. Install them with: pip install 'markcheck[text]'"
            ) from exc
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except Exception:  # pragma: no cover - older transformers
            pass
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.names = (observer, performer)
        try:
            self.tok = AutoTokenizer.from_pretrained(observer, local_files_only=local_only)
            self.obs = AutoModelForCausalLM.from_pretrained(observer, local_files_only=local_only,
                                                            dtype=torch.float32).eval()
            self.perf = AutoModelForCausalLM.from_pretrained(performer, local_files_only=local_only,
                                                             dtype=torch.float32).eval()
        except OSError as exc:
            raise MissingModels(
                "The language models for the text check are not on this computer yet. "
                "Fetch them once with: python -m markcheck.fetch_models --text"
            ) from exc

    def score(self, text: str) -> TokenStats:
        torch = self.torch
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True,
                       truncation=True, max_length=MAX_TOKENS + 1)
        ids = enc["input_ids"]
        offsets = [tuple(o) for o in enc["offset_mapping"][0].tolist()]
        kept = int(ids.shape[1])
        bos = self.tok.bos_token_id
        # Prepend a start token so the first word is scored too, unless the tokenizer added one
        # itself. Tokenizers without one (Qwen, for example) leave the first word unscored.
        if bos is not None and (kept == 0 or int(ids[0, 0]) != bos):
            ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
            offsets = [(0, 0)] + offsets
        if ids.shape[1] < 2:
            raise ValueError("The text has too few word pieces to score.")
        # Truncated only if the tokenizer dropped pieces, whether or not a start token was added.
        truncated = len(self.tok(text)["input_ids"]) > kept
        with torch.no_grad():
            lo = self.obs(ids).logits[0, :-1].float()
            lp = self.perf(ids).logits[0, :-1].float()
        target = ids[0, 1:]
        logp_o = torch.log_softmax(lo, -1)
        logp_p = torch.log_softmax(lp, -1)
        p_p = logp_p.exp()
        ll_o = logp_o.gather(-1, target[:, None])[:, 0]
        ll_p = logp_p.gather(-1, target[:, None])[:, 0]
        # Binoculars cross-perplexity: observer's distribution scored against performer's log-probs.
        p_o = logp_o.exp()
        xent_tok = -(p_o * logp_p).sum(-1)
        # Fast-DetectGPT (analytic, one model): how much likelier the text is than the model's own samples.
        mean_ref = (p_p * logp_p).sum(-1)
        var_ref = (p_p * logp_p.pow(2)).sum(-1) - mean_ref.pow(2)
        fdg = ((ll_p.sum() - mean_ref.sum()) / var_ref.sum().clamp_min(1e-6).sqrt()).item()
        rank = (logp_p > ll_p[:, None]).sum(-1)
        n = int(target.numel())
        return TokenStats(
            n_tokens=n,
            logppl_obs=float(-ll_o.mean()),
            logppl_perf=float(-ll_p.mean()),
            xent=float(xent_tok.mean()),
            fdg=float(fdg),
            top10=float((rank < 10).float().mean()),
            token_ll=[float(v) for v in ll_p],
            token_xent=[float(v) for v in xent_tok],
            offsets=offsets[1:],
            truncated=truncated,
        )


def finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)
