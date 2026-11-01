"""Estimate the chance that a text was AI-written, with reasons and honest error rates.

This is a separate check from the label checks. It does not look for a label;
it looks at how predictable the words are to two small open language models.
It is an estimate. It is wrong sometimes. Read docs/text-research.md first.

Wording rules (hard; tests/test_text_wording.py checks them): never use the
words that test bans, never give advice on changing a text's score, and always
print CAVEAT.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources

from .features import normalise, sentence_length_spread, sentences, words

CAVEAT = ("This is an estimate. It is wrong sometimes, it can be unfair to people writing in "
          "a language that is not their first, and it should not be the only basis for any "
          "decision about a person.")


def load_calibration() -> dict:
    return json.loads(resources.files("markcheck").joinpath("data/text_calibration.json")
                      .read_text(encoding="utf-8"))


@dataclass
class TextResult:
    n_words: int
    status: str                      # "scored" or "too_short"
    probability: float | None = None  # 0..1, None when not scored
    band: str | None = None          # "likely_ai", "uncertain", "unlikely_ai"
    headline: str = ""
    reasons: list[str] = field(default_factory=list)
    sentences: list[dict] = field(default_factory=list)  # [{"text", "effect"}]
    measures: dict = field(default_factory=dict)
    error_rates: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    caveat: str = CAVEAT

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "words": self.n_words,
            "probability_ai": None if self.probability is None else round(self.probability, 3),
            "band": self.band,
            "headline": self.headline,
            "reasons": self.reasons,
            "sentences": self.sentences,
            "measures": self.measures,
            "how_often_wrong": self.error_rates,
            "notes": self.notes,
            "caveat": self.caveat,
        }


def feature_vector(stats, spread: float | None, cal: dict) -> list[float]:
    vals = {
        "bino": stats.binoculars,
        "logppl_perf": stats.logppl_perf,
        "logppl_obs": stats.logppl_obs,
        "xent": stats.xent,
        "fdg": stats.fdg,
        "top10": stats.top10,
        "spread": cal["fill"]["spread"] if spread is None else spread,
        "log_tokens": math.log(max(stats.n_tokens, 1)),
    }
    return [vals[name] for name in cal["features"]]


def probability(x: list[float], cal: dict) -> float:
    z = cal["intercept"]
    for v, mu, sd, w in zip(x, cal["mean"], cal["std"], cal["coef"]):
        z += w * (v - mu) / sd
    return 1.0 / (1.0 + math.exp(-z))


def band_of(p: float, cal: dict) -> str:
    lo, hi = cal["uncertain_band"]
    if p >= hi:
        return "likely_ai"
    if p <= lo:
        return "unlikely_ai"
    return "uncertain"


def pct(p: float) -> str:
    """Whole percent, never 0% or 100%: the check is never that sure."""
    return f"{min(99, max(1, round(100 * p)))}%"


def headline(p: float, band: str, cal: dict) -> str:
    lo, hi = cal["uncertain_band"]
    base = (f"Estimated chance this was AI-written: {pct(p)} "
            f"(uncertain band {pct(lo)}-{pct(hi)} means we can't tell).")
    tail = {"likely_ai": "This is above the uncertain band: it looks more like AI writing than not.",
            "unlikely_ai": "This is below the uncertain band: it looks less like AI writing than not.",
            "uncertain": "This is inside the uncertain band, so we can't tell."}[band]
    return f"{base} {tail}"


def _describe(name: str, value: float, ref: dict) -> str | None:
    """One plain sentence about one measured number, with typical values from our examples."""
    typ = ref.get(name)
    if name == "top10":
        return (f"{pct(value)} of words were among a language model's top 10 guesses "
                f"(typical in our examples: {pct(typ['ai'])} for AI text, {pct(typ['person'])} for people's text)")
    if name == "bino":
        return (f"compared with a chat model's own writing, the words scored {value:.2f} for predictability "
                f"(typical: {typ['ai']:.2f} for AI text, {typ['person']:.2f} for people's text; lower is more "
                f"predictable)")
    if name == "spread":
        return (f"sentence lengths had a spread of {value:.2f} "
                f"(typical: {typ['ai']:.2f} for AI text, {typ['person']:.2f} for people's text; lower is more even)")
    return None


# What a value looks like, by which side of the midpoint between typical AI and typical people's text it sits.
_LOOKS = {
    "top10": ("Word choices were very predictable for a language model",
              "Word choices were often surprising for a language model"),
    "bino": ("The words were about as predictable as a chat model's own writing",
             "The words were less predictable than a chat model's own writing"),
    "spread": ("Sentence lengths were unusually even",
               "Sentence lengths varied a lot"),
}


def _looks_like_ai(name: str, value: float, ref: dict) -> bool:
    typ = ref[name]
    mid = (typ["ai"] + typ["person"]) / 2
    return value >= mid if typ["ai"] > typ["person"] else value <= mid


def contributions(x: list[float], cal: dict) -> dict[str, float]:
    """How far each measure moved the estimate (in the fitted model's own units)."""
    return {name: w * (v - mu) / sd
            for name, v, mu, sd, w in zip(cal["features"], x, cal["mean"], cal["std"], cal["coef"])}


def reasons_for(stats, spread: float | None, cal: dict, x: list[float] | None = None) -> list[str]:
    """Reasons from this text's own numbers, biggest effect first, in the direction the model used them."""
    ref = cal["reference"]
    x = x if x is not None else feature_vector(stats, spread, cal)
    contrib = contributions(x, cal)
    values = {"top10": stats.top10, "bino": stats.binoculars, "spread": spread}
    out = []
    for name, c in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
        if name not in _LOOKS:
            continue
        if name == "spread" and spread is None:
            out.append("There were too few sentences to judge how even their lengths were.")
            continue
        if abs(c) < 0.25:
            lead = {"top10": "Word predictability", "bino": "Predictability compared with a chat model",
                    "spread": "Sentence length"}[name]
            out.append(f"{lead} hardly moved the estimate: {_describe(name, values[name], ref)}.")
            continue
        lead = _LOOKS[name][0] if _looks_like_ai(name, values[name], ref) else _LOOKS[name][1]
        way = "raised" if c > 0 else "lowered"
        out.append(f"{lead}, which {way} the estimate: {_describe(name, values[name], ref)}.")
    return out


def sentence_effects(text: str, stats, cal: dict | None = None, limit: int = 2) -> list[dict]:
    """The sentences that pushed the estimate most, each way, from the per-word numbers.

    Uses the per-sentence version of the Binoculars score, and the sign the fitted
    model gave that score, so "raised" and "lowered" match the estimate.
    """
    if not stats.token_ll:
        return []
    sign = -1.0
    if cal and "bino" in cal.get("features", []):
        sign = 1.0 if cal["coef"][cal["features"].index("bino")] > 0 else -1.0
    overall = stats.binoculars
    rows, seen = [], set()
    for a, b in sentences(text):
        key = " ".join(text[a:b].split())
        if key in seen:
            continue
        seen.add(key)
        ll, xe, n = 0.0, 0.0, 0
        for (s, e), lv, xv in zip(stats.offsets, stats.token_ll, stats.token_xent):
            if s >= a and e <= b and e > s:
                ll += -lv
                xe += xv
                n += 1
        if n < 5 or xe <= 0:
            continue
        local = ll / xe
        rows.append({"text": text[a:b], "score": local, "push": sign * (local - overall)})
    if len(rows) < 2:
        return []
    up = sorted([r for r in rows if r["push"] > 0], key=lambda r: -r["push"])[:limit]
    down = sorted([r for r in rows if r["push"] < 0], key=lambda r: r["push"])[:limit]
    out = []
    for r in up:
        out.append({"text": _clip(r["text"]), "effect": "raised the estimate",
                    "why": f"its words were more predictable than the rest of the text (score {r['score']:.2f})"})
    for r in down:
        out.append({"text": _clip(r["text"]), "effect": "lowered the estimate",
                    "why": f"its words were less predictable than the rest of the text (score {r['score']:.2f})"})
    return out


def _clip(s: str, n: int = 120) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def error_rates(cal: dict) -> dict:
    return cal["heldout"]["how_often_wrong"]


def check_text(text: str, scorer=None, cal: dict | None = None) -> TextResult:
    """Score one text. `scorer` is anything with .score(text) -> TokenStats (tests pass a stand-in)."""
    cal = cal or load_calibration()
    text = normalise(text)
    n = len(words(text))
    res = TextResult(n_words=n, status="too_short", error_rates=error_rates(cal))
    if n < cal["min_words"]:
        res.headline = (f"Too short to estimate: {n} words. markcheck needs at least {cal['min_words']} words. "
                        f"On shorter texts our held-out checks were wrong too often to give a number.")
        return res
    if scorer is None:
        from .scorer import PairScorer
        scorer = PairScorer(*cal["models"])
    stats = scorer.score(text)
    spread = sentence_length_spread(text)
    x = feature_vector(stats, spread, cal)
    p = probability(x, cal)
    res.status = "scored"
    res.probability = p
    res.band = band_of(p, cal)
    res.headline = headline(p, res.band, cal)
    res.reasons = reasons_for(stats, spread, cal, x)
    res.sentences = sentence_effects(text, stats, cal)
    res.measures = {"tokens": stats.n_tokens, "binoculars": round(stats.binoculars, 4),
                    "top10_share": round(stats.top10, 4),
                    "sentence_length_spread": None if spread is None else round(spread, 4),
                    "fast_detectgpt": round(stats.fdg, 4)}
    if n < cal.get("full_confidence_words", 0):
        res.notes.append(f"This text is short ({n} words). Our held-out checks were less accurate "
                         f"below {cal['full_confidence_words']} words.")
    if stats.truncated:
        res.notes.append("Only the first part of the text (about 512 word pieces) was scored.")
    res.notes.append(cal["data_note"])
    return res
