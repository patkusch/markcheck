"""Fit the text check's calibration on the train side and measure it on held-out texts.

    python scripts/measure_text.py PERFORMER [--features a,b,c] [--train-on hc3+mage+raid]
                                   [--report] [--baseline] [--json-out FILE] [--write]

Needs splits.jsonl (scripts/text_data.py) and the cached scores (scripts/text_features.py).
With --write, saves src/markcheck/data/text_calibration.json.

Each text gets a source (hc3, mage, raid, newer, or a people-only set), a model (or "human"),
a kind of writing, an attack ("none" or e.g. "paraphrase") and a document id. Texts are put
on the train side or the held-out side:
  - HC3: by its own split (by question).
  - MAGE: half by text hash; the reworded copies are held out only.
  - RAID: half by source document, so a person's text and the AI texts prompted from it are
    always on the same side.
  - Everything else (newer models, essay sets) is held out only.
All refits (leave one model out, leave one kind out, leave news out) run in this process.

Numbers: AUROC (the chance a random AI text scores higher than a random person's text),
share of AI caught when 1 in 100 people's texts is wrongly flagged, calibration error
(average gap between the stated chance and how often it came true), and the shares of
texts above, inside and below the "can't tell" band.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markcheck.textcheck.features import sentence_length_spread, words  # noqa: E402

DATA = Path.home() / ".cache" / "markcheck" / "textdata"
ROOT = Path(__file__).resolve().parents[1]
CAL_PATH = ROOT / "src" / "markcheck" / "data" / "text_calibration.json"
LENGTH_BINS = [(30, 59), (60, 99), (100, 149), (150, 249), (250, 10**6)]
PAIRS = {
    "HuggingFaceTB/SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M",
    "gpt2-medium": "gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B",
}
HC3_DOMAINS = ("finance", "medicine", "wiki_csai", "reddit_eli5")
TRAINABLE = ("hc3", "mage", "raid")
# One name per kind of writing across sources, so "leave news out" removes news from every source.
KIND_MAP = {
    "cnn": "news", "imdb": "reviews", "pubmed": "abstracts", "dialogsum": "dialogue summaries",
    "finance": "answers (finance)", "medicine": "answers (medicine)", "wiki_csai": "answers (computing)",
    "reddit_eli5": "answers (explain like I'm five)", "toefl": "student essays (non-native)",
    "us_students": "student essays (US)", "college": "student essays (US college)",
    "cs224n": "student abstracts (Stanford)",
}
# Plain names for the newer models in the "How often" list.
SHOW_MODEL = {"gpt-4o": "GPT-4o", "gemini-2.5-flash": "Gemini 2.5 Flash", "deepseek-v3": "DeepSeek-V3",
              "qwen-max": "Qwen-Max"}
OLD_SETS = ["train", "heldout", "mage", "mage_para", "toefl", "us_students", "college", "cs224n"]


# ---------------------------------------------------------------- data

def norm_kind(k: str) -> str:
    k = (k or "unknown").strip()
    return KIND_MAP.get(k, KIND_MAP.get(k.lower(), k.lower()))


# The same model under two names in different sources.
MODEL_ALIASES = {"gpt4": "gpt-4", "gpt-3.5-turbo": "chatgpt"}
NEWER = ("detectrlx", "beemo")


def norm_model(m) -> str:
    m = str(m).strip() if m else "unknown"
    return MODEL_ALIASES.get(m.lower(), m)


def attack_name(att: str) -> str:
    """Plain set name for an attack. Rewording by another AI is called "reworded"."""
    return "reworded" if att == "paraphrase" else att


def load(perf: str):
    rows = {}
    for line in open(DATA / "splits.jsonl"):
        r = json.loads(line)
        rows[r["id"]] = r
    out = []
    for line in open(DATA / f"feat_{perf.replace('/', '__')}.jsonl"):
        f = json.loads(line)
        if f["id"] not in rows:
            continue
        r = dict(rows[f["id"]])
        r.update(f)
        if r.get("n_tokens") is None or r.get("bino") is None:
            continue
        r["log_tokens"] = math.log(max(r["n_tokens"], 1))
        # Cheap measures are recomputed with the current rules, so a change to sentence or
        # word rules is picked up without rescoring with the models.
        r["n_words"] = len(words(r["text"]))
        r["spread"] = sentence_length_spread(r["text"])
        describe(r)
        out.append(r)
    return out


def describe(r):
    """Fill source/model/kind/attack/doc_id for old rows that do not have them."""
    split = r["split"]
    if not r.get("source"):
        if split in ("train", "heldout"):
            r["source"] = "hc3"
        elif split in ("mage", "mage_para"):
            r["source"] = "mage"
        else:
            r["source"] = split
    r["source"] = str(r["source"]).lower()
    if r["label"] == 0:
        r["model"] = "human"
    elif not r.get("model"):
        r["model"] = {"hc3": "chatgpt", "mage": "gpt-4"}.get(r["source"], "unknown")
    r["model"] = norm_model(r["model"])
    r["kind"] = norm_kind(r.get("kind") or r.get("domain"))
    att = r.get("attack")
    if not att or str(att).lower() in ("none", "null", "nan", ""):
        att = "paraphrase" if (split == "mage_para" and r["label"] == 1) else "none"
    r["attack"] = str(att).lower()
    if r.get("doc_id") is None:
        r["doc_id"] = hashlib.sha256(r["text"].encode()).hexdigest()[:16]
    r["doc_id"] = str(r["doc_id"])


def half(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2


def assign(rows, train_on=("hc3", "mage"), reworded_train=False):
    """Give each row a side ("train" or "test") and a held-out set name.

    Returns new dicts with r["side"] and r["set"]. Rows not usable anywhere are dropped."""
    mage_train_people = {r["text"] for r in rows if r["split"] == "mage" and r["label"] == 0
                         and half(r["text"]) == 0}
    out = []
    for r in rows:
        r = dict(r)
        src = r["source"]
        if src == "hc3":
            r["side"] = "train" if r["split"] == "train" else "test"
            r["set"] = "heldout"
        elif src == "mage":
            if r["split"] == "mage":
                r["side"] = "train" if half(r["text"]) == 0 else "test"
                r["set"] = "mage"
            else:  # mage_para: held out only; drop people's texts whose twin is fitted on
                if r["label"] == 0 and r["text"] in mage_train_people:
                    continue
                r["side"], r["set"] = "test", "mage_para"
        elif src == "raid":
            r["side"] = "train" if half("raid:" + r["doc_id"]) == 0 else "test"
            r["set"] = "raid" if r["attack"] == "none" else "raid_" + attack_name(r["attack"])
        elif src in NEWER and "newer" in train_on:
            # Only when asked: then these models are no longer unseen.
            r["side"] = "train" if half(r["doc_id"]) == 0 else "test"
            r["set"] = r["split"]
        else:
            r["side"], r["set"] = "test", r["split"] if r["split"] not in ("train", "heldout") else src
        if r["side"] == "train":
            if src not in train_on and not (src in NEWER and "newer" in train_on):
                r["side"] = "unused"
            elif r["attack"] != "none" and not (reworded_train and r["label"] == 1):
                r["side"] = "unused"
        out.append(r)
    return out


# ---------------------------------------------------------------- fitting

def matrix(rows, feats, fill):
    return np.array([[fill[k] if r.get(k) is None else r[k] for k in feats] for r in rows], float)


def fit_logistic(X, y, l2=1e-2, iters=50, weights=None):
    n, d = X.shape
    sw = np.ones(n) if weights is None else np.asarray(weights, float)
    A = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -40, 40)))
        g = A.T @ (sw * (p - y)) + l2 * np.r_[w[:-1], 0]
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + l2 * np.diag(np.r_[np.ones(d), 0])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w[:-1], w[-1]


def weights_for(tr, scheme):
    """Training weights. "source": each source counts the same in total (round 3).
    "source_label": each source, and people and AI within it, count the same.
    "source_label_model": as source_label, and each AI model within a source counts the same."""
    n = len(tr)
    if scheme == "none":
        return np.ones(n)
    srcs = sorted({r["source"] for r in tr})
    w = np.zeros(n)
    if scheme == "source":
        for s in srcs:
            m = np.array([r["source"] == s for r in tr])
            w[m] = 1.0 / (len(srcs) * m.sum())
        return w * n
    cells = defaultdict(list)
    for i, r in enumerate(tr):
        cells[(r["source"], r["label"])].append(i)
    for (s, lab), idx in cells.items():
        share = 1.0 / (len(srcs) * len({lab2 for (s2, lab2) in cells if s2 == s}))
        if scheme == "source_label_model" and lab == 1:
            by_m = defaultdict(list)
            for i in idx:
                by_m[tr[i]["model"]].append(i)
            for ids in by_m.values():
                w[ids] = share / (len(by_m) * len(ids))
        else:
            w[idx] = share / len(idx)
    return w * n


class Fit:
    def __init__(self, tr, feats, weights="source", band="auto", l2=1e-2):
        self.feats = feats
        self.fill = {k: float(np.median([r[k] for r in tr if r.get(k) is not None])) for k in feats}
        X = matrix(tr, feats, self.fill)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        y = np.array([r["label"] for r in tr], float)
        self.coef, self.b = fit_logistic((X - self.mu) / self.sd, y, l2=l2, weights=weights_for(tr, weights))
        self.band = self._band(tr, y, band)

    @classmethod
    def from_json(cls, cal):
        f = cls.__new__(cls)
        f.feats, f.fill = cal["features"], {k: cal["fill"].get(k, 0.5) for k in cal["features"]}
        f.mu, f.sd = np.array(cal["mean"]), np.array(cal["std"])
        f.coef, f.b = np.array(cal["coef"]), cal["intercept"]
        f.band = tuple(cal["uncertain_band"])
        return f

    def predict(self, rs):
        if not rs:
            return np.zeros(0)
        X = (matrix(rs, self.feats, self.fill) - self.mu) / self.sd
        return 1 / (1 + np.exp(-np.clip(X @ self.coef + self.b, -40, 40)))

    def _band(self, tr, y, band):
        if band not in ("auto", "auto_max"):
            return tuple(float(v) for v in band.split(","))
        # Top of the band: no lower than 65%, and high enough that at most 1 in 100 people's
        # training texts reach it ("auto_max": in every source, not just overall). Bottom: no
        # higher than 35%, and low enough that at most 1 in 100 AI training texts fall under it.
        p = self.predict(tr)
        if band == "auto":
            hi_q = float(np.quantile(p[y == 0], 0.99))
        else:
            hi_q = max(float(np.quantile(p[(y == 0) & np.array([r["source"] == s for r in tr])], 0.99))
                       for s in {r["source"] for r in tr})
        hi = max(0.65, hi_q)
        lo = min(0.35, float(np.quantile(p[y == 1], 0.01)))
        return (round(lo, 2), min(0.99, round(hi + 0.005, 2)))


# ---------------------------------------------------------------- metrics

def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    allv = np.r_[s[y == 1], s[y == 0]]
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    ranks = np.empty(len(allv))
    # average ranks for ties
    _, first, counts = np.unique(sv, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2 + 1
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tpr_at_fpr(s, y, fpr=0.01):
    s, y = np.asarray(s), np.asarray(y)
    neg = np.sort(s[y == 0])
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(math.ceil((1 - fpr) * len(neg))) - 1)]
    return float((s[y == 1] > thr).mean())


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    if len(p) == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def summary(p, y, band):
    p, y = np.asarray(p), np.asarray(y)
    lo, hi = band
    hum, ai = p[y == 0], p[y == 1]
    d = {"n": int(len(p)), "auroc": auroc(p, y), "tpr_at_1pct_fpr": tpr_at_fpr(p, y), "ece": ece(p, y)}
    if len(hum):
        d["n_people"] = int(len(hum))
        d["people_above_band"] = float((hum >= hi).mean())
        d["people_in_band"] = float(((hum > lo) & (hum < hi)).mean())
    if len(ai):
        d["n_ai"] = int(len(ai))
        d["ai_below_band"] = float((ai <= lo).mean())
        d["ai_above_band"] = float((ai >= hi).mean())
        d["ai_in_band"] = float(((ai > lo) & (ai < hi)).mean())
    return d


def score(fit, rs):
    return summary(fit.predict(rs), np.array([r["label"] for r in rs]), fit.band)


def fmt(d):
    keys = ["n", "auroc", "tpr_at_1pct_fpr", "ece", "people_above_band", "people_in_band",
            "ai_above_band", "ai_below_band"]
    return "  ".join(f"{k}={d[k]:.3f}" if isinstance(d.get(k), float) else f"{k}={d.get(k, '-')}"
                     for k in keys if k in d)


def clean(o):
    """JSON has no NaN: write null instead."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


# ---------------------------------------------------------------- experiment

class Exp:
    def __init__(self, rows, a):
        self.a = a
        self.feats = a.features.split(",")
        self.rows = assign(rows, tuple(a.train_on.split("+")), a.reworded_train)
        ok = [r for r in self.rows if r["n_words"] >= a.min_words]
        self.tr = [r for r in ok if r["side"] == "train"]
        self.te = [r for r in ok if r["side"] == "test"]

    def fit(self, tr=None):
        a = self.a
        return Fit(self.tr if tr is None else tr, self.feats, a.weights, a.band, a.l2)

    # held-out text groups -------------------------------------------------
    def test_clean(self):
        """Held-out texts with no attack, from sources that have both people and AI."""
        both = {s for s in {r["source"] for r in self.te}
                if {0, 1} <= {r["label"] for r in self.te if r["source"] == s}}
        return [r for r in self.te if r["attack"] == "none" and r["source"] in both and r["set"] != "mage_para"]

    def by_set(self, fit):
        out = {}
        for s in sorted({r["set"] for r in self.te}, key=lambda s: (s not in OLD_SETS, s)):
            rs = [r for r in self.te if r["set"] == s]
            out[s] = score(fit, rs)
        trs = [r for r in self.tr]
        out["train"] = score(fit, trs)
        return out

    def by_model(self, fit, rows=None):
        """Each AI model's held-out texts against the people's texts from the same sources."""
        rows = self.test_clean() if rows is None else rows
        out = {}
        for m in sorted({r["model"] for r in rows if r["label"] == 1}):
            ai = [r for r in rows if r["model"] == m]
            srcs = {r["source"] for r in ai}
            kinds = {r["kind"] for r in ai}
            people = [r for r in rows if r["label"] == 0 and r["source"] in srcs and r["kind"] in kinds]
            d = score(fit, ai + people)
            d["sources"] = sorted(srcs)
            out[m] = d
        return out

    def by_kind(self, fit, rows=None):
        rows = self.test_clean() if rows is None else rows
        return {k: score(fit, [r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})}

    def by_source(self, fit):
        rows = self.test_clean()
        return {s: score(fit, [r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})}

    def people_only(self, fit):
        """Sets with only people's texts (essays by students and the like)."""
        out = {}
        for s in sorted({r["set"] for r in self.te}):
            rs = [r for r in self.te if r["set"] == s]
            if all(r["label"] == 0 for r in rs):
                out[s] = score(fit, rs)
                groups = sorted({str(r.get("group")) for r in rs if r.get("group") is not None})
                if len(groups) > 1:
                    out[s]["by_group"] = {g: score(fit, [r for r in rs if str(r.get("group")) == g]) for g in groups}
        return out

    def reworded(self, fit):
        out = {}
        for src in sorted({r["source"] for r in self.te}):
            for att in sorted({r["attack"] for r in self.te if r["source"] == src and r["attack"] != "none"}):
                ai = [r for r in self.te if r["source"] == src and r["attack"] == att and r["label"] == 1]
                if not ai:
                    continue
                kinds = {r["kind"] for r in ai}
                people = [r for r in self.te if r["source"] == src and r["label"] == 0 and r["attack"] == "none"
                          and r["kind"] in kinds]
                if src == "mage":  # MAGE's reworded file has its own copy of the people's texts
                    people = [r for r in self.te if r["set"] == "mage_para" and r["label"] == 0]
                d = score(fit, ai + people)
                d["by_model"] = {m: score(fit, [r for r in ai if r["model"] == m] + people)
                                 for m in sorted({r["model"] for r in ai})}
                out[f"{src}_{attack_name(att)}"] = d
        return out

    def by_length(self, fit):
        allh = [r for r in self.rows if r["side"] == "test" and r["set"] in ("heldout", "mage")]
        out = {}
        for lo_w, hi_w in LENGTH_BINS:
            rs = [r for r in allh if lo_w <= r["n_words"] <= hi_w]
            if rs:
                out[f"{lo_w}-{hi_w if hi_w < 10**6 else ''}"] = score(fit, rs)
        return out

    # refits ---------------------------------------------------------------
    def leave_one_model_out(self, min_n=30):
        """Refit without model M's AI texts (people kept); test on M's held-out texts."""
        out = {}
        tc = self.test_clean()
        for m in sorted({r["model"] for r in self.tr if r["label"] == 1}):
            fit = self.fit([r for r in self.tr if r["model"] != m])
            ai = [r for r in tc if r["model"] == m]
            if len([r for r in ai]) < min_n:
                continue
            srcs, kinds = {r["source"] for r in ai}, {r["kind"] for r in ai}
            people = [r for r in tc if r["label"] == 0 and r["source"] in srcs and r["kind"] in kinds]
            d = score(fit, ai + people)
            d["band"] = list(fit.band)
            out[m] = d
        return out

    def leave_one_kind_out(self, min_n=30):
        """Refit without any text of kind K; test on K's held-out texts (all sources)."""
        out = {}
        tc = self.test_clean()
        for k in sorted({r["kind"] for r in self.tr}):
            rs = [r for r in tc if r["kind"] == k]
            if len(rs) < min_n:
                continue
            fit = self.fit([r for r in self.tr if r["kind"] != k])
            d = score(fit, rs)
            d["band"] = list(fit.band)
            d["by_source"] = {s: score(fit, [r for r in rs if r["source"] == s])
                              for s in sorted({r["source"] for r in rs})}
            out[k] = d
        return out


def worst(d, key, default=0.0):
    vals = [v.get(key) for v in d.values() if v.get(key) is not None]
    return max(vals) if vals else default


def report(exp, fit, full=True):
    t0 = time.time()
    res = {"band": list(fit.band), "coef": dict(zip(exp.feats, map(float, fit.coef))),
           "intercept": float(fit.b), "n_train": len(exp.tr),
           "n_train_by_source": {s: sum(r["source"] == s for r in exp.tr) for s in sorted({r["source"] for r in exp.tr})},
           "sets": exp.by_set(fit), "by_source": exp.by_source(fit), "by_model": exp.by_model(fit),
           "by_kind": exp.by_kind(fit), "people_only": exp.people_only(fit), "reworded": exp.reworded(fit),
           "by_length": exp.by_length(fit)}
    if full:
        res["leave_one_model_out"] = exp.leave_one_model_out()
        res["leave_one_kind_out"] = exp.leave_one_kind_out()
    tc = exp.test_clean()
    res["overall"] = score(fit, tc)
    # The same, only for the sources the ship rule's 1-in-100 limit is about: HC3, MAGE and RAID.
    res["overall_fit_sources"] = score(fit, [r for r in tc if r["source"] in TRAINABLE])
    res["seconds"] = round(time.time() - t0, 1)
    return res


def print_report(res):
    print("band", res["band"], " n_train", res["n_train"], res["n_train_by_source"])
    print("coef", {k: round(v, 3) for k, v in res["coef"].items()}, "intercept", round(res["intercept"], 3))
    print("overall held-out (no attack):", fmt(res["overall"]))
    for name in ("sets", "by_source", "by_model", "by_kind", "people_only", "reworded",
                 "leave_one_model_out", "leave_one_kind_out", "by_length"):
        if name in res and res[name]:
            print(f"-- {name}")
            for k, v in res[name].items():
                print(f"   {k[:34]:34s}", fmt(v))


# ---------------------------------------------------------------- comparing two fits

MAX_PEOPLE_FLAGGED = 0.01


def people_rates(res):
    """People's texts put above the band, per kind of writing and per people-only set."""
    out = {f"kind: {k}": v for k, v in res["by_kind"].items() if v.get("n_people")}
    out.update({f"set: {k}": v for k, v in res["people_only"].items() if v.get("n_people")})
    return {k: (v["people_above_band"], v["n_people"]) for k, v in out.items()}


def compare(old, new):
    """The ship rule: people's texts flagged overall stay at 1 in 100 or fewer, and no kind
    of writing (or people-only set) has more people's texts flagged than with the old fit."""
    po, pn = people_rates(old), people_rates(new)
    rows, worse = [], []
    for k in sorted(set(po) | set(pn)):
        o, n = po.get(k, (None, 0)), pn.get(k, (None, 0))
        rows.append({"group": k, "n": n[1] or o[1], "old": o[0], "new": n[0]})
        if o[0] is not None and n[0] is not None and n[0] > o[0]:
            worse.append(k)
    key = "overall_fit_sources" if "overall_fit_sources" in new else "overall"
    overall_new = new[key].get("people_above_band", 1.0)
    ok = overall_new <= MAX_PEOPLE_FLAGGED and not worse
    print("-- compare (people's texts above the band: old -> new)")
    print(f"   overall ({key}) {old.get(key, old['overall']).get('people_above_band', float('nan')):.4f} -> "
          f"{overall_new:.4f}   AI caught {old.get(key, old['overall']).get('ai_above_band', float('nan')):.3f} -> "
          f"{new[key].get('ai_above_band', float('nan')):.3f}")
    for r in rows:
        print(f"   {r['group'][:40]:40s} n={r['n']:5d}  {r['old'] if r['old'] is None else round(r['old'], 4)}"
              f" -> {r['new'] if r['new'] is None else round(r['new'], 4)}" + ("   WORSE" if r["group"] in worse else ""))
    print("   ship rule:", "PASS" if ok else f"FAIL (overall {overall_new:.4f}; worse: {', '.join(worse) or 'none'})")
    return {"ship": ok, "worse": worse, "overall_key": key,
            "overall_old": old.get(key, old["overall"]), "overall_new": new[key],
            "rows": rows, "old_sets": old["sets"], "old_by_model": old["by_model"],
            "old_reworded": old["reworded"]}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perf")
    ap.add_argument("--features", default="bino,top10,spread,fdg,logppl_perf,log_tokens")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--full-words", type=int, default=150)
    ap.add_argument("--band", default="auto", help="'auto', 'auto_max' or 'lo,hi'")
    ap.add_argument("--train-on", default="hc3+mage",
                    help="sources to fit on, joined by +: hc3, mage, raid, newer (DetectRL-X and Beemo)")
    ap.add_argument("--weights", default="source", choices=["none", "source", "source_label", "source_label_model"])
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--reworded-train", action="store_true", help="also fit on reworded AI texts (train side)")
    ap.add_argument("--hold-kind", default=None, help="leave this kind of writing out of fitting")
    ap.add_argument("--baseline", action="store_true", help="measure the current calibration json, no refit")
    ap.add_argument("--quick", action="store_true", help="skip the leave-one-out refits")
    ap.add_argument("--compare-with", default=None,
                    help="an older calibration json: measure it on the same held-out texts and apply the ship rule")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--fairness", default=None, help="fair_results.json to merge under 'fairness' on --write")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rows = load(a.perf)
    exp = Exp(rows, a)
    if a.hold_kind:
        exp.tr = [r for r in exp.tr if r["kind"] != norm_kind(a.hold_kind)]
    print(f"loaded {len(rows)} texts in {time.time() - t0:.0f}s; train {len(exp.tr)}, held out {len(exp.te)}")
    fit = Fit.from_json(json.loads(CAL_PATH.read_text())) if a.baseline else exp.fit()
    res = report(exp, fit, full=not a.quick)
    res["args"] = vars(a)
    res["cpu_seconds_total"] = round(time.process_time(), 1)
    print_report(res)
    if a.compare_with:
        old = Fit.from_json(json.loads(Path(a.compare_with).read_text()))
        res["compare"] = compare(report(exp, old, full=False), res)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(clean(res), indent=1) + "\n")
    if a.write:
        write(a, exp, fit, res)


# ---------------------------------------------------------------- writing the calibration

def ref_values(tr, key):
    return {"ai": float(np.median([r[key] for r in tr if r["label"] == 1 and r.get(key) is not None])),
            "person": float(np.median([r[key] for r in tr if r["label"] == 0 and r.get(key) is not None]))}


def in_100(x):
    if x is None or not math.isfinite(x):
        return "not measured"
    return f"{round(100 * x)} in 100" if x >= 0.005 else "fewer than 1 in 100"


def mean_of(d, key, names=None):
    vals = [v[key] for k, v in d.items() if (names is None or k in names) and v.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def srcs_fit(exp):
    return {r["source"] for r in exp.tr}


def write(a, exp, fit, res):
    sets = res["sets"]
    h, m = sets.get("heldout", {}), sets.get("mage", {})
    po = res["people_only"]
    lomo, loko = res.get("leave_one_model_out", {}), res.get("leave_one_kind_out", {})
    rew = res["reworded"]
    raid = res["by_source"].get("raid", {})
    newer = sets.get("newmodels", {})
    news = loko.get("news", {})
    worst_kind = max(loko, key=lambda k: loko[k].get("people_above_band", 0)) if loko else None
    worst_model = min(lomo, key=lambda k: lomo[k].get("ai_above_band", 1)) if lomo else None
    toefl = po.get("toefl", {})
    nonnative = {k: v for k, v in po.items() if k not in ("us_students", "college", "cs224n")}
    nn_above = worst(nonnative, "people_above_band", float("nan"))
    rew_main = rew.get("raid_reworded") or rew.get("mage_reworded") or {}
    # Ship one pooled number per reworded set, with no breakdown that could coach anyone.
    rew = {k: {kk: vv for kk, vv in v.items() if kk != "by_model"} for k, v in rew.items()}
    how = {
        "people's texts put above the band (similar topics)": in_100(h.get("people_above_band")),
        "AI texts caught above the band (similar topics)": in_100(h.get("ai_above_band")),
        "AI texts put below the band (similar topics)": in_100(h.get("ai_below_band")),
        "people's texts put above the band (news, reviews, summaries, abstracts; GPT-4)": in_100(m.get("people_above_band")),
        "AI texts caught above the band (news, reviews, summaries, abstracts; GPT-4)": in_100(m.get("ai_above_band")),
        "AI texts put below the band (news, reviews, summaries, abstracts; GPT-4)": in_100(m.get("ai_below_band")),
        "AI texts reworded by another AI, caught above the band": in_100(rew_main.get("ai_above_band")),
        "essays by non-native English students put above the band": in_100(toefl.get("people_above_band")),
        "essays by non-native English students put inside the band": in_100(toefl.get("people_in_band")),
        "essays by US school and college students put inside or above the band": in_100(
            mean_of({k: {"x": v["people_above_band"] + v["people_in_band"]} for k, v in po.items()
                     if k in ("us_students", "college")}, "x")),
    }
    if raid:
        how["people's texts put above the band (RAID: 8 kinds of writing, 11 AI models)"] = in_100(raid.get("people_above_band"))
        how["AI texts caught above the band (RAID: 8 kinds of writing, 11 AI models)"] = in_100(raid.get("ai_above_band"))
    seen = sorted({r["model"] for r in exp.tr if r["label"] == 1})
    unseen = {m: v for m, v in res["by_model"].items()
              if m not in seen and set(v.get("sources", [])) <= set(NEWER) and v.get("n_ai", 0) >= 30}
    # Fitting on these models (--train-on ...+newer) makes them seen: then no row claims otherwise.
    for m in sorted(unseen, key=lambda m: (m not in SHOW_MODEL, m.lower())):
        if m in SHOW_MODEL:
            how[f"AI texts by {SHOW_MODEL[m]} (2024-2025), a model never used for fitting, caught above the band"] = \
                in_100(unseen[m].get("ai_above_band"))
    beemo = res["by_source"].get("beemo", {})
    if beemo.get("n_ai") and "beemo" not in srcs_fit(exp):
        how["AI texts by ten open chat models never used for fitting (Llama 2, Gemma, Mistral, Tulu, Zephyr), "
            "caught above the band"] = in_100(beemo.get("ai_above_band"))
    if newer.get("people_above_band") is not None:
        how["people's texts put above the band (the newer-model sets)"] = in_100(newer.get("people_above_band"))
    if lomo:
        how["AI texts caught above the band from a model left out of fitting (average over models)"] = in_100(mean_of(lomo, "ai_above_band"))
        how[f"AI texts caught above the band from a model left out of fitting (worst: {worst_model})"] = in_100(lomo[worst_model].get("ai_above_band"))
    if loko:
        how["people's texts put above the band in a kind of writing left out of fitting (average over kinds)"] = in_100(mean_of(loko, "people_above_band"))
    if news:
        how["people's news put above the band when news was left out of fitting"] = in_100(news.get("people_above_band"))
    if worst_kind:
        how[f"people's {worst_kind} put above the band when that kind of writing was left out of fitting (the worst case we tried)"] = in_100(loko[worst_kind].get("people_above_band"))
    if len(nonnative) > 1 and math.isfinite(nn_above):
        how["essays by non-native English writers put above the band (worst of our essay sets)"] = in_100(nn_above)

    unseen_bits = []
    if lomo:
        unseen_bits.append(f"On AI models left out of fitting it caught {in_100(mean_of(lomo, 'ai_above_band'))} AI texts "
                           f"on average ({in_100(lomo[worst_model]['ai_above_band'])} for the hardest, {worst_model}).")
    if newer:
        unseen_bits.append(f"On newer AI models (2024-2025) it never saw, it caught {in_100(newer.get('ai_above_band'))}.")
    if loko and worst_kind:
        unseen_bits.append(f"When one kind of writing was left out of fitting, {in_100(loko[worst_kind]['people_above_band'])} "
                           f"people's texts of the worst kind ({worst_kind}) were put above the band.")
    if rew_main:
        unseen_bits.append(f"AI text reworded by another AI tool is mostly missed "
                           f"(caught {in_100(rew_main.get('ai_above_band'))}).")
    if toefl:
        unseen_bits.append(f"Essays by non-native English students were put above the band {in_100(toefl.get('people_above_band'))} "
                           f"times and inside it {in_100(toefl.get('people_in_band'))} times.")
    unseen_bits.append("Expect more mistakes on writing unlike our examples.")
    srcs = set(exp.a.train_on.split("+"))
    parts = []
    if "hc3" in srcs:
        parts.append("answers from HC3 (ChatGPT, 2023)")
    if "mage" in srcs:
        parts.append("news, reviews, dialogue summaries and medical abstracts from MAGE (GPT-4)")
    if "raid" in srcs:
        parts.append("eight kinds of writing from RAID (11 AI models, 2019-2023)")
    cal = {
        "models": [PAIRS[a.perf], a.perf],
        "features": exp.feats,
        "mean": [float(v) for v in fit.mu], "std": [float(v) for v in fit.sd],
        "coef": [float(v) for v in fit.coef], "intercept": float(fit.b),
        "fill": {"spread": fit.fill.get("spread", 0.5)},
        "uncertain_band": list(fit.band),
        "min_words": a.min_words,
        "full_confidence_words": a.full_words,
        "reference": {k: ref_values(exp.tr, k) for k in ("top10", "bino", "spread")},
        "fit_recipe": {"train_on": a.train_on, "weights": a.weights, "band": a.band, "l2": a.l2,
                       "reworded_train": a.reworded_train, "n_train": res["n_train"],
                       "ai_models_fitted_on": seen,
                       "n_train_by_source": res["n_train_by_source"]},
        "heldout": {
            "how_often_wrong": how,
            "unseen_kinds": {k: {kk: v[kk] for kk in ("auroc", "tpr_at_1pct_fpr", "ece", "people_above_band",
                                                       "people_in_band", "ai_above_band", "ai_below_band") if kk in v}
                             for k, v in loko.items()},
            "metrics": sets,
            "by_source": res["by_source"],
            "by_model": res["by_model"],
            "by_kind": res["by_kind"],
            "people_only": po,
            "leave_one_model_out": lomo,
            "leave_one_kind_out": loko,
            "reworded": rew,
            "by_length": res["by_length"],
        },
        "unseen_note": " ".join(unseen_bits),
        "data_note": ("The chance is fitted on public examples by people and by AI: " + "; ".join(parts) +
                      ". People and AI count equally. It holds only for similar text with a similar mix; "
                      "where AI text is rarer, the chance is lower than shown."),
    }
    if a.fairness and Path(a.fairness).exists():
        cal["fairness"] = json.loads(Path(a.fairness).read_text())
    CAL_PATH.write_text(json.dumps(clean(cal), indent=2) + "\n", encoding="utf-8")
    print("wrote", CAL_PATH)


if __name__ == "__main__":
    main()
