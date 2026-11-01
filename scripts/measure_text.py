"""Fit the text check's calibration on the train split and measure it on held-out texts.

    python scripts/measure_text.py PERFORMER [--features a,b,c] [--min-words N] [--write]

Needs splits.jsonl (scripts/text_data.py) and the cached scores
(scripts/text_features.py). With --write, saves src/markcheck/data/text_calibration.json.
Numbers: AUROC (the chance a random AI text scores higher than a random person's
text), share of AI caught when 1 in 100 people's texts is wrongly flagged,
calibration error (average gap between the stated chance and how often it came true).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DATA = Path.home() / ".cache" / "markcheck" / "textdata"
ROOT = Path(__file__).resolve().parents[1]
LENGTH_BINS = [(30, 59), (60, 99), (100, 149), (150, 249), (250, 10**6)]
PAIRS = {
    "HuggingFaceTB/SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M",
    "gpt2-medium": "gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B",
}


def load(perf: str):
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(DATA / "splits.jsonl")}
    out = []
    for line in open(DATA / f"feat_{perf.replace('/', '__')}.jsonl"):
        f = json.loads(line)
        r = dict(rows[f["id"]])
        r.update(f)
        r["log_tokens"] = math.log(max(r["n_tokens"], 1))
        out.append(r)
    return out


def text_half(t: str) -> int:
    import hashlib
    return int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % 2


def mix_in_mage(rows, hold_domain=None):
    """Move half of MAGE (by text hash) into training. Reworded copies of people's texts in
    the training half are dropped from the rewording test, so no text is both fitted and tested."""
    train_people = {r["text"] for r in rows if r["split"] == "mage" and r["label"] == 0
                    and text_half(r["text"]) == 0}
    out = []
    for r in rows:
        r = dict(r)
        if r["split"] == "mage" and text_half(r["text"]) == 0:
            if hold_domain and r["domain"] == hold_domain:
                continue
            r["split"] = "train"
        elif r["split"] == "mage_para" and r["label"] == 0 and r["text"] in train_people:
            continue
        out.append(r)
    return out


def matrix(rows, feats, fill):
    return np.array([[fill[k] if r[k] is None else r[k] for k in feats] for r in rows], float)


def fit_logistic(X, y, l2=1e-2, iters=50, weights=None):
    n, d = X.shape
    sw = np.ones(n) if weights is None else np.asarray(weights, float)
    A = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (sw * (p - y)) + l2 * np.r_[w[:-1], 0]
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + l2 * np.diag(np.r_[np.ones(d), 0])
        w -= np.linalg.solve(H, g)
    return w[:-1], w[-1]


def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.r_[pos, neg][order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(s, y, fpr=0.01):
    s, y = np.asarray(s), np.asarray(y)
    neg = np.sort(s[y == 0])
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(math.ceil((1 - fpr) * len(neg))) - 1)]
    return float((s[y == 1] > thr).mean())


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
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
        d["people_above_band"] = float((hum >= hi).mean())
        d["people_in_band"] = float(((hum > lo) & (hum < hi)).mean())
    if len(ai):
        d["ai_below_band"] = float((ai <= lo).mean())
        d["ai_above_band"] = float((ai >= hi).mean())
        d["ai_in_band"] = float(((ai > lo) & (ai < hi)).mean())
    return d


def fmt(d):
    keys = ["n", "auroc", "tpr_at_1pct_fpr", "ece", "people_above_band", "people_in_band",
            "ai_above_band", "ai_below_band"]
    return "  ".join(f"{k}={d[k]:.3f}" if isinstance(d.get(k), float) else f"{k}={d.get(k, '-')}"
                     for k in keys if k in d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perf")
    ap.add_argument("--features", default="bino,top10,spread")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--full-words", type=int, default=150)
    ap.add_argument("--band", default="0.35,0.65")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--train-on", default="hc3", choices=["hc3", "hc3+mage"],
                    help="hc3+mage also fits on half of MAGE (by text hash); the other half stays held out")
    ap.add_argument("--hold-domain", default=None, help="leave this MAGE kind of writing out of fitting")
    a = ap.parse_args()
    feats = a.features.split(",")
    rows = load(a.perf)
    if a.train_on == "hc3+mage":
        rows = mix_in_mage(rows, a.hold_domain)
    tr = [r for r in rows if r["split"] == "train" and r["n_words"] >= a.min_words]
    fill = {k: float(np.median([r[k] for r in tr if r[k] is not None])) for k in feats}
    Xtr = matrix(tr, feats, fill)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    ytr = np.array([r["label"] for r in tr], float)
    # Give each source (HC3, MAGE) equal total weight, so the bigger one does not set the curve alone.
    src = np.array([0 if r["domain"] in ("finance", "medicine", "wiki_csai", "reddit_eli5") else 1 for r in tr])
    wts = np.where(src == 0, 0.5 / max((src == 0).sum(), 1), 0.5 / max((src == 1).sum(), 1)) * len(tr)
    if (src == 1).sum() == 0:
        wts = np.ones(len(tr))
    coef, b = fit_logistic((Xtr - mu) / sd, ytr, weights=wts)

    def predict(rs):
        X = (matrix(rs, feats, fill) - mu) / sd
        return 1 / (1 + np.exp(-(X @ coef + b)))

    if a.band == "auto":
        # Top of the band: no lower than 65%, and high enough that at most 1 in 100 people's
        # training texts reach it. Bottom: no higher than 35%, and low enough that at most
        # 1 in 100 AI training texts fall under it.
        ptr = predict(tr)
        hi = max(0.65, float(np.quantile(ptr[ytr == 0], 0.99)))
        lo = min(0.35, float(np.quantile(ptr[ytr == 1], 0.01)))
        band = (round(lo, 2), round(hi + 0.005, 2))
    else:
        band = tuple(float(v) for v in a.band.split(","))
    print("band", band)
    print(f"pair {PAIRS.get(a.perf, '?')} / {a.perf}; features {feats}; min words {a.min_words}; "
          f"train on {a.train_on}; held-out domain {a.hold_domain}")
    print("coef", dict(zip(feats, np.round(coef, 3))), "intercept", round(float(b), 3))
    results = {}
    for split in ["train", "heldout", "mage", "mage_para", "toefl"]:
        rs = [r for r in rows if r["split"] == split and r["n_words"] >= a.min_words]
        if not rs:
            continue
        p, y = predict(rs), np.array([r["label"] for r in rs])
        results[split] = summary(p, y, band)
        print(f"{split:10s}", fmt(results[split]))
        if split in ("heldout", "mage"):
            for dom in sorted({r["domain"] for r in rs}):
                m = np.array([r["domain"] == dom for r in rs])
                print(f"   {dom:12s}", fmt(summary(p[m], y[m], band)))
    print("by length (all held-out HC3 + MAGE, no minimum):")
    allh = [r for r in rows if r["split"] in ("heldout", "mage")]
    by_len = {}
    for lo_w, hi_w in LENGTH_BINS:
        rs = [r for r in allh if lo_w <= r["n_words"] <= hi_w]
        if rs:
            p, y = predict(rs), np.array([r["label"] for r in rs])
            by_len[f"{lo_w}-{hi_w if hi_w < 10**6 else ''}"] = summary(p, y, band)
            print(f"   {lo_w:4d}-{hi_w:<7d}", fmt(by_len[f"{lo_w}-{hi_w if hi_w < 10**6 else ''}"]))
    tf = [r for r in rows if r["split"] == "toefl"]
    if tf:
        pt = predict(tf)
        print(f"toefl all lengths: n={len(tf)} above band={np.mean(pt >= band[1]):.3f} "
              f"in band={np.mean((pt > band[0]) & (pt < band[1])):.3f} median p={np.median(pt):.3f}")
    if a.write:
        write(a, feats, fill, mu, sd, coef, b, band, rows, results, by_len, predict, tr)


def ref_values(tr, key):
    return {"ai": float(np.median([r[key] for r in tr if r["label"] == 1 and r[key] is not None])),
            "person": float(np.median([r[key] for r in tr if r["label"] == 0 and r[key] is not None]))}


def in_100(x):
    return f"{round(100 * x)} in 100" if x >= 0.005 else "fewer than 1 in 100"


def unseen_kinds(a) -> dict:
    """Refit with each MAGE kind of writing left out, and test on that kind: how the check
    does on writing it has never seen. Runs this script again once per kind."""
    import re
    import subprocess
    import sys
    out = {}
    for dom in ("cnn", "dialogsum", "imdb", "pubmed"):
        res = subprocess.run([sys.executable, __file__, a.perf, "--features", a.features, "--min-words",
                              str(a.min_words), "--band", a.band, "--train-on", "hc3+mage",
                              "--hold-domain", dom], capture_output=True, text=True, check=True).stdout
        line = next(ln for ln in res.splitlines() if ln.startswith(f"   {dom} "))
        out[dom] = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", line) if k != "n"}
    return out


KIND_NAMES = {"cnn": "news", "dialogsum": "dialogue summaries", "imdb": "film reviews", "pubmed": "medical abstracts"}


def write(a, feats, fill, mu, sd, coef, b, band, rows, results, by_len, predict, tr):
    h, m, mp = results["heldout"], results["mage"], results.get("mage_para", {})
    tf = results.get("toefl", {})
    unseen = unseen_kinds(a) if a.train_on == "hc3+mage" else {}
    worst = max(unseen, key=lambda d: unseen[d]["people_above_band"]) if unseen else None
    cal = {
        "models": [PAIRS[a.perf], a.perf],
        "features": feats,
        "mean": [float(v) for v in mu], "std": [float(v) for v in sd],
        "coef": [float(v) for v in coef], "intercept": float(b),
        "fill": {"spread": fill.get("spread", 0.5)},
        "uncertain_band": list(band),
        "min_words": a.min_words,
        "full_confidence_words": a.full_words,
        "reference": {k: ref_values(tr, k) for k in ("top10", "bino", "spread")},
        "heldout": {
            "how_often_wrong": {
                "people's texts put above the band (similar topics)": in_100(h["people_above_band"]),
                "AI texts caught above the band (similar topics)": in_100(h["ai_above_band"]),
                "AI texts put below the band (similar topics)": in_100(h["ai_below_band"]),
                "people's texts put above the band (news, reviews, summaries, abstracts; GPT-4)": in_100(m["people_above_band"]),
                "AI texts caught above the band (news, reviews, summaries, abstracts; GPT-4)": in_100(m["ai_above_band"]),
                "AI texts put below the band (news, reviews, summaries, abstracts; GPT-4)": in_100(m["ai_below_band"]),
                "AI texts reworded by another AI, caught above the band": in_100(mp.get("ai_above_band", 0)),
                "essays by non-native English students put above the band": in_100(tf.get("people_above_band", 0)),
                **({f"people's {KIND_NAMES[worst]} put above the band when that kind of writing was left "
                    f"out of fitting (the worst case we tried)": in_100(unseen[worst]["people_above_band"])}
                   if worst else {}),
            },
            "unseen_kinds": unseen,
            "metrics": results,
            "by_length": by_len,
        },
        "data_note": ("The chance is fitted on public examples by people and by AI: answers from HC3 "
                      "(ChatGPT, 2023) and news, reviews, dialogue summaries and medical abstracts from MAGE "
                      "(GPT-4), half AI and half people. It holds only for similar text with a similar mix; "
                      "where AI text is rarer, the chance is lower than shown."),
    }
    out = ROOT / "src" / "markcheck" / "data" / "text_calibration.json"
    out.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
