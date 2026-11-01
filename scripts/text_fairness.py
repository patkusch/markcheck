"""How fair is the text check to different groups of writers, and what would a wider band cost?

    python scripts/text_fairness.py [--out-json PATH] [--out-md PATH] [--tops 0.84,0.88,0.90,0.92,0.95]

Uses the shipped calibration (src/markcheck/data/text_calibration.json) and the cached
scores (scripts/text_features.py), loaded with the same code as scripts/measure_text.py.
Every chance is worked out with markcheck's own scoring function, so these numbers are
what a user would see.

For each group of people's writing it reports, with 95% Wilson intervals (the range the
true rate is likely to be in, given how few texts we have):
  - flagged: share put above the band (the check says "looks more like AI writing");
  - can't tell: share put inside the band.
Only texts of at least the minimum length are counted, as only they get a number.

Then the band trade-off: for several possible tops of the band, how many non-native and
native writers' texts would be flagged, and how much held-out AI text would still be caught.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_text as mt  # noqa: E402  (same loading and splitting as the fit)
from markcheck.textcheck import probability  # noqa: E402  (the shipped scoring function)

CAL = ROOT / "src" / "markcheck" / "data" / "text_calibration.json"
SCRATCH = Path("/private/tmp/claude-501/-Users-patriciakusch-anthropic-academy/"
               "fd2a9156-a850-4003-80bd-1bd9d2c5eba6/scratchpad")

# Round-3 people's-writing sets, by split name.
SPLIT_GROUPS = {
    "toefl": ("TOEFL essays, non-native students", "non-native"),
    "us_students": ("US eighth-grade essays", "native"),
    "college": ("US college essays", "native"),
    "cs224n": ("Stanford course abstracts", "native"),
}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def native_status(r: dict) -> str | None:
    """'non-native', 'native' or None (unknown) for a people's text."""
    if r["split"] in SPLIT_GROUPS:
        return SPLIT_GROUPS[r["split"]][1]
    tag = " ".join(str(r.get(k) or "") for k in ("group", "kind", "source")).lower()
    word = r"(?<![a-z]){}(?![a-z])"
    if re.search(r"non[-_ ]?native|toefl", tag) or re.search(word.format("(esl|l2)"), tag):
        return "non-native"
    if re.search(word.format(r"non[-_ ]?ell"), tag) or re.search(word.format("native"), tag):
        return "native"
    if re.search(word.format("ell"), tag):
        return "non-native"
    return None


def group_name(r: dict) -> str | None:
    if r["split"] in SPLIT_GROUPS:
        return SPLIT_GROUPS[r["split"]][0]
    if r.get("group"):
        src = r.get("source") or r["split"]
        return f"{src}: {r['group']}"
    return None


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)] if n else None}


def scores(rows: list[dict], cal: dict) -> list[float]:
    fill = cal.get("fill", {})
    out = []
    for r in rows:
        x = [fill.get(k, 0.0) if r.get(k) is None else r[k] for k in cal["features"]]
        out.append(probability(x, cal))
    return out


def held_out_rows(cal: dict, perf: str) -> list[dict]:
    """Rows with the same train/held-out sides the fit used (measure_text.assign)."""
    rows = mt.load(perf)
    recipe = cal.get("fit_recipe") or {}
    if hasattr(mt, "assign"):
        train_on = tuple(str(recipe.get("train_on", "hc3+mage")).split("+"))
        rows = mt.assign(rows, train_on, bool(recipe.get("reworded_train", False)))
    else:  # round-3 measure_text
        rows = mt.mix_in_mage(rows)
        for r in rows:
            r["side"] = "train" if r["split"] == "train" else "test"
            r["set"] = r["split"]
    return [r for r in rows if r.get("side") == "test"]


def ai_set_name(r: dict) -> str | None:
    """Plain name of a held-out AI set, or None to leave it out.

    Altered text other than plain rewording (RAID's character and spelling changes) is left
    out on purpose: a table of which changes lower the catch rate would coach people."""
    names = {"mage": "GPT-4 (MAGE)", "mage_para": "reworded by another AI (MAGE)",
             "raid": "RAID (11 models)", "heldout": "ChatGPT (HC3)"}
    s = r.get("set") or r["split"]
    if s in names:
        return names[s]
    if s.startswith("raid_"):
        return "reworded by another AI (RAID)" if s in ("raid_reworded", "raid_paraphrase") else None
    if r.get("attack") not in (None, "none"):
        return None
    return s


def fingerprint(cal: dict) -> dict:
    """What the fairness numbers were computed with; a test checks it matches the shipped file."""
    return {"uncertain_band": list(cal["uncertain_band"]), "intercept": cal["intercept"],
            "coef": list(cal["coef"]), "min_words": cal["min_words"]}


def analyse(cal: dict, tops: list[float], perf: str | None = None) -> dict:
    perf = perf or cal["models"][1]
    minw = cal["min_words"]
    lo, hi = cal["uncertain_band"]
    test = held_out_rows(cal, perf)
    long_rows = [r for r in test if r["n_words"] >= minw]
    people = [r for r in long_rows if r["label"] == 0]
    p_people = dict(zip((id(r) for r in people), scores(people, cal)))

    groups: dict[str, list[dict]] = defaultdict(list)
    by_status: dict[str, list[dict]] = defaultdict(list)
    for r in people:
        g = group_name(r)
        if g:
            groups[g].append(r)
        s = native_status(r)
        if s:
            by_status[s].append(r)

    def describe(rs):
        ps = sorted(p_people[id(r)] for r in rs)
        return {"status": native_status(rs[0]),
                "flagged": rate(sum(p >= hi for p in ps), len(ps)),
                "cant_tell": rate(sum(lo < p < hi for p in ps), len(ps)),
                "median_chance": round(ps[len(ps) // 2], 4)}

    fairness = {g: describe(rs) for g, rs in sorted(groups.items())}
    pooled = {s: describe(rs) for s, rs in sorted(by_status.items())}
    skipped_short = sum(1 for r in test if r["label"] == 0 and group_name(r) and r["n_words"] < minw)

    ai: dict[str, list[dict]] = defaultdict(list)
    for r in long_rows:
        name = ai_set_name(r) if r["label"] == 1 else None
        if name:
            ai[name].append(r)
    p_ai = {k: scores(v, cal) for k, v in sorted(ai.items())}
    p_nn = [p_people[id(r)] for r in by_status.get("non-native", [])]
    p_nat = [p_people[id(r)] for r in by_status.get("native", [])]
    tradeoff = []
    for t in sorted(set([hi] + tops)):
        tradeoff.append({
            "band_top": t,
            "non_native_flagged": rate(sum(p >= t for p in p_nn), len(p_nn)),
            "native_flagged": rate(sum(p >= t for p in p_nat), len(p_nat)),
            "ai_caught": {k: rate(sum(p >= t for p in ps), len(ps)) for k, ps in p_ai.items()}})
    return {"computed_with": fingerprint(cal), "band": [lo, hi], "min_words": minw,
            "note": ("People's texts only, at least min_words long, never used for fitting. "
                     "flagged = put above the band; cant_tell = inside the band. "
                     "ci95 = 95% Wilson interval."),
            "groups": fairness, "pooled": pooled, "short_texts_left_out": skipped_short,
            "band_tradeoff": tradeoff}


def pct(r: dict | None) -> str:
    if not r or not r["n"]:
        return "-"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}% ({100 * lo:.1f}-{100 * hi:.1f})"


def markdown(res: dict) -> str:
    lo, hi = res["band"]
    out = [f"Band {lo:.0%}-{hi:.0%}, texts of {res['min_words']}+ words, people's texts never used for fitting.",
           "Rates with 95% Wilson intervals in brackets.", "",
           "| Group | Writers | Texts | Flagged (above band) | Can't tell (in band) | Median chance |",
           "|---|---|---|---|---|---|"]
    for name, d in list(res["groups"].items()) + [(f"ALL {k}", v) for k, v in res["pooled"].items()]:
        out.append(f"| {name} | {d['status'] or '?'} | {d['flagged']['n']} | {pct(d['flagged'])} | "
                   f"{pct(d['cant_tell'])} | {d['median_chance']:.1%} |")
    out += ["", f"Short texts left out (under {res['min_words']} words): {res['short_texts_left_out']}", "",
            "Band trade-off (flagged = at or above the top of the band):", ""]
    ai_names = sorted({k for row in res["band_tradeoff"] for k in row["ai_caught"]})
    out.append("| Band top | Non-native flagged | Native flagged | " + " | ".join(f"Caught: {k}" for k in ai_names) + " |")
    out.append("|---" * (3 + len(ai_names)) + "|")
    for row in res["band_tradeoff"]:
        out.append(f"| {row['band_top']:.0%} | {pct(row['non_native_flagged'])} | {pct(row['native_flagged'])} | "
                   + " | ".join(pct(row["ai_caught"].get(k)) for k in ai_names) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tops", default="0.84,0.88,0.90,0.92,0.95")
    ap.add_argument("--perf", default=None, help="performer model (default: the calibration's)")
    ap.add_argument("--out-json", default=str(SCRATCH / "fair_results.json"))
    ap.add_argument("--out-md", default=None, help="also write the tables here")
    ap.add_argument("--merge", action="store_true",
                    help="also store the results as 'fairness' in the shipped calibration (no other key changes)")
    a = ap.parse_args()
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    res = analyse(cal, [float(t) for t in a.tops.split(",")], a.perf)
    md = markdown(res)
    print(md)
    Path(a.out_json).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    if a.out_md:
        Path(a.out_md).write_text(md, encoding="utf-8")
    if a.merge:
        cal["fairness"] = res
        CAL.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")
        print("merged into", CAL)


if __name__ == "__main__":
    main()
