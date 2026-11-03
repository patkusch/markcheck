"""Score a sample of held-out texts with a second model pair, to compare pairs.

    python scripts/text_compare_pairs.py OBSERVER PERFORMER --source detectrlx+beemo --limit 300

Scoring every text with a bigger pair takes hours on a laptop. This scores a
fixed, evenly spread sample from the named sources instead. It writes
~/.cache/markcheck/textdata/pair_<performer>.jsonl (resumes if stopped).
Compare the result with the shipped pair using --report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA = Path.home() / ".cache" / "markcheck" / "textdata"
SHIPPED = "HuggingFaceTB/SmolLM2-360M-Instruct"


def sample(sources: set[str], limit: int, min_words: int = 100) -> list[dict]:
    """An evenly spread, repeatable sample of held-out, unreworded texts."""
    rows = []
    for line in open(DATA / "splits.jsonl"):
        r = json.loads(line)
        if r["split"] == "train" or r["source"] not in sources or r["attack"] != "none":
            continue
        if len(r["text"].split()) < min_words:
            continue
        rows.append(r)
    step = max(1, len(rows) // limit)
    return rows[::step][:limit]


def auroc(scores: list[float], labels: list[int]) -> float:
    """Chance a random AI text scores higher than a random person's text."""
    ai = [s for s, y in zip(scores, labels) if y == 1]
    people = [s for s, y in zip(scores, labels) if y == 0]
    if not ai or not people:
        return float("nan")
    wins = sum((a > p) + 0.5 * (a == p) for a in ai for p in people)
    return wins / (len(ai) * len(people))


def score(args) -> None:
    from markcheck.textcheck.scorer import PairScorer

    out = DATA / f"pair_{args.performer.replace('/', '__')}.jsonl"
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    rows = sample(set(args.source.split("+")), args.limit)
    sc = PairScorer(args.observer, args.performer, local_only=False, threads=args.threads)
    t0, n = time.time(), 0
    with open(out, "a") as fh:
        for r in rows:
            if r["id"] in done:
                continue
            st = sc.score(r["text"])
            fh.write(json.dumps({"id": r["id"], "bino": st.binoculars, "n_tokens": st.n_tokens}) + "\n")
            fh.flush()
            n += 1
    if n:
        print(f"scored {n} texts in {time.time() - t0:.0f}s ({(time.time() - t0) / n:.1f}s each)")


def report(args) -> None:
    def load(path: Path) -> dict[int, float]:
        return {d["id"]: d["bino"] for d in map(json.loads, open(path))}

    new = load(DATA / f"pair_{args.performer.replace('/', '__')}.jsonl")
    old = load(DATA / f"feat_{SHIPPED.replace('/', '__')}.jsonl")
    rows = [r for r in sample(set(args.source.split("+")), args.limit) if r["id"] in new and r["id"] in old]
    print(f"{len(rows)} texts scored by both pairs")
    groups: dict[str, list[dict]] = {"all": rows}
    people = [r for r in rows if r["label"] == 0]
    for r in rows:
        if r["label"] == 1:
            groups.setdefault(r["model"], []).append(r)
    for name, rs in groups.items():
        if name != "all":
            rs = rs + people
        labels = [r["label"] for r in rs]
        # A lower Binoculars score means more AI-like, so flip the sign.
        a_old = auroc([-old[r["id"]] for r in rs], labels)
        a_new = auroc([-new[r["id"]] for r in rs], labels)
        print(f"  {name:28s} n={len(rs):4d}  shipped pair {a_old:.3f}  new pair {a_new:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("observer")
    ap.add_argument("performer")
    ap.add_argument("--source", default="detectrlx+beemo", help="sources joined by +")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="compare with the shipped pair, no scoring")
    args = ap.parse_args()
    report(args) if args.report else score(args)


if __name__ == "__main__":
    main()
