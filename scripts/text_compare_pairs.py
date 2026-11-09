"""Score a sample of held-out texts with a second model pair, to compare pairs.

    python scripts/text_compare_pairs.py OBSERVER PERFORMER --source detectrlx+beemo --limit 300

Scoring every text with a bigger pair takes hours on a laptop. This scores a
fixed, evenly spread sample from the named sources instead. It writes
~/.cache/markcheck/textdata/pair_<performer>.jsonl (resumes if stopped).
Compare the result with the shipped pair using --report.

On a machine without the full data set (such as a GitHub runner), pass --rows with
the newer-model file from scripts/text_fetch_newmodels.py, score both pairs, and
compare them with --report --against:

    python scripts/text_compare_pairs.py HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-360M-Instruct \
        --rows ~/.cache/markcheck/textdata/newmodels/rows.jsonl --name shipped
    python scripts/text_compare_pairs.py Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct \
        --rows ~/.cache/markcheck/textdata/newmodels/rows.jsonl --name qwen1.5b
    python scripts/text_compare_pairs.py x x --rows ... --name qwen1.5b --against shipped --report
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


def read_rows(path: Path | None) -> list[dict]:
    """The full data set, or a plain rows file such as the newer-model one.

    Rows from a plain file are all held out and unreworded, and get an id from their own fields.
    """
    if path is None:
        return [json.loads(line) for line in open(DATA / "splits.jsonl")]
    rows = []
    for line in open(path):
        r = json.loads(line)
        r.setdefault("split", "newmodels")
        r.setdefault("attack", "none")
        r.setdefault("id", f"{r['doc_id']}:{r['label']}:{r.get('model', '')}")
        rows.append(r)
    return rows


def sample(sources: set[str], limit: int, min_words: int = 100, path: Path | None = None) -> list[dict]:
    """An evenly spread, repeatable sample of held-out, unreworded texts: half AI, half people's."""
    rows = []
    for r in read_rows(path):
        if r["split"] == "train" or r["source"] not in sources or r["attack"] != "none":
            continue
        if len(r["text"].split()) < min_words:
            continue
        rows.append(r)
    # Alternate AI and people's texts, so a run stopped part way still has both to compare.
    ai = [r for r in rows if r["label"] == 1]
    people = [r for r in rows if r["label"] == 0]
    ai = ai[::max(1, len(ai) // (limit - limit // 2))][:limit - limit // 2]
    people = people[::max(1, len(people) // (limit // 2 or 1))][:limit // 2]
    out = []
    for i in range(max(len(ai), len(people))):
        out += [r for r in (ai[i] if i < len(ai) else None, people[i] if i < len(people) else None) if r]
    return out[:limit]


def auroc(scores: list[float], labels: list[int]) -> float:
    """Chance a random AI text scores higher than a random person's text."""
    ai = [s for s, y in zip(scores, labels) if y == 1]
    people = [s for s, y in zip(scores, labels) if y == 0]
    if not ai or not people:
        return float("nan")
    wins = sum((a > p) + 0.5 * (a == p) for a in ai for p in people)
    return wins / (len(ai) * len(people))


def pair_file(args) -> Path:
    """Where one pair's scores are cached. --name keeps the file name short for local folders."""
    name = args.name or args.performer.replace("/", "__")
    return DATA / f"pair_{name}.jsonl"


def score(args) -> None:
    from markcheck.textcheck.scorer import PairScorer

    out = pair_file(args)
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    rows = sample(set(args.source.split("+")), args.limit, path=args.rows)
    sc = PairScorer(args.observer, args.performer, local_only=False, threads=args.threads, half=args.half)
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

    new = load(pair_file(args))
    old = load(DATA / (f"pair_{args.against}.jsonl" if args.against else f"feat_{SHIPPED.replace('/', '__')}.jsonl"))
    rows = [r for r in sample(set(args.source.split("+")), args.limit, path=args.rows)
            if r["id"] in new and r["id"] in old]
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
    ap.add_argument("--half", action="store_true", help="half precision: half the memory, for bigger pairs")
    ap.add_argument("--name", default=None, help="short name for the cached scores file")
    ap.add_argument("--rows", type=Path, default=None,
                    help="read texts from this rows file instead of the full data set")
    ap.add_argument("--against", default=None,
                    help="with --report: compare with this pair's cached scores (its --name)")
    ap.add_argument("--report", action="store_true", help="compare with the shipped pair, no scoring")
    args = ap.parse_args()
    report(args) if args.report else score(args)


if __name__ == "__main__":
    main()
