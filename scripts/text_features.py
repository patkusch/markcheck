"""Score every text in splits.jsonl with one model pair; cache the numbers.

    python scripts/text_features.py OBSERVER PERFORMER
Writes ~/.cache/markcheck/textdata/feat_<performer>.jsonl (resumes if stopped).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markcheck.textcheck.features import sentence_length_spread, words  # noqa: E402
from markcheck.textcheck.scorer import PairScorer  # noqa: E402

DATA = Path.home() / ".cache" / "markcheck" / "textdata"


def main() -> None:
    obs, perf = sys.argv[1], sys.argv[2]
    out = DATA / f"feat_{perf.replace('/', '__')}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in open(out)}
    sc = PairScorer(obs, perf, local_only=False)
    rows = [json.loads(line) for line in open(DATA / "splits.jsonl")]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 1  # score every Nth text, for quick comparisons
    rows = rows[::every]
    t0 = time.time()
    with open(out, "a") as fh:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            st = sc.score(r["text"])
            fh.write(json.dumps({
                "id": r["id"], "n_words": len(words(r["text"])), "n_tokens": st.n_tokens,
                "bino": st.binoculars, "logppl_obs": st.logppl_obs, "logppl_perf": st.logppl_perf,
                "xent": st.xent, "fdg": st.fdg, "top10": st.top10,
                "spread": sentence_length_spread(r["text"]),
            }) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(i, len(rows), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
