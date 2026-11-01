"""Fetch essays by English learners and by other students, to measure fairness (nothing is committed).

    python -I scripts/text_fetch_fairness.py

Two corpora from the same research group, both CC BY 4.0, both downloadable without an account:
  PERSUADE 2.0 (https://github.com/scrosseye/persuade_corpus_2.0): about 25,000 essays by US
    students in grades 6-12, each marked with whether the student is an English language learner
    (ELL). The training file is a 617 MB CSV on Google Drive, linked from the README.
  ELLIPSE (https://github.com/scrosseye/ELLIPSE-Corpus): about 6,500 essays by English language
    learners in grades 8-12. The training CSV is 10 MB on GitHub.

From PERSUADE we take every-so-many essays by learners and, for each one, an essay by a
non-learner on the same prompt in the same grade, so the two groups are comparable.
Writes ~/.cache/markcheck/textdata/fairness/rows.jsonl, which text_data.py reads.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.home() / ".cache" / "markcheck" / "textdata"
PERSUADE = ROOT / "persuade" / "persuade_2.0_train.csv"
PERSUADE_URL = ("https://drive.usercontent.google.com/download?"
                "id=13phHyDzIsb0MHyJr6q-B-qIa9P2tM135&export=download&confirm=t")
ELLIPSE = ROOT / "ellipse" / "ELLIPSE_Final_github_train.csv"
ELLIPSE_URL = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/ELLIPSE_Final_github_train.csv"
N_PERSUADE_ELL = 400  # and as many matched non-learner essays
N_ELLIPSE = 300
MIN_WORDS = 30


def _get(path: Path, url: str) -> None:
    import requests

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)


def key(t: str) -> str:
    return hashlib.sha256(" ".join(t.split())[:300].encode()).hexdigest()[:16]


def main() -> None:
    _get(PERSUADE, PERSUADE_URL)
    _get(ELLIPSE, ELLIPSE_URL)
    csv.field_size_limit(10**9)
    rng = random.Random(23)
    # The PERSUADE file has one line per argument part; keep one line per essay.
    ess = {}
    for r in csv.DictReader(open(PERSUADE, encoding="utf-8", errors="replace")):
        ess.setdefault(r["essay_id_comp"], r)
    ess = [ess[k] for k in sorted(ess) if len(ess[k]["full_text"].split()) >= MIN_WORDS]
    ell = [e for e in ess if e["ell_status"].strip() == "Yes"]
    pool = defaultdict(list)
    for e in ess:
        if e["ell_status"].strip() == "No":
            pool[(e["prompt_name"], e["grade_level"])].append(e)
    for v in pool.values():
        rng.shuffle(v)
    rng.shuffle(ell)
    rows = []
    for e in ell:
        if sum(r["group"] == "ell" for r in rows) >= N_PERSUADE_ELL:
            break
        match = pool.get((e["prompt_name"], e["grade_level"]))
        if not match:
            continue
        m = match.pop()
        for d, group in ((e, "ell"), (m, "non_ell")):
            rows.append({"source": "persuade", "group": group, "grade": d["grade_level"],
                         "prompt": d["prompt_name"], "doc_id": "persuade:" + d["essay_id_comp"],
                         "text": d["full_text"]})
    seen = {key(r["text"]) for r in rows} | {key(e["full_text"]) for e in ess}
    el = [r for r in csv.DictReader(open(ELLIPSE, encoding="utf-8", errors="replace"))
          if len(r["full_text"].split()) >= MIN_WORDS]
    el.sort(key=lambda r: r["text_id_kaggle"])
    rng.shuffle(el)
    dup = 0
    kept = 0
    for r in el:
        if kept >= N_ELLIPSE:
            break
        if key(r["full_text"]) in seen:  # the same essay may also be in PERSUADE
            dup += 1
            continue
        rows.append({"source": "ellipse", "group": "ell", "grade": r["grade"], "prompt": r["prompt"],
                     "doc_id": "ellipse:" + r["text_id_kaggle"], "text": r["full_text"]})
        kept += 1
    out = ROOT / "fairness" / "rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(len(rows), "skipped as also in PERSUADE:", dup)
    print(Counter((r["source"], r["group"]) for r in rows))
    print(Counter((r["source"], r["group"], r["grade"]) for r in rows))


if __name__ == "__main__":
    main()
