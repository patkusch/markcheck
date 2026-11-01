"""Build the fixed train / held-out splits for the text check.

Run once after downloading the public data (see docs/rounds.md):
    python scripts/text_data.py
Writes ~/.cache/markcheck/textdata/splits.jsonl. Nothing here is committed:
HC3 is CC-BY-SA-4.0, MAGE is Apache-2.0 / CC BY 4.0, and the TOEFL essays
(Liang et al. 2023) have no licence file, so we only use them locally.

Splits:
  train    HC3 questions whose hash is even: one person's answer and one ChatGPT answer each
  heldout  HC3 questions whose hash is odd (never used for fitting)
  mage     MAGE out-of-domain GPT-4 test file (held-out, other topics, newer model)
  mage_para  the same after rewording (held-out)
  toefl    91 essays by non-native English students (held-out, all by people)
  us_students, college, cs224n  essays and abstracts by US students from the same study (held-out)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markcheck.textcheck.features import normalise, words  # noqa: E402

DATA = Path.home() / ".cache" / "markcheck" / "textdata"
HC3 = {"finance": 300, "medicine": 300, "wiki_csai": 300, "reddit_eli5": 300}
MIN_WORDS = 30


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def main() -> None:
    rows = []
    rng = random.Random(7)
    for dom, n in HC3.items():
        qs = [json.loads(line) for line in open(DATA / f"{dom}.jsonl", encoding="utf-8")]
        rng.shuffle(qs)
        kept = 0
        for q in qs:
            if kept >= n:
                break
            hum = [normalise(a) for a in q["human_answers"] if len(words(a)) >= MIN_WORDS]
            ai = [normalise(a) for a in q["chatgpt_answers"] if len(words(a)) >= MIN_WORDS]
            if not hum or not ai:
                continue
            split = "train" if h(q["question"]) % 2 == 0 else "heldout"
            rows.append({"split": split, "domain": dom, "label": 0, "text": hum[0]})
            rows.append({"split": split, "domain": dom, "label": 1, "text": ai[0]})
            kept += 1
    for fname, split in (("test_ood_set_gpt.csv", "mage"), ("test_ood_set_gpt_para.csv", "mage_para")):
        with open(DATA / fname, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                t = normalise(r["text"])
                if len(words(t)) < MIN_WORDS:
                    continue
                # MAGE: label 1 = written by a person, 0 = machine. We use 1 = AI.
                rows.append({"split": split, "domain": r["src"].split("_")[0],
                             "label": 0 if r["label"] == "1" else 1, "text": t})
    for d in json.load(open(DATA / "toefl.json", encoding="utf-8")):
        rows.append({"split": "toefl", "domain": "toefl", "label": 0, "text": normalise(d["document"])})
    # More writing by people from the same study, for a fairness comparison (no licence file;
    # used only for measuring): US eighth-grade essays, US college essays, Stanford course abstracts.
    for fname, split in (("hewlett.json", "us_students"), ("CollegeEssay_real_70.json", "college"),
                         ("CS224N_real_145.json", "cs224n")):
        if (DATA / fname).exists():
            for d in json.load(open(DATA / fname, encoding="utf-8")):
                rows.append({"split": split, "domain": split, "label": 0, "text": normalise(d["document"])})
    for i, r in enumerate(rows):
        r["id"] = i
    with open(DATA / "splits.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter((r["split"], r["label"]) for r in rows))


if __name__ == "__main__":
    main()
