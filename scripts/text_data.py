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

Round 4 adds, after all the rows above (so their ids never change):
  newmodels  DetectRL-X (GPT-4o, Gemini 2.5 Flash, DeepSeek-V3, Qwen-Max) and Beemo (open chat models)
  persuade   US school essays by English learners and matched non-learners (fairness)
  ellipse    essays by English learners (fairness)
  raid       a balanced sample of RAID: 11 AI models and people, 8 kinds of writing
  raid_para  RAID texts after its "paraphrase" rewording attack
Fetch them first with scripts/text_fetch_raid.py, text_fetch_newmodels.py and
text_fetch_fairness.py (run with python -I). Use --old-only to build only the rows above.

Every row has: split, domain, label (1 = AI), text, id, and
  source  where it came from (hc3, mage, raid, detectrlx, beemo, persuade, ...)
  model   "human" or the AI model that wrote it
  kind    the kind of writing (qa, news, reviews, abstracts, essays, ...)
  attack  "none" or "paraphrase" (reworded by another AI)
  doc_id  the source document; split by this so an AI text and the person's text it was
          prompted from never land on different sides
  group   only on fairness rows: who wrote it (ell, non_ell, non_native, us)
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
MAGE_KIND = {"cnn": "news", "imdb": "reviews", "pubmed": "medical_qa", "dialogsum": "dialogue"}
OLD_FAIR = {"toefl": ("essays", "non_native"), "us_students": ("essays", "us"),
            "college": ("essays", "us"), "cs224n": ("abstracts", "us")}


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def hx(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def unwrap(t: str) -> str:
    """Join lines broken only to fit a page width (single line breaks inside a paragraph).

    RAID's abstracts by people keep arXiv's hard line breaks; its AI abstracts do not.
    Left alone, a detector could learn the line breaks instead of the writing.
    """
    return "\n\n".join(" ".join(p.split("\n")) for p in t.split("\n\n"))


def load(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path}: run the matching scripts/text_fetch_*.py first, or use --old-only")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def round4_rows() -> list[dict]:
    rows = []
    for r in load(DATA / "newmodels" / "rows.jsonl"):
        t = normalise(r["text"])
        if len(words(t)) < MIN_WORDS:
            continue
        rows.append({"split": "newmodels", "domain": r["kind"], "label": r["label"], "text": t,
                     "source": r["source"], "model": r["model"], "kind": r["kind"], "attack": "none",
                     "doc_id": r["doc_id"]})
    for r in load(DATA / "fairness" / "rows.jsonl"):
        t = normalise(r["text"])
        if len(words(t)) < MIN_WORDS:
            continue
        rows.append({"split": r["source"], "domain": r["source"], "label": 0, "text": t,
                     "source": r["source"], "model": "human", "kind": "essays", "attack": "none",
                     "doc_id": r["doc_id"], "group": r["group"], "grade": r["grade"],
                     "prompt": r["prompt"]})
    # RAID comes last: it is the slowest to fetch, so the other rows can be scored first.
    if not (DATA / "raid" / "rows.jsonl").exists():
        print("warning: no RAID sample yet; run scripts/text_fetch_raid.py", file=sys.stderr)
        return rows
    for r in load(DATA / "raid" / "rows.jsonl"):
        t = normalise(unwrap(r["generation"]) if r["domain"] == "abstracts" else r["generation"])
        if len(words(t)) < MIN_WORDS:
            continue
        rows.append({"split": "raid" if r["attack"] == "none" else "raid_para", "domain": r["domain"],
                     "label": 0 if r["model"] == "human" else 1, "text": t, "source": "raid",
                     "model": r["model"], "kind": r["domain"], "attack": r["attack"],
                     "doc_id": "raid:" + r["source_id"], "decoding": r["decoding"] or "none",
                     "raid_id": r["id"]})
    return rows


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
            meta = {"source": "hc3", "kind": "qa", "attack": "none",
                    "doc_id": f"hc3:{dom}:{hx(q['question'])}"}
            rows.append({"split": split, "domain": dom, "label": 0, "text": hum[0], "model": "human", **meta})
            rows.append({"split": split, "domain": dom, "label": 1, "text": ai[0], "model": "chatgpt", **meta})
            kept += 1
    for fname, split in (("test_ood_set_gpt.csv", "mage"), ("test_ood_set_gpt_para.csv", "mage_para")):
        with open(DATA / fname, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                t = normalise(r["text"])
                if len(words(t)) < MIN_WORDS:
                    continue
                # MAGE: label 1 = written by a person, 0 = machine. We use 1 = AI.
                # The people's texts are the same in both files, so their doc_id matches.
                dom = r["src"].split("_")[0]
                ai = 0 if r["label"] == "1" else 1
                rows.append({"split": split, "domain": dom, "label": ai, "text": t, "source": "mage",
                             "model": "gpt-4" if ai else "human", "kind": MAGE_KIND.get(dom, dom),
                             "attack": "paraphrase" if ai and split == "mage_para" else "none",
                             "doc_id": "mage:" + hx(t)})
    for d in json.load(open(DATA / "toefl.json", encoding="utf-8")):
        t = normalise(d["document"])
        rows.append({"split": "toefl", "domain": "toefl", "label": 0, "text": t})
    # More writing by people from the same study, for a fairness comparison (no licence file;
    # used only for measuring): US eighth-grade essays, US college essays, Stanford course abstracts.
    for fname, split in (("hewlett.json", "us_students"), ("CollegeEssay_real_70.json", "college"),
                         ("CS224N_real_145.json", "cs224n")):
        if (DATA / fname).exists():
            for d in json.load(open(DATA / fname, encoding="utf-8")):
                t = normalise(d["document"])
                rows.append({"split": split, "domain": split, "label": 0, "text": t})
    for r in rows:
        if r["split"] in OLD_FAIR:
            kind, group = OLD_FAIR[r["split"]]
            r.update({"source": r["split"], "model": "human", "kind": kind, "attack": "none",
                      "doc_id": f"{r['split']}:{hx(r['text'])}", "group": group})
    if "--old-only" not in sys.argv:
        rows += round4_rows()
    for i, r in enumerate(rows):
        r["id"] = i
    with open(DATA / "splits.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), Counter((r["split"], r["label"]) for r in rows))


if __name__ == "__main__":
    main()
