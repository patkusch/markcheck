"""Fetch AI text from newer models for the text check (nothing is committed).

    python -I scripts/text_fetch_newmodels.py

Two openly licensed sources:
  DetectRL-X (https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X, MIT): people's texts paired
    with texts written by GPT-4o, Gemini 2.5 Flash, DeepSeek-V3 and Qwen-Max (2024-2025 models),
    in six kinds of writing. Its "general" file is 879 MB and holds eight languages. It is
    sorted by language, so we read only the two stretches that hold English (about 95 MB).
  Beemo (https://huggingface.co/datasets/toloka/beemo, MIT): answers by ten open chat models
    (Llama 2, Gemma, Mistral, Mixtral, Zephyr, Tulu 2) to prompts that people also answered.
    One 8 MB file.

Writes ~/.cache/markcheck/textdata/newmodels/rows.jsonl, which text_data.py reads.
The sample is fixed by a seed.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.home() / ".cache" / "markcheck" / "textdata"
DRX_URL = "https://huggingface.co/datasets/WUJUNCHAO/DetectRL-X/resolve/main/Binary/binary_general_open.json"
# Byte ranges that hold the English rows, found by reading 60 KB every 7 MB (see docs/text-research.md).
DRX_RANGES = [(183_000_000, 265_000_000), (754_000_000, 777_000_000)]
BEEMO_URL = "https://huggingface.co/datasets/toloka/beemo/resolve/main/data/train-00000-of-00001.parquet"
DRX_PER_CELL = 30    # per (model, kind): 4 models x 6 kinds x 30 = 720 AI texts, plus their people's texts
BEEMO_PER_MODEL = 30  # 10 models x 30 = 300 AI texts, plus the people's answers to the same prompts
MIN_WORDS = 30


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def fetch_detectrlx() -> list[dict]:
    import requests

    out_f = ROOT / "detectrlx" / "english.jsonl"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    if not out_f.exists():
        dec = json.JSONDecoder()
        seen = set()
        with open(out_f.with_suffix(".tmp"), "w", encoding="utf-8") as fh:
            for a, b in DRX_RANGES:
                for s in range(a, b, 4_000_000):  # 4 MB at a time, overlapping by 200 KB
                    t = requests.get(DRX_URL, headers={"Range": f"bytes={s}-{min(b, s + 4_200_000)}"},
                                     timeout=300).content.decode("utf-8", "replace")
                    for m in re.finditer(r'\{"human_written_text": ', t):
                        try:
                            d, _ = dec.raw_decode(t, m.start())
                        except ValueError:
                            continue
                        if d.get("lang") != "english":
                            continue
                        k = _h(d["human_written_text"] + "\x00" + d["llm_generated_text"])
                        if k in seen:
                            continue
                        seen.add(k)
                        fh.write(json.dumps(d) + "\n")
                    print("detectrlx", s, len(seen), flush=True)
        out_f.with_suffix(".tmp").rename(out_f)
    return [json.loads(line) for line in open(out_f, encoding="utf-8")]


def fetch_beemo() -> list[dict]:
    import pyarrow.parquet as pq
    import requests

    f = ROOT / "beemo" / "train.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_bytes(requests.get(BEEMO_URL, timeout=300).content)
    return pq.read_table(f).to_pylist()


def nwords(t: str) -> int:
    return len(t.split())


def main() -> None:
    rng = random.Random(11)
    rows = []
    drx = fetch_detectrlx()
    print("detectrlx english rows", len(drx), Counter((d["model"], d["domain"]) for d in drx))
    cells = defaultdict(list)
    for d in drx:
        if nwords(d["llm_generated_text"]) >= MIN_WORDS and nwords(d["human_written_text"]) >= MIN_WORDS:
            cells[(d["model"], d["domain"])].append(d)
    humans = {}
    for key in sorted(cells):
        c = sorted(cells[key], key=lambda d: _h(d["llm_generated_text"]))
        rng.shuffle(c)
        for d in c[:DRX_PER_CELL]:
            doc = "detectrlx:" + _h(d["human_written_text"])
            kind = d["domain"].lower()
            rows.append({"source": "detectrlx", "model": d["model"], "kind": kind, "label": 1,
                         "doc_id": doc, "text": d["llm_generated_text"], "orig_split": d["split"]})
            humans.setdefault(doc, {"source": "detectrlx", "model": "human", "kind": kind, "label": 0,
                                    "doc_id": doc, "text": d["human_written_text"], "orig_split": d["split"]})
    rows += [humans[k] for k in sorted(humans)]
    bee = fetch_beemo()
    bycat = defaultdict(list)
    for d in bee:
        if nwords(d["model_output"]) >= MIN_WORDS:
            bycat[d["model"]].append(d)
    bhum = {}
    for model in sorted(bycat):
        c = sorted(bycat[model], key=lambda d: d["prompt_id"])
        rng.shuffle(c)
        for d in c[:BEEMO_PER_MODEL]:
            doc = "beemo:" + d["prompt_id"]
            kind = d["category"].lower().replace(" ", "_")
            rows.append({"source": "beemo", "model": model.split("/")[-1], "kind": kind, "label": 1,
                         "doc_id": doc, "text": d["model_output"]})
            if nwords(d["human_output"]) >= MIN_WORDS:
                bhum.setdefault(doc, {"source": "beemo", "model": "human", "kind": kind, "label": 0,
                                      "doc_id": doc, "text": d["human_output"]})
    rows += [bhum[k] for k in sorted(bhum)]
    out = ROOT / "newmodels" / "rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(len(rows), Counter((r["source"], r["model"]) for r in rows))


if __name__ == "__main__":
    main()
