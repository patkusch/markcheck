"""Fetch a small, balanced sample of RAID for the text check (nothing is committed).

    python -I scripts/text_fetch_raid.py          # download pieces, then pick the sample
    python -I scripts/text_fetch_raid.py --pick   # only pick again from pieces already downloaded

RAID (https://huggingface.co/datasets/liamdugan/raid, MIT licence) has 5.6 million rows
in one 11.8 GB CSV file, sorted by kind of writing. We do not download all of it.
Instead we read small pieces of it from the server (about 150 MB in all), keep the rows
that are whole, and pick a sample from them. Only the train
file is used: the test file has its answers hidden.

Writes ~/.cache/markcheck/textdata/raid/probe.jsonl (the map), pool.jsonl (every usable row seen) and
~/.cache/markcheck/textdata/raid/rows.jsonl (the sample text_data.py reads).
The sample is fixed by a seed, so picking again gives the same rows.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path.home() / ".cache" / "markcheck" / "textdata" / "raid"
URL = "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv"
SIZE = 11_779_491_051
STEP = 2_000_000     # probe every 2 MB
PROBE = 24_000       # bytes read per probe
PIECE = 250_000      # bytes read per piece (about 100 rows)
PIECES_NONE = 4      # pieces per (kind, model) block of untouched texts
PIECES_PARA = 2      # pieces per (kind, model) block of reworded texts
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
START = re.compile(rf"\n(?={UUID},{UUID},{UUID},)")
COLS = ["id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty",
        "attack", "domain", "title", "prompt", "generation"]
KINDS = ["abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"]
PER_CELL = 55        # untouched texts per (kind, model): 8 kinds x 12 models x 55 = about 5,300
PARA_PER_KIND = 190  # reworded texts per kind: about 1,500


def _get(start: int, n: int) -> list[dict]:
    """Read n bytes from the file at start; return the whole rows found in them."""
    import requests

    import time

    for attempt in range(8):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{start + n - 1}"}, timeout=120)
            r.raise_for_status()
            break
        except Exception:  # noqa: BLE001 - retry network hiccups and "too many requests"
            if attempt == 7:
                raise
            time.sleep(15 * 2 ** attempt)
    text = r.content.decode("utf-8", "replace")
    m = START.search(text)
    if not m:
        return []
    out = []
    for rec in list(csv.reader(io.StringIO(text[m.end():])))[:-1]:  # the last row may be cut off
        if len(rec) == len(COLS) and re.fullmatch(UUID, rec[0]):
            out.append(dict(zip(COLS, rec)))
    return out


def fetch() -> None:
    """Two steps. First read 24 KB every 2 MB to map where each (kind, model, attack) block
    sits in the file. RAID stores rows in blocks of about 1,000 by model and attack.
    Then read 250 KB pieces only inside the blocks we want, spread across each block."""
    from concurrent.futures import ThreadPoolExecutor

    DATA.mkdir(parents=True, exist_ok=True)
    probe_f = DATA / "probe.jsonl"
    done = set()
    if probe_f.exists():
        done = {json.loads(line)["off"] for line in open(probe_f, encoding="utf-8")}
    offs = [o for o in range(0, SIZE - PROBE, STEP) if o not in done]

    def probe(o: int) -> dict:
        rows = _get(o, PROBE)
        return {"off": o, "cells": sorted({(r["domain"], r["model"], r["attack"]) for r in rows})}

    with open(probe_f, "a", encoding="utf-8") as fh, ThreadPoolExecutor(4) as ex:
        for i, res in enumerate(ex.map(probe, offs)):
            fh.write(json.dumps(res) + "\n")
            if i % 500 == 0:
                print("probe", i, len(offs), flush=True)
    where = defaultdict(list)
    for line in open(probe_f, encoding="utf-8"):
        d = json.loads(line)
        for dom, model, attack in d["cells"]:
            if attack in ("none", "paraphrase") and dom in KINDS:
                where[(dom, model, attack)].append(d["off"])
    want = []
    for key in sorted(where):
        offs = sorted(where[key])
        k = PIECES_NONE if key[2] == "none" else PIECES_PARA
        idx = sorted({int(i * (len(offs) - 1) / max(k - 1, 1)) for i in range(k)})
        want += [offs[i] for i in idx]
    pool_f = DATA / "pool.jsonl"
    got = set()
    if pool_f.exists():
        got = {json.loads(line)["piece"] for line in open(pool_f, encoding="utf-8")}
    want = [o for o in sorted(set(want)) if o not in got]
    print("pieces to read", len(want), flush=True)
    with open(pool_f, "a", encoding="utf-8") as fh, ThreadPoolExecutor(4) as ex:
        for i, (o, rows) in enumerate(zip(want, ex.map(lambda o: _get(o, PIECE), want))):
            kept = [r for r in rows if r["attack"] in ("none", "paraphrase") and r["domain"] in KINDS]
            for r in kept:
                r["piece"] = o
                fh.write(json.dumps(r) + "\n")
            if not kept:
                fh.write(json.dumps({"piece": o, "attack": "skip"}) + "\n")
            if i % 50 == 0:
                print("piece", i, len(want), flush=True)


def pick() -> None:
    rows = [json.loads(line) for line in open(DATA / "pool.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["attack"] in ("none", "paraphrase")]
    uniq = {}
    for r in rows:
        uniq.setdefault(r["id"], r)
    rows = sorted(uniq.values(), key=lambda r: r["id"])
    # Repetition penalty is an unusual setting that RAID added to make detection harder.
    # We keep only texts made without it; greedy and random-sampling decoding stay mixed.
    rows = [r for r in rows if r["repetition_penalty"] in ("", "no")]
    rng = random.Random(4)
    cells = defaultdict(list)
    for r in rows:
        if r["attack"] == "none":
            cells[(r["domain"], r["model"])].append(r)
    chosen = []
    for key in sorted(cells):
        c = cells[key]
        rng.shuffle(c)
        chosen += c[:PER_CELL]
    para = defaultdict(list)
    for r in rows:
        if r["attack"] == "paraphrase":
            para[r["domain"]].append(r)
    for kind in sorted(para):
        c = para[kind]
        rng.shuffle(c)
        # people's texts too, if RAID reworded them, at about 1 in 6
        hum = [r for r in c if r["model"] == "human"][: PARA_PER_KIND // 6]
        ai = [r for r in c if r["model"] != "human"][: PARA_PER_KIND - len(hum)]
        chosen += hum + ai
    chosen.sort(key=lambda r: (r["attack"] != "none", r["domain"], r["model"], r["id"]))
    with open(DATA / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in chosen:
            r.pop("piece", None)
            fh.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(chosen))
    print(Counter((r["attack"], r["domain"]) for r in chosen))
    print(Counter((r["attack"], r["model"]) for r in chosen))
    print(Counter((r["attack"], r["decoding"]) for r in chosen))


if __name__ == "__main__":
    if "--pick" not in sys.argv:
        fetch()
    pick()
