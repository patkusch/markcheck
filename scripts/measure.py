"""Measure how well markcheck does on the test files, and which labels survive ordinary sharing.

    .venv/bin/python scripts/measure.py              # print the tables
    .venv/bin/python scripts/measure.py --json out.json

It reads every tests/fixtures/*/manifest.json:

    {"files": [{"file": "x.png", "expect": ["sd_watermark"], "note": "..."}]}

`expect` lists the checks that should say "found"; an empty list marks a
control file where nothing should be found.

For each check it reports:
- hits: labelled files where the expected check said "found" / labelled files
- false alarms: control files where this check said "found" / control files

Then, for every labelled picture, it makes three everyday copies in a temp
folder and checks whether each expected label is still found:
- JPEG q75: saved as JPEG at quality 75, like many upload sites do
- 50% resize: half the width and height, saved in the same format
  (EXIF, XMP and ICC data carried over where Pillow can)
- screenshot-like: decoded to plain pixels and saved as a fresh PNG with no metadata

Failures are expected. The point is to report them honestly.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, PngImagePlugin  # noqa: E402

from markcheck.cli import check_file  # noqa: E402
from markcheck.filetypes import sniff  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
VARIANTS = ["original", "JPEG q75", "50% resize", "screenshot-like"]


def load_manifests(root: Path = FIXTURES) -> list[tuple[Path, dict]]:
    items = []
    for man in sorted(root.glob("*/manifest.json")):
        try:
            doc = json.loads(man.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: cannot read {man}: {exc}", file=sys.stderr)
            continue
        for entry in doc.get("files", []):
            p = man.parent / entry["file"]
            if p.is_file():
                items.append((p, entry))
            else:
                print(f"warning: {p} is listed but missing", file=sys.stderr)
    return items


def found_detectors(path: Path, options: dict | None = None) -> set[str]:
    rep = check_file(path, options)
    return {f.detector for f in rep.findings if f.found}


def jpeg_q75(src: Path, out_dir: Path) -> Path:
    with Image.open(src) as im:
        dst = out_dir / (src.stem + "_q75.jpg")
        im.convert("RGB").save(dst, "JPEG", quality=75)
    return dst


def half_size(src: Path, out_dir: Path) -> Path:
    with Image.open(src) as im:
        fmt = im.format or "PNG"
        small = im.resize((max(1, im.width // 2), max(1, im.height // 2)), Image.Resampling.LANCZOS)
        kw = {}
        if im.info.get("icc_profile"):
            kw["icc_profile"] = im.info["icc_profile"]
        exif = im.getexif()
        if exif:
            kw["exif"] = exif.tobytes()
        xmp = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp")
        if fmt == "PNG":
            info = PngImagePlugin.PngInfo()
            for k, v in im.info.items():
                if isinstance(v, str):
                    info.add_itxt(k, v)
            kw["pnginfo"] = info
        elif xmp and fmt in ("JPEG", "WEBP", "TIFF"):
            kw["xmp"] = xmp if isinstance(xmp, bytes) else str(xmp).encode()
        if fmt == "JPEG":
            kw["quality"] = 95
            if small.mode not in ("RGB", "L", "CMYK"):
                small = small.convert("RGB")
        dst = out_dir / (src.stem + "_half" + src.suffix)
        small.save(dst, fmt, **kw)
    return dst


def screenshot_like(src: Path, out_dir: Path) -> Path:
    with Image.open(src) as im:
        im.seek(0)
        pixels = Image.frombytes("RGB", im.size, im.convert("RGB").tobytes())  # pixels only, nothing else
    dst = out_dir / (src.stem + "_screen.png")
    pixels.save(dst, "PNG")
    return dst


MAKERS = {"JPEG q75": jpeg_q75, "50% resize": half_size, "screenshot-like": screenshot_like}


def measure(root: Path = FIXTURES) -> dict:
    items = load_manifests(root)
    hits = defaultdict(lambda: [0, 0])           # detector -> [found, labelled]
    alarms = defaultdict(int)                    # detector -> found on controls
    n_controls = 0
    survival = defaultdict(lambda: {v: [0, 0] for v in VARIANTS})
    per_file = []
    all_detectors: set[str] = set()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for path, entry in items:
            expect = entry.get("expect", [])
            found = found_detectors(path)
            all_detectors |= set(expect) | found
            row = {"file": str(path.relative_to(root)), "expect": expect, "found": sorted(found)}
            if not expect:
                n_controls += 1
                for d in found:
                    alarms[d] += 1
            for d in expect:
                hits[d][1] += 1
                hits[d][0] += d in found
            is_image = sniff(path, path.read_bytes()[:64]) == "image"
            if expect and is_image:
                row["survival"] = {}
                for d in expect:
                    survival[d]["original"][1] += 1
                    survival[d]["original"][0] += d in found
                for vname, make in MAKERS.items():
                    try:
                        copy = make(path, tmpdir)
                        vfound = found_detectors(copy)
                    except Exception as exc:  # a copy we could not make counts as "not found"
                        print(f"warning: {vname} copy of {path.name} failed: {exc}", file=sys.stderr)
                        vfound = set()
                    row["survival"][vname] = sorted(vfound)
                    for d in expect:
                        survival[d][vname][1] += 1
                        survival[d][vname][0] += d in vfound
            per_file.append(row)

    return {
        "files": len(items),
        "controls": n_controls,
        "detectors": {
            d: {"hits": hits[d][0], "labelled": hits[d][1], "false_alarms": alarms.get(d, 0),
                "controls": n_controls}
            for d in sorted(all_detectors)
        },
        "survival": {d: {v: {"found": s[v][0], "of": s[v][1]} for v in VARIANTS} for d, s in sorted(survival.items())},
        "per_file": per_file,
    }


def tables(res: dict) -> str:
    out = [f"Files measured: {res['files']} ({res['controls']} controls with no label)", "",
           "| Check | Found on labelled files | False alarms on controls |",
           "|---|---|---|"]
    for d, r in res["detectors"].items():
        hit = f"{r['hits']}/{r['labelled']}" if r["labelled"] else "no labelled files"
        out.append(f"| {d} | {hit} | {r['false_alarms']}/{r['controls']} |")
    out += ["", "Does the label survive everyday handling? (labelled pictures still found / total)", "",
            "| Label | " + " | ".join(VARIANTS) + " |",
            "|---|" + "---|" * len(VARIANTS)]
    for d, s in res["survival"].items():
        out.append(f"| {d} | " + " | ".join(f"{s[v]['found']}/{s[v]['of']}" for v in VARIANTS) + " |")
    if not res["survival"]:
        out.append("| (no labelled pictures) |" + " |" * len(VARIANTS))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", metavar="PATH", help="also write the full results as JSON to PATH")
    ap.add_argument("--fixtures", default=str(FIXTURES), help="folder holding */manifest.json")
    args = ap.parse_args(argv)
    res = measure(Path(args.fixtures))
    print(tables(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
