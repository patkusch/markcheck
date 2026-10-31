"""Measure how well markcheck does on the test files, and which labels survive ordinary sharing.

    .venv/bin/python scripts/measure.py              # print the tables
    .venv/bin/python scripts/measure.py --json out.json

It reads every tests/fixtures/**/manifest.json:

    {"files": [{"file": "x.png", "expect": ["sd_watermark"], "note": "..."}]}

`expect` lists the checks that should say "found"; an empty list marks a
control file where nothing should be found.

For each check it reports:
- hits: labelled files where the expected check said "found" / labelled files
- false alarms: control files where this check said "found" / control files

Then, for every labelled picture, it makes everyday copies in a temp folder
and checks whether each expected label is still found. Two things happen
when people share a picture, and they are measured apart:

1. The pixels change (re-compressed, resized, screenshotted).
2. The file's metadata (EXIF, XMP, ICC colour profile, PNG text, the C2PA
   signed label) is either carried along into the new file, as editors that
   keep metadata do, or dropped, as most upload sites and plain "save as"
   code do.

Copies (see NOTES for the wording printed under the table):
- JPEG q75, kept / dropped: saved as JPEG at quality 75.
- 50% resize, kept / dropped: half the width and height, same format.
- screenshot-like: only the pixels, as a fresh PNG.

"Kept" copies the existing metadata bytes unchanged into the new file. This
is ordinary handling: it neither adds, removes nor alters any label. For
C2PA, a label carried along onto changed pixels no longer matches the file;
the checker then reports "changed after signing". Those are counted apart
("say changed"), not as a plain survive.

Failures are expected. The point is to report them honestly.
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import tempfile
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from markcheck.cli import check_file  # noqa: E402
from markcheck.filetypes import sniff  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
VARIANTS = ["original", "JPEG q75, metadata kept", "JPEG q75, metadata dropped",
            "50% resize, metadata kept", "50% resize, metadata dropped", "screenshot-like"]
JPEG_Q = 75
RESIZE_JPEG_Q = 90

NOTES = [
    "How each copy was made (Pillow, in a temp folder; the test files are never changed):",
    "- **original**: the file as made. Four C2PA files are meant to say changed: our "
    "`changed_after_signing.jpg` and the three broken C2PA public test files (`E-dat`, `E-sig`, `E-uri`).",
    f"- **JPEG q{JPEG_Q}**: decoded and saved as a JPEG at quality {JPEG_Q}, like many upload sites do.",
    f"- **50% resize**: half the width and height (Lanczos), saved in the same format "
    f"(JPEG at quality {RESIZE_JPEG_Q}, PNG and WebP as PNG and WebP).",
    "- **metadata kept**: the original EXIF, XMP, ICC profile, PNG text and C2PA label bytes are copied "
    "unchanged into the new file, as an editor that keeps metadata does. Same-format JPEG copies keep "
    "every APP segment byte for byte. GIF and TIFF files are left out of these columns (we do not "
    "carry their signed labels).",
    "- **metadata dropped**: only the pixels are saved, as most upload sites and plain re-save code do.",
    "- **screenshot-like**: only the pixels, as a fresh PNG with no metadata.",
    "- **say changed** (C2PA only): the label is still in the file but reports that the file or label "
    "changed after signing, or that its signature does not check out. It is not counted as a survive.",
    "- PNG text tags (the AIGC text chunk, Stable Diffusion `parameters`) have no place in a JPEG, so a "
    "PNG-to-JPEG copy loses them even when metadata is kept.",
    "- Sound, video and PDF files are counted in the first table but not here.",
]


# ---------------------------------------------------------------- manifests

def load_manifests(root: Path = FIXTURES) -> list[tuple[Path, dict]]:
    items = []
    for man in sorted(root.glob("**/manifest.json")):
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


def findings(path: Path, options: dict | None = None) -> dict:
    """detector -> Finding, for the checks that said found."""
    rep = check_file(path, options)
    return {f.detector: f for f in rep.findings if f.found}


def found_detectors(path: Path, options: dict | None = None) -> set[str]:
    return set(findings(path, options))


def says_changed(f) -> bool:
    """A C2PA label that is in the file but no longer vouches for it: the file or the
    label changed after signing, or the signature does not check out."""
    return f.detector == "c2pa" and str(f.details.get("changed_since_signing", "no")) != "no"


# ---------------------------------------------------------------- metadata carried along

@dataclass
class Meta:
    exif: bytes | None = None          # TIFF-structured EXIF, without the "Exif\0\0" header
    xmp: bytes | None = None
    icc: bytes | None = None
    png_text: list[tuple[bytes, bytes]] = field(default_factory=list)  # (chunk type, chunk data)
    c2pa: bytes | None = None          # the JUMBF manifest store
    jpeg_app: list[bytes] = field(default_factory=list)  # whole APPn segments (not APP0) of a JPEG
    carried: bool = True               # False: we cannot carry this file's label


XMP_HEAD = b"http://ns.adobe.com/xap/1.0/\x00"


def _jpeg_segments(data: bytes):
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:
            return
        n = struct.unpack(">H", data[i + 2:i + 4])[0]
        yield marker, data[i:i + 2 + n]
        i += 2 + n


def _jumbf_from_app11(segs: list[bytes]) -> bytes | None:
    """Join JPEG APP11 'JP' segments back into one JUMBF box (ISO 19566-5)."""
    parts = []
    for seg in segs:
        body = seg[4:]
        if body[:2] != b"JP":
            continue
        z = struct.unpack(">I", body[4:8])[0]
        box = body[8:]
        if z > 1:  # continuation: box header (LBox, TBox[, XLBox]) is repeated
            box = box[16:] if struct.unpack(">I", box[:4])[0] == 1 else box[8:]
        parts.append((z, box))
    return b"".join(b for _, b in sorted(parts)) or None


def _app11_from_jumbf(box: bytes, en: int = 0x0211) -> list[bytes]:
    head = box[:8]
    first_max = 65533 - 8
    later_max = 65533 - 8 - len(head)
    chunks = [box[:first_max]]
    rest = box[first_max:]
    while rest:
        chunks.append(head + rest[:later_max])
        rest = rest[later_max:]
    out = []
    for z, c in enumerate(chunks, 1):
        body = b"JP" + struct.pack(">HI", en, z) + c
        out.append(b"\xff\xeb" + struct.pack(">H", len(body) + 2) + body)
    return out


def _png_chunks(data: bytes):
    i = 8
    while i + 8 <= len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        yield data[i + 4:i + 8], data[i + 8:i + 8 + n]
        i += 12 + n


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)


def _riff_chunks(data: bytes):
    i = 12
    while i + 8 <= len(data):
        n = struct.unpack("<I", data[i + 4:i + 8])[0]
        yield data[i:i + 4], data[i + 8:i + 8 + n]
        i += 8 + n + (n & 1)


def _itxt_text(body: bytes) -> bytes | None:
    """The text of a PNG iTXt chunk: keyword, flag, method, language, translated keyword, text."""
    try:
        _, rest = body.split(b"\x00", 1)
        compressed, rest = rest[0], rest[2:]
        _, rest = rest.split(b"\x00", 1)
        _, text = rest.split(b"\x00", 1)
        return zlib.decompress(text) if compressed else text
    except (ValueError, IndexError, zlib.error):
        return None


def read_meta(path: Path) -> Meta:
    data = path.read_bytes()
    m = Meta()
    with Image.open(path) as im:
        m.icc = im.info.get("icc_profile")
        fmt = im.format
    if fmt == "JPEG":
        app11 = []
        for marker, seg in _jpeg_segments(data):
            if 0xE1 <= marker <= 0xEF:
                m.jpeg_app.append(seg)
            body = seg[4:]
            if marker == 0xE1 and body.startswith(b"Exif\x00\x00"):
                m.exif = body[6:]
            elif marker == 0xE1 and body.startswith(XMP_HEAD):
                m.xmp = body[len(XMP_HEAD):]
            elif marker == 0xEB:
                app11.append(seg)
        m.c2pa = _jumbf_from_app11(app11)
    elif fmt == "PNG":
        for ctype, body in _png_chunks(data):
            if ctype in (b"tEXt", b"iTXt", b"zTXt"):
                m.png_text.append((ctype, body))
                if ctype == b"iTXt" and body.startswith(b"XML:com.adobe.xmp\x00"):
                    m.xmp = _itxt_text(body)
            elif ctype == b"eXIf":
                m.exif = body
            elif ctype == b"caBX":
                m.c2pa = body
    elif fmt == "WEBP":
        for cid, body in _riff_chunks(data):
            if cid == b"EXIF":
                m.exif = body[6:] if body.startswith(b"Exif\x00\x00") else body
            elif cid == b"XMP ":
                m.xmp = body
            elif cid == b"C2PA":
                m.c2pa = body
    else:  # GIF, TIFF: we do not carry their signed label across
        m.carried = False
    return m


def _insert_jpeg_segments(jpeg: bytes, segs: list[bytes]) -> bytes:
    """Put APP segments right after SOI and any APP0 (JFIF) segment."""
    i = 2
    if jpeg[2:4] == b"\xff\xe0":
        i = 4 + struct.unpack(">H", jpeg[4:6])[0]
    return jpeg[:i] + b"".join(segs) + jpeg[i:]


def save_with(im: Image.Image, fmt: str, meta: Meta | None, dst: Path, quality: int | None = None) -> Path:
    """Save pixels as `fmt`. With meta, copy the original metadata bytes in unchanged."""
    buf = io.BytesIO()
    if fmt == "JPEG":
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(buf, "JPEG", quality=quality or JPEG_Q)
        out = buf.getvalue()
        if meta:
            if meta.jpeg_app:  # JPEG to JPEG: every APP segment, byte for byte
                segs = list(meta.jpeg_app)
            else:
                segs = []
                if meta.exif:
                    segs.append(b"\xff\xe1" + struct.pack(">H", len(meta.exif) + 8) + b"Exif\x00\x00" + meta.exif)
                if meta.xmp:
                    segs.append(b"\xff\xe1" + struct.pack(">H", len(meta.xmp) + len(XMP_HEAD) + 2) + XMP_HEAD + meta.xmp)
                if meta.c2pa:
                    segs += _app11_from_jumbf(meta.c2pa)
            if meta.icc and not meta.jpeg_app:
                tmp = io.BytesIO()
                im.save(tmp, "JPEG", quality=quality or JPEG_Q, icc_profile=meta.icc)
                out = tmp.getvalue()
            out = _insert_jpeg_segments(out, segs)
    elif fmt == "PNG":
        im.save(buf, "PNG", icc_profile=meta.icc if meta and meta.icc else None)
        out = buf.getvalue()
        if meta:
            extra = [_png_chunk(t, b) for t, b in meta.png_text]
            if meta.xmp and not any(b.startswith(b"XML:com.adobe.xmp\x00") for _, b in meta.png_text):
                extra.append(_png_chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + meta.xmp))
            if meta.exif:
                extra.append(_png_chunk(b"eXIf", meta.exif))
            if meta.c2pa:
                extra.append(_png_chunk(b"caBX", meta.c2pa))
            idat = out.find(b"IDAT") - 4
            out = out[:idat] + b"".join(extra) + out[idat:]
    elif fmt == "WEBP":
        kw = {}
        if meta:
            kw = {k: v for k, v in (("exif", meta.exif and b"Exif\x00\x00" + meta.exif),
                                    ("xmp", meta.xmp), ("icc_profile", meta.icc)) if v}
        im.save(buf, "WEBP", quality=quality or 80, **kw)
        out = buf.getvalue()
        if meta and meta.c2pa:
            body = meta.c2pa
            out += b"C2PA" + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")
            out = out[:4] + struct.pack("<I", len(out) - 8) + out[8:]
    else:
        im.save(buf, fmt)
        out = buf.getvalue()
    dst.write_bytes(out)
    return dst


# ---------------------------------------------------------------- the everyday copies

def _pixels(src: Path) -> tuple[Image.Image, str]:
    with Image.open(src) as im:
        im.seek(0)
        fmt = im.format or "PNG"
        im.load()
        px = im.copy()
    if px.mode not in ("RGB", "RGBA", "L"):
        px = px.convert("RGBA" if "A" in px.mode or "transparency" in px.info else "RGB")
    return px, fmt


def jpeg_q75(src: Path, out_dir: Path, keep: bool) -> Path | None:
    meta = read_meta(src) if keep else None
    if meta and not meta.carried:
        return None
    px, _ = _pixels(src)
    return save_with(px.convert("RGB"), "JPEG", meta, out_dir / f"{src.stem}_q75_{'kept' if keep else 'dropped'}.jpg")


def half_size(src: Path, out_dir: Path, keep: bool) -> Path | None:
    meta = read_meta(src) if keep else None
    if meta and not meta.carried:
        return None
    px, fmt = _pixels(src)
    if fmt not in ("JPEG", "PNG", "WEBP"):
        fmt = "PNG"
    small = px.resize((max(1, px.width // 2), max(1, px.height // 2)), Image.Resampling.LANCZOS)
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]
    return save_with(small, fmt, meta, out_dir / f"{src.stem}_half_{'kept' if keep else 'dropped'}{ext}",
                     quality=RESIZE_JPEG_Q if fmt == "JPEG" else None)


def screenshot_like(src: Path, out_dir: Path) -> Path:
    px, _ = _pixels(src)
    rgb = Image.frombytes("RGB", px.size, px.convert("RGB").tobytes())  # pixels only, nothing else
    dst = out_dir / (src.stem + "_screen.png")
    rgb.save(dst, "PNG")
    return dst


MAKERS = {
    "JPEG q75, metadata kept": lambda s, d: jpeg_q75(s, d, True),
    "JPEG q75, metadata dropped": lambda s, d: jpeg_q75(s, d, False),
    "50% resize, metadata kept": lambda s, d: half_size(s, d, True),
    "50% resize, metadata dropped": lambda s, d: half_size(s, d, False),
    "screenshot-like": screenshot_like,
}


# ---------------------------------------------------------------- measuring

def _cell() -> dict:
    return {"found": 0, "says_changed": 0, "of": 0}


def _count(cell: dict, d: str, found: dict) -> None:
    cell["of"] += 1
    f = found.get(d)
    if f is None:
        return
    if says_changed(f):
        cell["says_changed"] += 1
    else:
        cell["found"] += 1


def measure(root: Path = FIXTURES) -> dict:
    items = load_manifests(root)
    hits = defaultdict(lambda: [0, 0])           # detector -> [found, labelled]
    alarms = defaultdict(int)                    # detector -> found on controls
    alarm_files = []
    n_controls = 0
    survival = defaultdict(lambda: {v: _cell() for v in VARIANTS})
    per_file = []
    all_detectors: set[str] = set()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for path, entry in items:
            expect = entry.get("expect", [])
            found = findings(path)
            all_detectors |= set(expect) | set(found)
            rel = str(path.relative_to(root))
            row = {"file": rel, "expect": expect, "found": sorted(found)}
            if not expect:
                n_controls += 1
                for d in found:
                    alarms[d] += 1
                    alarm_files.append({"file": rel, "detector": d})
            for d in expect:
                hits[d][1] += 1
                hits[d][0] += d in found
            is_image = sniff(path, path.read_bytes()[:64]) == "image"
            if expect and is_image:
                row["survival"] = {}
                for d in expect:
                    _count(survival[d]["original"], d, found)
                for vname, make in MAKERS.items():
                    try:
                        copy = make(path, tmpdir)
                        if copy is None:  # this file type is left out of this column
                            row["survival"][vname] = None
                            continue
                        vfound = findings(copy)
                    except Exception as exc:  # a copy we could not make counts as "not found"
                        print(f"warning: {vname} copy of {path.name} failed: {exc}", file=sys.stderr)
                        vfound = {}
                    row["survival"][vname] = sorted(
                        d + (" (says changed)" if says_changed(f) else "") for d, f in vfound.items())
                    for d in expect:
                        _count(survival[d][vname], d, vfound)
            per_file.append(row)

    return {
        "files": len(items),
        "controls": n_controls,
        "detectors": {
            d: {"hits": hits[d][0], "labelled": hits[d][1], "false_alarms": alarms.get(d, 0),
                "controls": n_controls}
            for d in sorted(all_detectors)
        },
        "false_alarm_files": alarm_files,
        "survival": {d: s for d, s in sorted(survival.items())},
        "per_file": per_file,
    }


def _fmt(c: dict) -> str:
    if not c["of"]:
        return "-"
    s = f"{c['found']}/{c['of']}"
    if c["says_changed"]:
        s += f" ({c['says_changed']} say changed)"
    return s


def tables(res: dict) -> str:
    out = [f"Files measured: {res['files']} ({res['controls']} controls with no label)", "",
           "| Check | Found on labelled files | False alarms on controls |",
           "|---|---|---|"]
    for d, r in res["detectors"].items():
        hit = f"{r['hits']}/{r['labelled']}" if r["labelled"] else "no labelled files"
        out.append(f"| {d} | {hit} | {r['false_alarms']}/{r['controls']} |")
    out += ["", "Does the label survive everyday handling? (labelled pictures where the label is still found "
                "and intact / pictures measured)", "",
            "| Label | " + " | ".join(VARIANTS) + " |",
            "|---|" + "---|" * len(VARIANTS)]
    for d, s in res["survival"].items():
        out.append(f"| {d} | " + " | ".join(_fmt(s[v]) for v in VARIANTS) + " |")
    if not res["survival"]:
        out.append("| (no labelled pictures) |" + " |" * len(VARIANTS))
    out += [""] + NOTES
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
