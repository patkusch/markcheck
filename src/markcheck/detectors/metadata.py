"""Metadata tags: plain-text notes inside a file that say how it was made.

What this check reads (detect and explain only; it never changes a file):

1. IPTC "digital source type" in the file's XMP text. Values and meanings
   come from the IPTC NewsCodes vocabulary at
   https://cv.iptc.org/newscodes/digitalsourcetype/ (read 2026-10-31).
2. China's "AIGC" label (standard GB 45438-2025). We could not read the
   official standard or the TC260 implementation guide. The layout used
   here follows secondary sources only (read 2026-10-31):
     - Alibaba Cloud and Tencent Cloud docs show a JSON object named AIGC
       with keys Label, ContentProducer, ProduceID, ReservedCode1,
       ContentPropagator, PropagateID, ReservedCode2, written as
       AIGC={...} or AIGC: {...};
     - BytePlus docs give Label 1 = AI-generated, 2 = possibly AI-generated,
       3 = suspected AI-generated (2 and 3 are set by sharing platforms);
     - a search summary of the TC260 image guide names the XMP namespace
       http://www.tc260.org.cn/ns/AIGC/1.0/ (unconfirmed).
   So we accept: an XMP property named AIGC (any namespace) holding JSON
   or the fields as sub-properties; the fields as properties in a
   tc260 namespace; a PNG text chunk or EXIF text named AIGC; and an
   "AIGC" key followed by a JSON object anywhere in the raw bytes.
3. Weaker hints: generation settings that Stable Diffusion tools write into
   PNG text (AUTOMATIC1111 "parameters", ComfyUI "prompt"/"workflow"), and a
   Software / CreatorTool field naming an AI image tool. These are plain
   text anyone can type, so they show what the file says, not who made it.
"""
from __future__ import annotations

import json
import re
import struct
import zlib
import xml.etree.ElementTree as ET

from ..findings import Finding

NAME = "metadata"
LABEL = "Metadata tags"

MAX_XMP_BYTES = 4 * 1024 * 1024      # ignore XMP packets bigger than this
MAX_PACKETS = 16
MAX_TEXT_CHUNK = 1024 * 1024
MAX_ZTXT_OUT = 2 * 1024 * 1024

XMP_JPEG_SIG = b"http://ns.adobe.com/xap/1.0/\x00"
XMP_EXT_SIG = b"http://ns.adobe.com/xmp/extension/\x00"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# ---------------------------------------------------------------- IPTC vocabulary
# code -> (group, IPTC name). Groups:
#   "ai"       made by generative AI                      -> flagged
#   "ai_part"  partly made or edited with generative AI   -> flagged
#   "program"  made by a computer program, not trained AI -> flagged (synthetic)
#   "maybe"    may or may not include generative AI       -> not flagged, explained
#   "other"    no AI involvement claimed                  -> not flagged
IPTC_TYPES = {
    "trainedAlgorithmicMedia": ("ai", "Created using Generative AI"),
    "compositeWithTrainedAlgorithmicMedia": ("ai_part", "Edited using Generative AI"),
    "compositeSynthetic": ("ai_part", "Composite including generative AI elements"),
    "algorithmicMedia": ("program", "Pure algorithmic media (made by a program, not trained AI)"),
    "composite": ("maybe", "Composite of elements, any of which may or may not be generative AI"),
    "virtualRecording": ("maybe", "Virtual event recording, may include generative AI"),
    "digitalCapture": ("other", "Digital capture sampled from real life"),
    "computationalCapture": ("other", "Multi-frame computational capture sampled from real life"),
    "negativeFilm": ("other", "Digitised from a transparent negative"),
    "positiveFilm": ("other", "Digitised from a transparent positive"),
    "print": ("other", "Digitised from a non-transparent medium"),
    "minorHumanEdits": ("other", "Original media with minor human edits (retired code)"),
    "humanEdits": ("other", "Human-edited media"),
    "algorithmicallyEnhanced": ("other", "Algorithmically-altered media"),
    "softwareImage": ("other", "Created by software (retired code)"),
    "digitalArt": ("other", "Digital art (retired code)"),
    "digitalCreation": ("other", "Digital creation"),
    "dataDrivenMedia": ("other", "Data-driven media"),
    "screenCapture": ("other", "Screen capture"),
    "compositeCapture": ("other", "Composite of captured elements"),
}
_IPTC_LOWER = {k.lower(): k for k in IPTC_TYPES}
_IPTC_URI = re.compile(r"^https?://cv\.iptc\.org/newscodes/digitalsourcetype/([A-Za-z]+)/?$", re.I)

GROUP_SUMMARY = {
    "ai": "The file's metadata says it was made by AI (IPTC: {code}).",
    "ai_part": "The file's metadata says it was partly made or edited with AI (IPTC: {code}).",
    "program": "The file's metadata says it was made by a computer program, not by trained AI (IPTC: {code}).",
}

# ---------------------------------------------------------------- AIGC
AIGC_FIELDS = ("Label", "ContentProducer", "ProduceID", "ReservedCode1",
               "ContentPropagator", "PropagateID", "ReservedCode2")
AIGC_LABELS = {
    "1": "the content was made by AI",
    "2": "the content may have been made by AI (a label sharing platforms use)",
    "3": "the content is suspected of being made by AI (a label sharing platforms use)",
}
_AIGC_RAW = re.compile(rb"AIGC[\"']?\s*[=:]\s*(\{[^{}]{1,8000}\})")

# ---------------------------------------------------------------- AI tool names
AI_TOOLS = [
    ("midjourney", "Midjourney"), ("dall-e", "DALL-E"), ("dall·e", "DALL-E"), ("dalle", "DALL-E"),
    ("stable diffusion", "Stable Diffusion"), ("stablediffusion", "Stable Diffusion"),
    ("automatic1111", "AUTOMATIC1111"), ("comfyui", "ComfyUI"), ("invokeai", "InvokeAI"),
    ("novelai", "NovelAI"), ("adobe firefly", "Adobe Firefly"), ("imagen", "Google Imagen"),
    ("ideogram", "Ideogram"), ("leonardo.ai", "Leonardo.Ai"), ("chatgpt", "ChatGPT"),
    ("gpt-image", "OpenAI GPT Image"), ("bing image creator", "Bing Image Creator"),
    ("image creator from designer", "Microsoft Designer"), ("nova canvas", "Amazon Nova Canvas"),
    ("titan image generator", "Amazon Titan Image Generator"), ("firefly", "Adobe Firefly"),
]
_TOOL_RE = [(re.compile(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", re.I), v) for k, v in AI_TOOLS]

PLAIN_TEXT = ("Metadata is plain text that anyone can add, change or remove; it is not signed, "
              "so it does not show who made the file or whether this note is correct.")
LOST = "If a file has no such tag, that proves nothing: tags are often removed when files are shared or re-saved."


# ================================================================ byte readers

def _jpeg_segments(data: bytes):
    """Yield (marker, payload) for JPEG header segments up to the image data."""
    if not data.startswith(b"\xff\xd8"):
        return
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # end of image / start of scan
            return
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if length < 2:
            return
        yield marker, data[i + 4:i + 2 + length]
        i += 2 + length


def _png_chunks(data: bytes):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return
    i, n = 8, len(data)
    while i + 8 <= n:
        length, ctype = struct.unpack(">I4s", data[i:i + 8])
        body = data[i + 8:i + 8 + length]
        yield ctype, body
        if ctype == b"IEND":
            return
        i += 12 + length


def _riff_chunks(data: bytes):
    if not (data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        return
    i, n = 12, len(data)
    while i + 8 <= n:
        ctype, length = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
        yield ctype, data[i + 8:i + 8 + length]
        i += 8 + length + (length & 1)


def _inflate(blob: bytes) -> bytes:
    d = zlib.decompressobj()
    out = d.decompress(blob, MAX_ZTXT_OUT)
    return out


def _png_text(ctype: bytes, body: bytes):
    """Return (keyword, text) for tEXt/zTXt/iTXt, or None."""
    try:
        if ctype == b"tEXt":
            k, _, v = body.partition(b"\x00")
            return k.decode("latin-1"), v[:MAX_TEXT_CHUNK].decode("latin-1")
        if ctype == b"zTXt":
            k, _, rest = body.partition(b"\x00")
            return k.decode("latin-1"), _inflate(rest[1:]).decode("latin-1")
        if ctype == b"iTXt":
            k, _, rest = body.partition(b"\x00")
            comp_flag = rest[0:1]
            rest = rest[2:]
            _lang, _, rest = rest.partition(b"\x00")
            _tkey, _, text = rest.partition(b"\x00")
            if comp_flag == b"\x01":
                text = _inflate(text)
            return k.decode("latin-1"), text[:MAX_TEXT_CHUNK].decode("utf-8", "replace")
    except (zlib.error, IndexError):
        return None
    return None


# TIFF / EXIF: only the few text tags we need.
_TIFF_TAGS = {0x010E: "ImageDescription", 0x0131: "Software", 0x013B: "Artist",
              0x02BC: "XMP", 0x9286: "UserComment", 0x9C9C: "XPComment", 0x8298: "Copyright"}


def _tiff_fields(tiff: bytes) -> dict:
    out: dict = {}
    if tiff[:4] == b"II*\x00":
        e = "<"
    elif tiff[:4] == b"MM\x00*":
        e = ">"
    else:
        return out
    n = len(tiff)
    seen = set()
    todo = [struct.unpack(e + "I", tiff[4:8])[0]]
    while todo and len(seen) < 8:
        off = todo.pop()
        if off in seen or off + 2 > n:
            continue
        seen.add(off)
        count = struct.unpack(e + "H", tiff[off:off + 2])[0]
        for k in range(min(count, 512)):
            p = off + 2 + 12 * k
            if p + 12 > n:
                break
            tag, typ, cnt = struct.unpack(e + "HHI", tiff[p:p + 8])
            if tag == 0x8769:  # Exif sub-IFD
                todo.append(struct.unpack(e + "I", tiff[p + 8:p + 12])[0])
                continue
            if tag not in _TIFF_TAGS:
                continue
            size = cnt * {1: 1, 2: 1, 7: 1}.get(typ, 0)
            if size == 0 or size > MAX_XMP_BYTES:
                continue
            if size <= 4:
                raw = tiff[p + 8:p + 8 + size]
            else:
                vo = struct.unpack(e + "I", tiff[p + 8:p + 12])[0]
                raw = tiff[vo:vo + size]
            name = _TIFF_TAGS[tag]
            if name == "XMP":
                out[name] = raw
            elif name == "XPComment":
                out[name] = raw.decode("utf-16-le", "replace").rstrip("\x00")
            elif name == "UserComment":
                head, body = raw[:8], raw[8:]
                enc = "utf-16-le" if head.startswith(b"UNICODE") else "latin-1"
                out[name] = body.decode(enc, "replace").rstrip("\x00 ")
            else:
                out[name] = raw.decode("utf-8", "replace").rstrip("\x00 ")
    return out


_XMP_FALLBACK = re.compile(rb"<x:xmpmeta[\s>].{0,%d}?</x:xmpmeta>" % MAX_XMP_BYTES, re.S)
_XPACKET_FALLBACK = re.compile(rb"<\?xpacket begin=.{0,%d}?<\?xpacket end=[^>]{0,20}\?>" % MAX_XMP_BYTES, re.S)


def collect(data: bytes) -> dict:
    """Pull XMP packets, text fields and where each came from out of the file."""
    xmp: list[tuple[str, bytes]] = []     # (where, packet)
    texts: list[tuple[str, str, str]] = []  # (where, key, value)
    exif_blobs: list[tuple[str, bytes]] = []

    # JPEG
    ext_parts: dict[bytes, dict[int, bytes]] = {}
    for marker, payload in _jpeg_segments(data):
        if marker == 0xE1 and payload.startswith(XMP_JPEG_SIG):
            xmp.append(("JPEG XMP", payload[len(XMP_JPEG_SIG):]))
        elif marker == 0xE1 and payload.startswith(XMP_EXT_SIG):
            body = payload[len(XMP_EXT_SIG):]
            if len(body) >= 40:
                guid, off = body[:32], struct.unpack(">I", body[36:40])[0]
                ext_parts.setdefault(guid, {})[off] = body[40:]
        elif marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            exif_blobs.append(("JPEG EXIF", payload[6:]))
        elif marker == 0xFE:
            texts.append(("JPEG comment", "Comment", payload[:MAX_TEXT_CHUNK].decode("utf-8", "replace")))
    for guid, parts in ext_parts.items():
        joined = b"".join(parts[k] for k in sorted(parts))
        if len(joined) <= MAX_XMP_BYTES:
            xmp.append(("JPEG extended XMP", joined))

    # PNG
    for ctype, body in _png_chunks(data):
        if ctype in (b"tEXt", b"zTXt", b"iTXt"):
            kv = _png_text(ctype, body)
            if not kv:
                continue
            key, val = kv
            if key == "XML:com.adobe.xmp":
                xmp.append(("PNG XMP", val.encode("utf-8")))
            else:
                texts.append((f"PNG text '{key}'", key, val))
        elif ctype == b"eXIf":
            exif_blobs.append(("PNG EXIF", body[6:] if body.startswith(b"Exif\x00\x00") else body))

    # WebP
    for ctype, body in _riff_chunks(data):
        if ctype == b"XMP ":
            xmp.append(("WebP XMP", body))
        elif ctype == b"EXIF":
            exif_blobs.append(("WebP EXIF", body[6:] if body.startswith(b"Exif\x00\x00") else body))

    # TIFF
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        exif_blobs.append(("TIFF", data))

    for where, blob in exif_blobs:
        for k, v in _tiff_fields(blob).items():
            if k == "XMP":
                xmp.append((f"{where} XMP", v))
            else:
                texts.append((f"{where} {k}", k, v))

    # Fallback for any other file type (PDF, MP4, HEIC, raw XMP sidecar...)
    if not xmp:
        for rx in (_XPACKET_FALLBACK, _XMP_FALLBACK):
            for m in rx.finditer(data):
                xmp.append(("embedded XMP", m.group(0)))
                if len(xmp) >= MAX_PACKETS:
                    break
            if xmp:
                break
    return {"xmp": xmp[:MAX_PACKETS], "texts": texts}


# ================================================================ XMP parsing

def parse_xmp(packet: bytes):
    """Parse an XMP packet safely. Returns an Element or None.

    XMP never needs a DOCTYPE, so any packet with one is refused: this
    blocks entity tricks. Size is capped. ElementTree never fetches
    external resources.
    """
    if len(packet) > MAX_XMP_BYTES:
        return None
    if b"<!DOCTYPE" in packet or b"<!ENTITY" in packet:
        return None
    text = packet.strip(b"\x00 \r\n\t")
    m = re.search(rb"<x:xmpmeta[\s>].*</x:xmpmeta>", text, re.S) or re.search(rb"<rdf:RDF[\s>].*</rdf:RDF>", text, re.S)
    if m:
        text = m.group(0)
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _ns(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def normalise_dst(value: str) -> str | None:
    """Turn a DigitalSourceType value (URI, short code or qcode) into the IPTC code."""
    v = (value or "").strip()
    m = _IPTC_URI.match(v)
    if m:
        v = m.group(1)
    elif v.lower().startswith("digsrctype:"):
        v = v.split(":", 1)[1]
    return _IPTC_LOWER.get(v.lower())


def _iptc_values(root) -> list[str]:
    vals = []
    for el in root.iter():
        for an, av in el.attrib.items():
            if _local(an) == "DigitalSourceType":
                vals.append(av)
        if _local(el.tag) == "DigitalSourceType":
            res = el.attrib.get(f"{{{RDF_NS}}}resource")
            if res:
                vals.append(res)
            if el.text and el.text.strip():
                vals.append(el.text.strip())
            for li in el.iter(f"{{{RDF_NS}}}li"):
                if li.text and li.text.strip():
                    vals.append(li.text.strip())
    return vals


def _parse_aigc_json(text: str) -> dict | None:
    t = (text or "").strip()
    m = re.match(r"^AIGC\s*[=:]\s*", t)
    if m:
        t = t[m.end():]
    if not t.startswith("{") or len(t) > 64 * 1024:
        return None
    try:
        obj = json.loads(t)
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("AIGC"), dict):
        obj = obj["AIGC"]
    if not isinstance(obj, dict):
        return None
    norm = {k.lower(): k for k in AIGC_FIELDS}
    out = {norm[k.lower()]: str(v) for k, v in obj.items() if isinstance(k, str) and k.lower() in norm}
    return out if ("Label" in out or "ContentProducer" in out) else None


def _aigc_from_xmp(root) -> list[dict]:
    found = []
    field_names = {f.lower(): f for f in AIGC_FIELDS}
    for el in root.iter():
        # attribute form: <rdf:Description TC260:AIGC='{...}'> or TC260:Label="1"
        tc_attrs = {}
        for an, av in el.attrib.items():
            if _local(an) == "AIGC":
                d = _parse_aigc_json(av)
                if d:
                    found.append(d)
            elif "tc260" in _ns(an).lower() and _local(an).lower() in field_names:
                tc_attrs[field_names[_local(an).lower()]] = av
        if "Label" in tc_attrs or "ContentProducer" in tc_attrs:
            found.append(tc_attrs)
        if _local(el.tag) == "AIGC":
            d = _parse_aigc_json(el.text or "")
            if d:
                found.append(d)
                continue
            sub = {}
            for child in el.iter():
                ln = _local(child.tag).lower()
                if ln in field_names and child.text and child.text.strip():
                    sub[field_names[ln]] = child.text.strip()
                for an, av in child.attrib.items():
                    if _local(an).lower() in field_names:
                        sub[field_names[_local(an).lower()]] = av
            if "Label" in sub or "ContentProducer" in sub:
                found.append(sub)
    # element form directly under a description in a tc260 namespace
    for el in root.iter():
        sub = {}
        for child in el:
            if "tc260" in _ns(child.tag).lower() and _local(child.tag).lower() in field_names and child.text:
                sub[field_names[_local(child.tag).lower()]] = child.text.strip()
        if "Label" in sub or "ContentProducer" in sub:
            found.append(sub)
    return found


def _creator_tools(root) -> list[str]:
    out = []
    for el in root.iter():
        for an, av in el.attrib.items():
            if _local(an) in ("CreatorTool", "Software", "softwareAgent"):
                out.append(av)
        if _local(el.tag) in ("CreatorTool", "Software", "softwareAgent") and el.text and el.text.strip():
            out.append(el.text.strip())
    return out


# ================================================================ hints

def _sd_settings(key: str, val: str) -> str | None:
    """Return the tool name if a PNG text chunk looks like SD generation settings."""
    k = key.strip().lower()
    if k == "parameters":
        hits = sum(1 for w in ("Steps:", "Sampler:", "CFG scale:", "Seed:", "Model:") if w in val)
        if hits >= 3:
            return "AUTOMATIC1111-style Stable Diffusion settings"
    if k in ("prompt", "workflow"):
        try:
            obj = json.loads(val)
        except ValueError:
            return None
        if k == "prompt" and isinstance(obj, dict) and obj and all(
                isinstance(v, dict) and "class_type" in v for v in obj.values()):
            return "ComfyUI workflow"
        if k == "workflow" and isinstance(obj, dict) and isinstance(obj.get("nodes"), list) and "links" in obj:
            return "ComfyUI workflow"
    if k in ("invokeai_metadata", "sd-metadata", "dream"):
        return "InvokeAI settings"
    return None


def _tool_name(value: str) -> str | None:
    for rx, name in _TOOL_RE:
        if rx.search(value or ""):
            return name
    return None


def _short(s: str, n: int = 120) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


# ================================================================ main

def _describe_aigc(d: dict) -> tuple[str, dict]:
    label = d.get("Label", "").strip()
    meaning = AIGC_LABELS.get(label)
    if meaning:
        summary = f"The file carries China's AIGC label saying {meaning} (Label {label})."
    else:
        summary = "The file carries China's AIGC label (its Label value is missing or not one we know)."
    details = {
        "label value": f"{label} ({meaning})" if meaning else (label or "missing"),
        "made by (service code)": d.get("ContentProducer", ""),
        "maker's file ID": d.get("ProduceID", ""),
        "shared by (platform code)": d.get("ContentPropagator", ""),
        "sharer's file ID": d.get("PropagateID", ""),
    }
    return summary, details


def check(path, data, kind, options) -> list[Finding]:
    got = collect(data)
    findings: list[Finding] = []
    notes: list[str] = []
    unreadable = 0

    iptc: dict[str, list[str]] = {}       # code -> where
    aigc: list[tuple[str, dict]] = []
    tools: list[tuple[str, str, str]] = []  # (where, tool, raw)
    settings: list[tuple[str, str]] = []    # (where, kind)

    for where, packet in got["xmp"]:
        root = parse_xmp(packet)
        if root is None:
            unreadable += 1
            continue
        for v in _iptc_values(root):
            code = normalise_dst(v)
            if code:
                iptc.setdefault(code, []).append(where)
        for d in _aigc_from_xmp(root):
            aigc.append((where, d))
        for t in _creator_tools(root):
            name = _tool_name(t)
            if name:
                tools.append((where + " CreatorTool", name, t))

    for where, key, val in got["texts"]:
        if key.strip().upper() == "AIGC":
            d = _parse_aigc_json(val)
            if d:
                aigc.append((where, d))
                continue
        m = re.search(r"AIGC\s*[=:]\s*\{", val or "")
        if m:
            d = _parse_aigc_json(val[m.start():val.find("}", m.end()) + 1])
            if d:
                aigc.append((where, d))
                continue
        s = _sd_settings(key, val)
        if s:
            settings.append((where, s))
            continue
        if key in ("Software",):
            name = _tool_name(val)
            if name:
                tools.append((where, name, val))

    # Last resort for files we cannot take apart (video, PDF, other): a raw AIGC={...} key.
    if not aigc and kind != "text":
        for m in _AIGC_RAW.finditer(data[: 64 * 1024 * 1024]):
            d = _parse_aigc_json(m.group(1).decode("utf-8", "replace"))
            if d:
                aigc.append(("raw file bytes", d))
                break

    # ---- IPTC
    flagged = [c for c in iptc if IPTC_TYPES[c][0] in GROUP_SUMMARY]
    order = {"ai": 0, "ai_part": 1, "program": 2}
    flagged.sort(key=lambda c: order[IPTC_TYPES[c][0]])
    if flagged:
        top = flagged[0]
        group = IPTC_TYPES[top][0]
        proves = ["The file contains an IPTC tag that describes it as "
                  + {"ai": "made by AI.", "ai_part": "partly made or edited with AI.",
                     "program": "made by a computer program (not trained AI)."}[group]]
        findings.append(Finding(
            NAME, "AI tag in metadata (IPTC)", "found",
            GROUP_SUMMARY[group].format(code=top),
            details={
                "tag value": ", ".join(flagged),
                "IPTC meaning": "; ".join(IPTC_TYPES[c][1] for c in flagged),
                "where in the file": ", ".join(sorted({w for c in flagged for w in iptc[c]})),
            },
            proves=proves,
            does_not_prove=[PLAIN_TEXT,
                            "It does not name the AI tool or company unless other fields do."],
        ))
    for c in iptc:
        g, meaning = IPTC_TYPES[c]
        if c in flagged:
            continue
        if g == "maybe":
            notes.append(f"An IPTC tag says '{meaning}' ({c}); that may or may not involve AI, so it is not counted.")
        else:
            notes.append(f"An IPTC tag describes the source as '{meaning}' ({c}); that is a plain-text claim and does not show how the file was made.")

    # ---- AIGC
    if aigc:
        where, d = aigc[0]
        summary, details = _describe_aigc(d)
        details["where in the file"] = ", ".join(sorted({w for w, _ in aigc}))
        findings.append(Finding(
            NAME, "China AIGC label in metadata", "found", summary, details=details,
            proves=["The file contains a field in the layout China's AIGC labelling rules use."],
            does_not_prove=[
                PLAIN_TEXT,
                "The service codes are not checked against any register; the protection codes (ReservedCode) are not checked.",
                "We read this field's layout from secondary sources (cloud providers' guides), not from the official "
                "GB 45438-2025 text, so files written to the exact standard may differ.",
            ],
        ))

    # ---- generation settings (hint)
    if settings:
        findings.append(Finding(
            NAME, "Generation settings in metadata (hint)", "found",
            f"The file contains text in the format AI image tools write ({settings[0][1]}).",
            details={"where in the file": ", ".join(sorted({w for w, _ in settings})),
                     "format": ", ".join(sorted({s for _, s in settings}))},
            proves=["The file contains text laid out like a Stable Diffusion tool's generation settings."],
            does_not_prove=[PLAIN_TEXT, "Anyone can paste such text into any image, so it is only a hint."],
        ))

    # ---- software name (hint)
    if tools:
        names = sorted({t for _, t, _ in tools})
        findings.append(Finding(
            NAME, "AI tool named in metadata (hint)", "found",
            f"The file's Software field names an AI tool ({', '.join(names)}).",
            details={"field text": "; ".join(sorted({_short(r) for _, _, r in tools})),
                     "where in the file": ", ".join(sorted({w for w, _, _ in tools}))},
            proves=["The file's software field names an AI image tool."],
            does_not_prove=[PLAIN_TEXT,
                            "A software name says which program last wrote the file, which may not be the one that made the picture."],
        ))

    if findings:
        return findings

    if not got["xmp"] and not got["texts"]:
        summary = "No metadata tags about AI were found (the file has no XMP or text metadata)."
    else:
        summary = "No metadata tags about AI were found."
    if notes:
        summary += " " + " ".join(notes)
    if unreadable:
        summary += f" {unreadable} XMP block(s) could not be read."
    return [Finding(NAME, LABEL, "not_found", summary,
                    details={"xmp blocks": len(got["xmp"]), "text fields": len(got["texts"])},
                    does_not_prove=[LOST])]
