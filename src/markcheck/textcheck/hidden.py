"""Find labels and other characters hidden inside a text.

C2PA can attach a signed label to plain text. The label is turned into invisible
characters (Unicode "variation selectors") and placed after the visible words,
starting with a zero-width no-break space (U+FEFF). This module finds that label
and reads its header. It also counts other invisible characters, which can come
from a label, a tool, or just copy and paste.

The layout follows C2PA Technical Specification 2.4, Annex A.8 ("Embedding
Manifests into Unstructured Text"), as written down by the open implementations
c2pa-unstructured-text (Rust) and c2pa-text (Python). We have not read the
normative text of the annex itself.

When the C2PA library is installed, the label's signature is read too.

This module only reads. It never changes the text.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

MARKER = "﻿"
MAGIC = b"C2PATXT\x00"
HEADER_LEN = 13  # 8 magic + 1 version + 4 length (big-endian)


def vs_to_byte(ch: str) -> int | None:
    """One variation selector back to the byte it stands for, or None."""
    cp = ord(ch)
    if 0xFE00 <= cp <= 0xFE0F:
        return cp - 0xFE00
    if 0xE0100 <= cp <= 0xE01EF:
        return cp - 0xE0100 + 0x10
    return None


@dataclass
class TextLabel:
    version: int
    declared_length: int
    payload_length: int
    complete: bool
    start: int              # character position of the marker
    end: int                # character position just after the selector run
    looks_like_c2pa: bool   # the payload starts like a C2PA manifest store (a JUMBF box)
    payload: bytes = field(repr=False, default=b"")

    def to_dict(self) -> dict:
        return {"version": self.version, "declared_length": self.declared_length,
                "payload_length": self.payload_length, "complete": self.complete,
                "looks_like_c2pa": self.looks_like_c2pa}


def find_text_labels(text: str) -> list[TextLabel]:
    """Every C2PA text label in `text`, in order. Pasting labelled texts together can give several."""
    out, start = [], 0
    while (lab := find_text_label(text, start)) is not None:
        out.append(lab)
        start = lab.end
    return out


def find_text_label(text: str, start: int = 0) -> TextLabel | None:
    """The first C2PA text label in `text` at or after `start`, or None."""
    pos = text.find(MARKER, start)
    while pos != -1:
        run = bytearray()
        i = pos + 1
        while i < len(text):
            b = vs_to_byte(text[i])
            if b is None:
                break
            run.append(b)
            i += 1
        if len(run) >= HEADER_LEN and bytes(run[:8]) == MAGIC:
            declared = int.from_bytes(run[9:13], "big")
            payload = bytes(run[HEADER_LEN:HEADER_LEN + declared])
            return TextLabel(version=run[8], declared_length=declared, payload_length=len(payload),
                             complete=len(payload) == declared, start=pos, end=i,
                             looks_like_c2pa=payload[4:8] == b"jumb", payload=payload)
        pos = text.find(MARKER, pos + 1)
    return None


# Invisible characters worth naming. Anything else in Unicode's "format" class is counted as "other".
NAMED = {
    "zero-width spaces and joiners": {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF},
    "direction controls": set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x200E, 0x200F},
}


BLACK_FLAG = 0x1F3F4  # flags such as England's are this emoji plus tag characters, ending in U+E007F
TAG_END = 0xE007F
MAX_SHOWN = 300


def _flag_positions(text: str) -> set[int]:
    """Positions of tag characters that are part of a flag emoji, which is their ordinary use."""
    out: set[int] = set()
    i = 0
    while i < len(text):
        if ord(text[i]) == BLACK_FLAG:
            j = i + 1
            while j < len(text) and 0xE0020 <= ord(text[j]) <= 0xE007E:
                j += 1
            if j < len(text) and ord(text[j]) == TAG_END and j > i + 1:
                out.update(range(i + 1, j + 1))
                i = j
        i += 1
    return out


def hidden_tag_text(text: str) -> str:
    """Text spelled out in invisible "tag" characters, outside flag emoji. Runs are joined with " | "."""
    flags = _flag_positions(text)
    runs, cur = [], []
    for i, ch in enumerate(text):
        cp = ord(ch)
        if 0xE0020 <= cp <= 0xE007E and i not in flags:
            cur.append(chr(cp - 0xE0000))
        elif cur:
            runs.append("".join(cur))
            cur = []
    if cur:
        runs.append("".join(cur))
    return " | ".join(r for r in runs if r.strip())


def invisible_counts(text: str) -> dict[str, int]:
    """How many invisible characters of each kind the text has."""
    counts: dict[str, int] = {}

    def add(kind: str):
        counts[kind] = counts.get(kind, 0) + 1

    flags = _flag_positions(text)
    for i, ch in enumerate(text):
        cp = ord(ch)
        if i in flags:
            continue  # part of a flag emoji
        if vs_to_byte(ch) is not None:
            add("variation selectors")
        elif 0xE0000 <= cp <= 0xE007F:
            add("tag characters")
        elif any(cp in s for s in NAMED.values()):
            add(next(k for k, s in NAMED.items() if cp in s))
        elif unicodedata.category(ch) == "Cf":
            add("other invisible formatting")
    return counts


@dataclass
class HiddenReport:
    label: TextLabel | None
    invisible: dict[str, int]
    visible_text: str        # the text without the labels' characters, for scoring the words
    label_count: int = 0
    signature: object = None  # a Finding from reading the label's signature, when that worked

    _label_selectors: int = 0  # how many variation selectors all the labels use together
    tag_text: str = ""         # text spelled out in invisible tag characters

    @property
    def found_anything(self) -> bool:
        return self.label is not None or bool(self.invisible)

    def lines(self) -> list[str]:
        out = []
        if self.label:
            size = f"{self.label.payload_length:,} bytes"
            if self.signature is not None:
                out.append(self.signature.summary)
            elif self.label.complete and self.label.looks_like_c2pa:
                out.append(f"A signed label (C2PA) is attached to this text, hidden as invisible characters "
                           f"after the words ({size}). markcheck does not check its signature yet, so it "
                           f"cannot say who signed it or whether the text changed since.")
            elif not self.label.complete:
                out.append(f"The start of a signed label (C2PA) was found, but it is cut short: it says "
                           f"{self.label.declared_length:,} bytes and only {size} are there. It cannot be read.")
            else:
                out.append(f"Something with a signed label's (C2PA) header is attached to this text ({size}), "
                           f"but its contents do not look like a C2PA label.")
        if self.label_count > 1:
            out.append(f"This text holds {self.label_count} hidden labels, perhaps from pasting labelled "
                       f"texts together. Only the first is described here.")
        rest = dict(self.invisible)
        if self.label:
            # The labels' own characters are not news; only count what is left over.
            rest["variation selectors"] = rest.get("variation selectors", 0) - self._label_selectors
            rest["zero-width spaces and joiners"] = rest.get("zero-width spaces and joiners", 0) - self.label_count
            rest = {k: v for k, v in rest.items() if v > 0}
        if self.tag_text:
            shown = self.tag_text if len(self.tag_text) <= MAX_SHOWN else self.tag_text[:MAX_SHOWN] + "..."
            out.append(f"Invisible \"tag\" characters spell out hidden text: \"{shown}\". Hidden text like "
                       f"this can carry instructions meant for an AI tool rather than for the reader.")
        if rest:
            parts = ", ".join(f"{v} {k}" for k, v in sorted(rest.items()))
            out.append(f"Invisible characters found: {parts}. These can come from a label, a writing tool, "
                       f"or copy and paste. On their own they say nothing about who wrote the text.")
        return out

    def to_dict(self) -> dict:
        label = self.label.to_dict() if self.label else None
        if label is not None and self.signature is not None:
            label["signature"] = self.signature.details
        return {"label": label, "label_count": self.label_count, "invisible_characters": self.invisible,
                "hidden_tag_text": self.tag_text or None}


def scan(text: str, read_signature: bool = True) -> HiddenReport:
    labels = find_text_labels(text)
    label = labels[0] if labels else None
    visible, cut = text, 0
    for lab in labels:
        visible = visible[:lab.start - cut] + visible[lab.end - cut:]
        cut += lab.end - lab.start
    signature = None
    if read_signature and label is not None and label.complete and label.looks_like_c2pa:
        from ..detectors.c2pa_check import inspect_text_label
        signature = inspect_text_label(label.payload)
    return HiddenReport(label=label, invisible=invisible_counts(text), visible_text=visible, signature=signature,
                        label_count=len(labels), _label_selectors=sum(lab.end - lab.start - 1 for lab in labels),
                        tag_text=hidden_tag_text(text))
