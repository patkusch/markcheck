"""Turn a text file's bytes into text.

Most text is UTF-8. Some programs, such as Notepad on older Windows, save UTF-16
or UTF-32 with a marker at the start (a "byte order mark") that says which. We
read those too. Anything else must be UTF-8, or decoding fails as before.
"""
from __future__ import annotations

import codecs

BOMS = [
    (codecs.BOM_UTF32_LE, "utf-32"),  # checked before UTF-16: its marker starts the same way
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
]


def decode_text(raw: bytes) -> str:
    """UTF-8 (with or without a marker), or UTF-16/UTF-32 when the file starts with their marker."""
    for bom, name in BOMS:
        if raw.startswith(bom):
            return raw.decode(name)
    return raw.decode("utf-8-sig")
