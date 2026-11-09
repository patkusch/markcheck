"""Reading text files saved as UTF-8, UTF-16 or UTF-32."""
from __future__ import annotations

import codecs

import pytest

from markcheck.cli import check_file
from markcheck.textcheck.encoding import decode_text

WORDS = "Café menu for Tuesday — soup of the day."


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32"])
def test_decodes_common_encodings(enc):
    raw = WORDS.encode(enc)
    if enc in ("utf-16-le", "utf-16-be"):  # these codecs leave out the marker, so add it as editors do
        raw = (codecs.BOM_UTF16_LE if enc.endswith("le") else codecs.BOM_UTF16_BE) + raw
    assert decode_text(raw) == WORDS


def test_unmarked_non_utf8_still_fails():
    with pytest.raises(UnicodeDecodeError):
        decode_text("café".encode("latin-1"))


def test_utf16_file_with_hidden_label_is_found(tmp_path):
    from markcheck.textcheck import hidden
    payload = (24).to_bytes(4, "big") + b"jumb" + bytes(16)
    framed = hidden.MAGIC + bytes([1]) + len(payload).to_bytes(4, "big") + payload
    label = hidden.MARKER + "".join(chr(0xFE00 + b) if b < 0x10 else chr(0xE0100 + b - 0x10) for b in framed)
    p = tmp_path / "notepad.txt"
    p.write_bytes((WORDS + label).encode("utf-16"))
    f = next(x for x in check_file(p).findings if x.detector == "text_label")
    assert f.status == "found"
