"""`markcheck <file>` on plain-text files with and without a hidden C2PA label."""
from __future__ import annotations

from markcheck.cli import check_file
from markcheck.detectors import text_label
from markcheck.textcheck import hidden


def label() -> str:
    payload = (24).to_bytes(4, "big") + b"jumb" + bytes(16)
    framed = hidden.MAGIC + bytes([1]) + len(payload).to_bytes(4, "big") + payload
    return hidden.MARKER + "".join(chr(0xFE00 + b) if b < 0x10 else chr(0xE0100 + b - 0x10) for b in framed)


def finding(rep):
    return next(f for f in rep.findings if f.detector == "text_label")


def test_text_file_with_label_is_found(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("Meeting moved to Tuesday." + label(), encoding="utf-8")
    f = finding(check_file(p))
    assert f.status == "found" and f.details["label"]["looks_like_c2pa"]
    assert any("signature" in d for d in f.does_not_prove)


def test_plain_text_file_finds_nothing(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Notes\n\nNothing hidden here.", encoding="utf-8")
    f = finding(check_file(p))
    assert f.status == "not_found" and f.summary == "No label is hidden in this text."


def test_invisible_characters_alone_are_not_a_label(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("Hello​ world", encoding="utf-8")
    f = finding(check_file(p))
    assert f.status == "not_found" and "Invisible characters found" in f.summary


def test_pictures_are_skipped_as_not_applicable():
    f = text_label.check(None, b"\x89PNG", "image", {})[0]
    assert f.status == "skipped" and f.summary.startswith("Only text files can carry")


def test_non_utf8_text_is_skipped(tmp_path):
    p = tmp_path / "old.txt"
    p.write_bytes(b"caf\xe9")
    assert finding(check_file(p)).status == "skipped"
