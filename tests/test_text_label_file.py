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


# ---- real signed labels, made by scripts/make_fixtures_text_label.py ----

FIX = __import__("pathlib").Path(__file__).parent / "fixtures" / "c2pa" / "text"


def test_real_signed_text_label_names_signer_and_ai():
    f = finding(check_file(FIX / "ai_text_label.txt"))
    assert f.status == "found"
    assert f.details["signature_valid"] == "yes"
    assert f.details["signed_by"].startswith("markcheck Test Signer")
    assert f.details["says_made_with_ai"].startswith("yes")
    assert "has not checked that the words are unchanged" in f.summary


def test_real_text_label_never_claims_unchanged():
    f = finding(check_file(FIX / "ai_text_label.txt"))
    assert f.details["changed_since_signing"].startswith("not checked")
    assert "unchanged since signing." not in f.summary.replace("are unchanged since signing", "")
    assert any("words are unchanged" in d for d in f.does_not_prove)


def test_broken_signature_on_text_label_is_reported():
    f = finding(check_file(FIX / "broken_signature.txt"))
    assert f.status == "found" and f.details["signature_valid"] == "no"
    assert "does not check out" in f.summary


def test_test_certificate_is_not_trusted():
    f = finding(check_file(FIX / "ai_text_label.txt"))
    assert f.details["on_c2pa_trust_list"] == "no"
    assert f.details.get("test_or_self_signed_certificate") == "yes"
