"""Finding C2PA text labels and other invisible characters: fast, offline."""
from __future__ import annotations

from markcheck.textcheck import hidden


def encode(payload: bytes, version: int = 1, declared: int | None = None) -> str:
    """Build a label the way the open implementations do, to test that we can read it."""
    n = len(payload) if declared is None else declared
    framed = hidden.MAGIC + bytes([version]) + n.to_bytes(4, "big") + payload
    return hidden.MARKER + "".join(chr(0xFE00 + b) if b < 0x10 else chr(0xE0100 + b - 0x10) for b in framed)


C2PA_LIKE = (64).to_bytes(4, "big") + b"jumb" + bytes(range(56))


def test_byte_mapping_edges():
    assert hidden.vs_to_byte("︀") == 0
    assert hidden.vs_to_byte("️") == 0x0F
    assert hidden.vs_to_byte("\U000E0100") == 0x10
    assert hidden.vs_to_byte("\U000E01EF") == 0xFF
    assert hidden.vs_to_byte("a") is None


def test_finds_a_label_after_the_words():
    text = "The quarterly report is ready." + encode(C2PA_LIKE)
    rep = hidden.scan(text)
    assert rep.label and rep.label.complete and rep.label.looks_like_c2pa
    assert rep.label.version == 1 and rep.label.payload == C2PA_LIKE
    assert rep.visible_text == "The quarterly report is ready."


def test_no_label_in_plain_text():
    rep = hidden.scan("Just words, nothing hidden.")
    assert rep.label is None and rep.invisible == {} and not rep.found_anything
    assert rep.lines() == []


def test_cut_short_label_is_reported_as_unreadable():
    text = "Words." + encode(C2PA_LIKE[:10], declared=len(C2PA_LIKE))
    rep = hidden.scan(text)
    assert rep.label and not rep.label.complete
    assert "cut short" in rep.lines()[0]


def test_marker_without_magic_is_not_a_label():
    rep = hidden.scan("A﻿B with a stray zero-width no-break space.")
    assert rep.label is None
    assert rep.invisible == {"zero-width spaces and joiners": 1}


def test_other_invisible_characters_are_counted_by_kind():
    text = "a​b‍c‮d\U000E0041e­f"
    counts = hidden.invisible_counts(text)
    assert counts == {"zero-width spaces and joiners": 2, "direction controls": 1,
                      "tag characters": 1, "other invisible formatting": 1}


def test_label_characters_are_not_counted_again_in_the_message():
    rep = hidden.scan("Words." + encode(C2PA_LIKE))
    assert len(rep.lines()) == 1  # the label line only, no leftover invisible characters


def test_wording_never_judges_the_writer():
    rep = hidden.scan("Words​ here." + encode(C2PA_LIKE))
    text = " ".join(rep.lines()).lower()
    for banned in ("human-written", "proof", "cheating", "authentic", "plagiar"):
        assert banned not in text


def test_two_pasted_labels_are_both_found_and_removed_from_the_words():
    text = "First part." + encode(C2PA_LIKE) + " Second part." + encode(C2PA_LIKE)
    rep = hidden.scan(text, read_signature=False)
    assert rep.label_count == 2
    assert rep.visible_text == "First part. Second part."
    lines = rep.lines()
    assert any("holds 2 hidden labels" in line for line in lines)
    assert not any("Invisible characters found" in line for line in lines)  # all belong to the labels


def test_one_label_says_nothing_about_a_count():
    rep = hidden.scan("Words." + encode(C2PA_LIKE), read_signature=False)
    assert rep.label_count == 1
    assert not any("hidden labels" in line for line in rep.lines())
    assert rep.to_dict()["label_count"] == 1
