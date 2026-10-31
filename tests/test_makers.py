from markcheck.makers import makers_for


def _who(kind):
    return " ".join(m["who"] for m in makers_for(kind, []))


def test_entries_have_shape():
    for kind in ("image", "audio", "video", "text", "pdf", "document"):
        entries = makers_for(kind, [])
        assert entries, kind
        for m in entries:
            assert m["who"] and m["where"] and m["note"]
            assert m["note"].count(". ") == 0  # one sentence


def test_image_entries():
    who = _who("image")
    for name in ("SynthID", "OpenAI", "Microsoft", "Amazon", "Anthropic"):
        assert name in who
    assert any("openai.com/verify" in m["where"] for m in makers_for("image", []))
    assert any("claude.com/check-content" in m["where"] for m in makers_for("image", []))


def test_audio_and_video():
    assert "ElevenLabs" in _who("audio") and "SynthID" in _who("audio")
    assert not any(m["who"].startswith("ElevenLabs") for m in makers_for("video", [])) and "SynthID" in _who("video")
    assert "Amazon" not in _who("audio")


def test_text_gets_text_marks_only():
    text = makers_for("text", [])
    who = _who("text")
    assert "Claude text" in who and "SynthID" in who
    assert not any("check-content" in m["where"] for m in text)
    assert not any("openai.com/verify" in m["where"] for m in text)


def test_synthid_says_it_is_shared():
    for kind in ("image", "text"):
        syn = [m for m in makers_for(kind, []) if "SynthID" in m["who"]][0]
        assert "does not say which one" in syn["note"]


def test_unknown_kind_empty():
    assert makers_for("other", []) == []


def test_returns_copies():
    makers_for("image", [])[0]["who"] = "changed"
    assert makers_for("image", [])[0]["who"] != "changed"


def test_addresses_we_did_not_open_say_so_in_the_printed_note():
    """An address we could not confirm by opening the page must say that where people read it."""
    from markcheck import makers

    entries = {id(e): e for lst in makers.BY_KIND.values() for e in lst}.values()
    for e in entries:
        if e["checked"].startswith(("unverified", "partly")):
            assert any(w in e["note"] for w in ("not opened", "refused", "unverified")), e["who"]
