"""Every fixture, through text and JSON output: no banned judgement words.

markcheck may only say what a label is and who signed it. It must never
judge the content ("real", "fake", "authentic", "not AI", "human-made" ...).
"""
import json
import re

import pytest

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

from markcheck.cli import main
from markcheck.report import NO_LABEL

BANNED = re.compile(
    r"\b(human[- ]made|made by a human|not AI|authentic|authenticity|genuine|real|true|verified|fake)\b",
    re.I,
)

# Clear non-judgement uses, quoted from an outside vocabulary.
ALLOWED = [
    "sampled from real life",  # IPTC's own name for the digitalCapture code, shown in quotes
]


def _banned_hits(text: str) -> list[str]:
    for phrase in ALLOWED:
        text = re.sub(re.escape(phrase), "", text, flags=re.I)
    hits = []
    for line in text.splitlines():
        if BANNED.search(line):
            hits.append(line.strip())
    return hits


@pytest.fixture(scope="module")
def text_out(request):
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([str(FIXTURES)])
    return buf.getvalue()


@pytest.fixture(scope="module")
def json_doc():
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([str(FIXTURES), "--json"])
    return json.loads(buf.getvalue())


def test_text_output_has_no_banned_words(text_out):
    assert _banned_hits(text_out) == []


def test_json_wording_has_no_banned_words(json_doc):
    lines = []
    for f in json_doc["files"]:
        for fd in f["findings"]:
            lines.append(fd["summary"])
            lines += fd["proves"]
            lines += fd["does_not_prove"]
            lines += [str(v) for v in fd["details"].values()]
    lines.append(json.dumps(json_doc.get("summary", "")))
    assert _banned_hits("\n".join(lines)) == []


def test_no_label_wording_exact(text_out):
    assert NO_LABEL == "No label found. This does not prove it was made by a person."
    assert NO_LABEL in text_out


def test_valid_signature_wording(text_out):
    assert re.search(r"[Ss]igned by .+; unchanged since signing", text_out)


def test_banned_regex_catches_the_obvious():
    assert _banned_hits("This photo is Real.")
    assert _banned_hits("It is not AI")
    assert _banned_hits("human-made picture")
    assert _banned_hits("label verified")
    assert not _banned_hits("unverified address")
    assert not _banned_hits("Digital capture sampled from real life")


def _all_strings(obj, skip=("path",)):
    """Every key and string value in the JSON, except file paths (which are the user's own names)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            yield k
            yield from _all_strings(v, skip)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v, skip)
    elif isinstance(obj, str):
        yield obj


def _json_hits(doc) -> list[str]:
    return _banned_hits("\n".join(_all_strings(doc)))


def test_json_every_field_has_no_banned_words(json_doc):
    """Labels, notes, maker entries, problems and detail keys too, not only summaries."""
    assert _json_hits(json_doc) == []


def test_json_scan_reaches_maker_notes_and_labels(json_doc, monkeypatch):
    import copy

    doc = copy.deepcopy(json_doc)
    f = next(f for f in doc["files"] if f["only_the_maker_can_check"])
    f["only_the_maker_can_check"][0]["note"] = "a genuine checker"
    assert _json_hits(doc)
    doc = copy.deepcopy(json_doc)
    doc["files"][0]["findings"][0]["label"] = "Real label"
    assert _json_hits(doc)


def _run(args):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(args)
    return buf.getvalue()


def test_setup_and_missing_library_branches_have_no_banned_words(monkeypatch):
    """Branches the committed fixtures never reach: model files missing, libraries missing."""
    import sys

    from markcheck.detectors import trustmark_check

    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant="Q": ["decoder_Q.ckpt"])
    monkeypatch.setitem(sys.modules, "c2pa", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    target = str(FIXTURES / "watermark")
    assert _banned_hits(_run([target])) == []
    assert _json_hits(json.loads(_run([target, "--json"]))) == []


def test_damaged_and_online_label_branches_have_no_banned_words():
    from markcheck.detectors import c2pa_check

    for f in (c2pa_check._from_read_error(Exception("remote manifest at https://example.test/x"), b""),
              c2pa_check._damaged("bad box"),
              c2pa_check._from_read_error(Exception("type is unsupported"), b"")):
        assert _banned_hits("\n".join([f.summary, *f.proves, *f.does_not_prove,
                                       *map(str, f.details.values())])) == []
