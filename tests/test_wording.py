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
