"""Signed labels (C2PA) hidden inside plain-text files.

The label sits after the words as invisible characters. See
markcheck.textcheck.hidden for how it is found. This reads only; it never
changes the file, and it does not check the label's signature yet.
"""
from __future__ import annotations

from ..findings import Finding
from ..textcheck.hidden import scan

NAME = "text_label"
LABEL = "Signed label hidden in text (C2PA)"

DOES_NOT_PROVE = [
    "who signed the label: markcheck does not check text label signatures yet",
    "that the text is unchanged since it was labelled",
    "who wrote the words",
]


def check(path, data, kind, options) -> list[Finding]:
    if kind != "text":
        return [Finding(NAME, LABEL, "skipped", "Only text files can carry this label.")]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [Finding(NAME, LABEL, "skipped", "This text file is not UTF-8, so it was not read.")]
    rep = scan(text)
    details = rep.to_dict()
    if rep.signature is not None:
        f = rep.signature
        # Plain, flat facts: the signature details plus the label's size, not the raw scan.
        f.detector, f.label = NAME, LABEL
        f.details = {**f.details, "label_size": f"{rep.label.payload_length:,} bytes"}
        return [f]
    if rep.label and rep.label.complete and rep.label.looks_like_c2pa:
        return [Finding(NAME, LABEL, "found", rep.lines()[0], details,
                        proves=["something attached a C2PA label to this text"],
                        does_not_prove=DOES_NOT_PROVE)]
    lines = rep.lines()
    summary = " ".join(lines) if lines else "No label is hidden in this text."
    return [Finding(NAME, LABEL, "not_found", summary, details)]
