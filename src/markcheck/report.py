"""Plain-English text output.

Wording rules (hard; tests/test_wording.py checks them):
- Never say "human-made" or "not AI". Say "No label found. This does not
  prove it was made by a person."
- Never say "authentic", "genuine", "real", "true" or "verified" about
  content. Say "signed by X; unchanged since signing".

Layout: for each file a heading, a one-line verdict, then the details of any
label found, then one line each for checks that found nothing or did not
apply. Jargon is explained in brackets the first time it appears in a run.
Things that would repeat for every file (the "only the maker can check"
list, setup notes such as missing TrustMark model files) are printed once
at the end.
"""
from __future__ import annotations

import re

from .findings import FileReport, Finding

NO_LABEL = "No label found. This does not prove it was made by a person."

KIND_ONE = {"image": "picture", "video": "video", "audio": "sound file", "text": "text file",
            "pdf": "PDF file", "document": "document", "other": "other file"}
KIND_MANY = {"image": "pictures", "video": "videos", "audio": "sound files", "text": "text files",
             "pdf": "PDF files", "document": "documents", "other": "other files"}

# Detail keys shown in text, with plain names. None = JSON only.
DETAIL_LABELS: dict[str, str | None] = {
    # signed label (C2PA)
    "signed_by": "Signed by",
    "certificate_issued_by": "Signer's certificate issued by",
    "signed_at": "Signed at",
    "made_with_tool": "Label written by",
    "software_or_ai_named": "Software or AI named in the label",
    "source_type": "How the label says it was made",
    "says_made_with_ai": "Says AI was used",
    "earlier_label_says_ai": "An earlier label inside says AI was used",
    "signature_valid": "Signature checks out",
    "changed_since_signing": "Changed since signing",
    "on_c2pa_trust_list": "Signer is on the C2PA trust list",
    "test_or_self_signed_certificate": "Test or home-made certificate",
    "earlier_labels": "Earlier labels inside",
    "label_location": "Label kept",
    "label_readable": "Label readable",
    "problems": None,
    "read_problem": None,
    # metadata
    "tag value": "Tag value",
    "IPTC meaning": "What the IPTC tag means",
    "where in the file": "Found in",
    "label value": "Label value",
    "xmp blocks": None,
    "text fields": None,
    # Stable Diffusion mark
    "best_match": "Best match",
    "bit_match": None,
    # TrustMark
    "variant": None,
    "payload_bits": None,
    "payload_hex": "Hidden message (as a number)",
    "schema": None,
    "payload_text": "Hidden message (as text)",
}

# Reasons for skipping that mean "this check does not apply to this kind of file".
NOT_APPLICABLE = ("Only pictures can carry", "not used in this kind of file", "cannot be read from this file type")
# Reasons that are about this computer's setup, not about the file: told once per run.
SETUP = ("markcheck.fetch_models", "pip install", "library is not available")

# Jargon, explained the first time it appears in a run. (term pattern, explanation)
GLOSSARY: list[tuple[str, str]] = [
    (r"\bC2PA\b", "C2PA: an open standard for signed labels, also called Content Credentials. "
                  "The label says which tool made the file. The signature shows whether the file changed since."),
    (r"\btrust list\b", "trust list: the official C2PA list of signing certificates whose owners have been checked."),
    (r"\bcertificate\b", "certificate: a digital ID that names the signer. Anyone can make one, "
                         "so the name only counts if a trusted body issued it."),
    (r"\bmetadata\b", "metadata: notes stored inside a file about the file, such as which program saved it."),
    (r"\bXMP\b", "XMP: a standard block of text inside a file where programs write such notes."),
    (r"\bIPTC\b", "IPTC: a news-industry body. Its \"digital source type\" tag says how a picture was made."),
    (r"\bAIGC\b", "AIGC: short for AI-generated content. China requires this label on files made by AI."),
    (r"\bhidden mark", "hidden mark: a pattern woven into the pixels that people cannot see. "
                       "It is also called a watermark."),
    (r"\bpayload\b", "payload: the short fixed message that a hidden mark carries."),
    (r"\bbits?\b", "bit: one yes-or-no piece of a hidden message."),
    (r"\bTrustMark\b", "TrustMark: an open hidden mark made by Adobe. Anyone can add or read it."),
    (r"\bSynthID\b", "SynthID: a hidden mark made by Google that some other companies also use. "
                        "Only their own checkers can read it."),
]


class Glossary:
    """Adds a one-line explanation after the first line that uses each term."""

    def __init__(self):
        self.done: set[str] = set()

    def apply(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        for line in lines:
            out.append(line)
            if not line.lstrip().startswith("=="):  # file names are not jargon
                out.extend(self._explain(line))
        return out

    def _explain(self, line: str) -> list[str]:
        hits = []
        for pat, expl in GLOSSARY:
            if pat in self.done:
                continue
            m = re.search(pat, line, re.I if pat != r"\bbits?\b" else 0)
            if m:
                hits.append((m.start(), pat, expl))
        hits.sort()
        indent = " " * (len(line) - len(line.lstrip()) + 2)
        extra: list[str] = []
        for _, pat, expl in hits:
            self.done.add(pat)
            extra.append(f"{indent}({expl})")
        # An explanation may itself use a term not yet explained; mark it as seen so it
        # is explained once, right after.
        more: list[str] = []
        for e in extra:
            more.extend(self._explain(e))
        return extra + more


def _value(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple, set)):
        return ", ".join(_value(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_value(x)}" for k, x in v.items())
    return str(v)


def _detail_label(key: str) -> str | None:
    if key in DETAIL_LABELS:
        return DETAIL_LABELS[key]
    k = key.replace("_", " ")
    return k[:1].upper() + k[1:]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9'\"])", text.strip()) if s]


def _plain_error(text: str) -> str:
    return re.sub(r"\b[A-Z][A-Za-z]*(Error|Exception|Warning): ", "", text)


def _skip_kind(f: Finding) -> str:
    if any(s in f.summary for s in SETUP):
        return "setup"
    if any(s in f.summary for s in NOT_APPLICABLE):
        return "n/a"
    return "other"


def _render_found(f: Finding) -> list[str]:
    lines = [f"   {f.label}", f"     {f.summary}"]
    shown = []
    for k, v in f.details.items():
        name = _detail_label(k)
        if name is None or v in (None, "", [], {}):
            continue
        shown.append(f"       {name}: {_value(v)}")
    if shown:
        lines.append("     Details:")
        lines += shown
    if f.proves:
        lines.append("     What this shows:")
        lines += [f"       - {_cap(s)}" for s in f.proves]
    if f.does_not_prove:
        lines.append("     What this does not show:")
        lines += [f"       - {_cap(s)}" for s in f.does_not_prove]
    return lines


def _cap(s: str) -> str:
    s = s.strip()
    s = s[:1].upper() + s[1:]
    return s if s.endswith((".", "!", "?")) else s + "."


def render_file(rep: FileReport) -> list[str]:
    lines = [f"== {rep.path}  ({KIND_ONE.get(rep.kind, rep.kind)})"]
    found = [f for f in rep.findings if f.found]
    if found:
        names = []
        for f in found:
            if f.label not in names:
                names.append(f.label)
        lines.append(f"   Label found: {'; '.join(names)}.")
    else:
        lines.append(f"   {NO_LABEL}")
        if rep.any_error:
            lines.append("   Some checks could not finish, so this file was not fully checked.")
    for n in rep.notes:
        lines.append(f"   Note: {n}")

    for f in found:
        lines.append("")
        lines += _render_found(f)

    nothing = [f for f in rep.findings if f.status == "not_found"]
    skipped = [f for f in rep.findings if f.status == "skipped"]
    errors = [f for f in rep.findings if f.status == "error"]
    if found and (nothing or skipped or errors):
        lines.append("")
    if nothing:
        lines.append(f"   Checked, nothing found: {', '.join(_unique(f.label for f in nothing))}.")
        for f in nothing:
            extra = _sentences(f.summary)[1:]
            if extra:
                lines.append(f"   Note on {f.label}: " + " ".join(extra))
    na = [f.label for f in skipped if _skip_kind(f) == "n/a"]
    setup = [f.label for f in skipped if _skip_kind(f) == "setup"]
    other = [f for f in skipped if _skip_kind(f) == "other"]
    if na:
        lines.append(f"   Not checked, as they do not apply to this kind of file: {', '.join(_unique(na))}.")
    if setup:
        lines.append(f"   Not checked, as this computer is missing something they need: "
                     f"{', '.join(_unique(setup))} (see the setup note at the end).")
    for f in other:
        lines.append(f"   Not checked: {f.label}. {f.summary}")
    for f in errors:
        lines.append(f"   Could not finish: {f.label}. {_plain_error(f.summary)}")
    return lines


def _unique(items) -> list[str]:
    out: list[str] = []
    for i in items:
        if i not in out:
            out.append(i)
    return out


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _render_makers(reports: list[FileReport]) -> list[str]:
    """Each maker-only checker once, with the kinds of file in this run it applies to."""
    makers: dict[str, dict] = {}
    kinds: dict[str, list[str]] = {}
    for r in reports:
        for m in r.makers:
            makers.setdefault(m["who"], m)
            ks = kinds.setdefault(m["who"], [])
            if r.kind not in ks:
                ks.append(r.kind)
    if not makers:
        return []
    lines = ["", "Only the maker can check these",
             "Some companies add hidden marks that only their own checker can read. "
             "markcheck cannot look for them. These checkers may help:"]
    for who, m in makers.items():
        lines.append(f"  - {who}")
        lines.append(f"      For: {_join([KIND_MANY.get(k, k) for k in kinds[who]])}")
        lines.append(f"      Where: {m['where']}")
        lines.append(f"      {m['note']}")
    return lines


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def summary_counts(reports: list[FileReport], problems: list[dict] | None = None) -> dict:
    found = sum(1 for r in reports if r.any_found)
    problems = problems or []
    with_errors = sum(1 for r in reports if r.any_error)
    text = (f"Checked {_plural(len(reports), 'file', 'files')}. A label was found in {found}. "
            f"No label was found in {len(reports) - found}.")
    if problems:
        text += f" Could not check {_plural(len(problems), 'file or folder', 'files or folders')}."
    return {
        "files_checked": len(reports),
        "label_found_in": found,
        "no_label_in": len(reports) - found,
        "files_with_a_failed_check": with_errors,
        "could_not_check": len(problems),
        "text": text,
    }


def render_text(reports: list[FileReport], problems: list[dict] | None = None) -> str:
    problems = problems or []
    if not reports and not problems:
        return "No files to check."
    lines: list[str] = []
    for r in reports:
        if lines:
            lines.append("")
        lines += render_file(r)

    lines += _render_makers(reports)

    setup_notes = _unique(f.summary for r in reports for f in r.findings
                          if f.status == "skipped" and _skip_kind(f) == "setup")
    if setup_notes:
        lines += ["", "Setup note"]
        lines += [f"  {s}" for s in setup_notes]

    if problems:
        lines += ["", "Could not check"]
        lines += [f"  {p['path']}: {p['problem']}" for p in problems]

    if len(reports) > 1 or problems:
        lines.append("")
        lines.append(summary_counts(reports, problems)["text"])
    else:
        lines.append("")
    if any(not r.any_found for r in reports):
        lines.append("A missing label does not prove a file was made by a person. "
                     "Labels are often lost when files are shared or re-saved.")
    return "\n".join(Glossary().apply(lines))
