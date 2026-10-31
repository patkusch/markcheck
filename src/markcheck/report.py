"""Plain-English text output.

Wording rules (hard):
- Never say "human-made" or "not AI". Say "no label found; this does not
  prove it was made by a person".
- Never say "authentic" or "true". Say "signed by X; unchanged since signing".
"""
from __future__ import annotations

from .findings import FileReport

NO_LABEL = "No label found. This does not prove it was made by a person."

MARK = {"found": "FOUND", "not_found": "none", "skipped": "skipped", "error": "error"}


def render_file(rep: FileReport) -> str:
    lines = [f"== {rep.path}  ({rep.kind})"]
    if rep.any_found:
        lines.append("Result: an AI-related label was found. Details below.")
    else:
        lines.append("Result: " + NO_LABEL)
    lines.append("")
    for f in rep.findings:
        lines.append(f"  [{MARK[f.status]}] {f.label}: {f.summary}")
        if f.found:
            for k, v in f.details.items():
                if v not in (None, "", [], {}):
                    lines.append(f"      {k.replace('_', ' ')}: {v}")
            for s in f.proves:
                lines.append(f"      This shows: {s}")
            for s in f.does_not_prove:
                lines.append(f"      This does not show: {s}")
    if rep.makers:
        lines.append("")
        lines.append("  Only the maker can check this:")
        lines.append("  Some companies add hidden marks that only their own checker can read.")
        for m in rep.makers:
            lines.append(f"    - {m['who']}: {m['where']}  ({m['note']})")
    return "\n".join(lines)


def render_text(reports: list[FileReport]) -> str:
    if not reports:
        return "No files to check."
    parts = [render_file(r) for r in reports]
    found = sum(1 for r in reports if r.any_found)
    parts.append("")
    parts.append(f"Checked {len(reports)} file(s). Label found in {found}. "
                 f"No label found in {len(reports) - found}.")
    if found < len(reports):
        parts.append("Remember: a missing label does not prove a file was made by a person. "
                     "Labels are often lost when files are shared or re-saved.")
    return "\n".join(parts)
