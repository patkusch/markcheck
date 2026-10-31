"""Command line entry point.

Exit codes: 0 = no label found, 1 = at least one label found, 2 = error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .detectors import run_all
from .filetypes import sniff
from .findings import FileReport
from .makers import makers_for
from .report import render_text

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}


def iter_files(target: Path):
    if target.is_file():
        yield target
        return
    for p in sorted(target.rglob("*")):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts) and not p.name.startswith("."):
            yield p


def check_file(path: Path, options: dict | None = None) -> FileReport:
    data = path.read_bytes()
    kind = sniff(path, data[:64])
    rep = FileReport(path=str(path), kind=kind)
    rep.findings = run_all(path, data, kind, options or {})
    rep.makers = makers_for(kind, rep.findings)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="markcheck",
        description="Look for AI labels in a file or folder and explain what they do and don't prove. "
                    "Runs on your machine; nothing is uploaded.",
    )
    ap.add_argument("target", help="a file or a folder")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    ap.add_argument("--version", action="version", version=f"markcheck {__version__}")
    args = ap.parse_args(argv)

    target = Path(args.target)
    if not target.exists():
        print(f"markcheck: cannot find {target}", file=sys.stderr)
        return 2

    reports: list[FileReport] = []
    had_error = False
    for p in iter_files(target):
        try:
            reports.append(check_file(p))
        except OSError as exc:
            print(f"markcheck: cannot read {p}: {exc}", file=sys.stderr)
            had_error = True

    if args.json:
        print(json.dumps({"version": __version__, "files": [r.to_dict() for r in reports]}, indent=2))
    else:
        print(render_text(reports))

    if any(r.any_found for r in reports):
        return 1
    if had_error or any(r.any_error for r in reports) or not reports:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
