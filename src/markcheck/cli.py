"""Command line entry point.

Exit codes:
  0  no label found in any file that was checked
  1  a label was found in at least one file (even if other files had problems)
  2  error: nothing could be checked, or some files or checks failed and no label was found

markcheck never goes online.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .detectors import run_all
from .filetypes import sniff
from .findings import FileReport
from .makers import makers_for
from .report import render_text, summary_counts

# Folders that hold tools, not people's files. Hidden folders (".name") are skipped too.
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}

# Kinds that can carry a label anywhere in the file, so we read them whole.
MEDIA_KINDS = {"image", "video", "audio", "pdf"}
# Other files are read only up to this size: none of our checks needs more of them.
MAX_NON_MEDIA_BYTES = 16 * 1024 * 1024

EPILOG = """\
exit codes:
  0  no label found (this does not prove a file was made by a person)
  1  a label was found in at least one file
  2  error: nothing could be checked, or something failed and no label was found

markcheck runs on your computer and never goes online.
Hidden files and folders, and tool folders such as .git and .venv, are skipped.
Files that are not pictures, video, sound or PDF are read only up to 16 MB.
"""


def plain_os_error(exc: OSError) -> str:
    if isinstance(exc, PermissionError):
        return "markcheck is not allowed to open it (permission denied)"
    if isinstance(exc, FileNotFoundError):
        return "it is missing (it may have been moved or deleted)"
    if isinstance(exc, IsADirectoryError):
        return "it is a folder, not a file"
    return f"it could not be read ({exc.strerror or exc})"


def iter_files(target: Path, problems: list[dict] | None = None):
    """Yield the files to check under target, in a stable order.

    Skips hidden files and folders and tool folders below target (not above it),
    never follows links to folders (so link loops cannot trap us), and records
    unreadable folders and broken links in `problems`.
    """
    problems = problems if problems is not None else []
    if target.is_file():
        yield target
        return

    def onerror(exc: OSError):
        problems.append({"path": str(exc.filename or target), "problem": plain_os_error(exc)})

    for root, dirs, files in os.walk(target, onerror=onerror, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
        for name in sorted(files):
            if name.startswith("."):
                continue
            p = Path(root) / name
            if p.is_symlink() and not p.exists():
                problems.append({"path": str(p), "problem": "it is a link to a file that is missing"})
                continue
            if not p.is_file():  # pipes, sockets, devices
                continue
            yield p


def check_file(path: Path, options: dict | None = None) -> FileReport:
    path = Path(path)
    notes: list[str] = []
    with open(path, "rb") as fh:
        head = fh.read(64)
        kind = sniff(path, head)
        if kind in MEDIA_KINDS:
            data = head + fh.read()
        else:
            data = head + fh.read(max(0, MAX_NON_MEDIA_BYTES - len(head)))
            if fh.read(1):
                notes.append(f"This file is large and is not a picture, video, sound or PDF file. "
                             f"Only its first {MAX_NON_MEDIA_BYTES // (1024 * 1024)} MB were read.")
    rep = FileReport(path=str(path), kind=kind, notes=notes)
    rep.findings = run_all(path, data, kind, options or {})
    rep.makers = makers_for(kind, rep.findings)
    return rep


def exit_code(reports: list[FileReport], problems: list[dict]) -> int:
    if any(r.any_found for r in reports):
        return 1
    if problems or not reports or any(r.any_error for r in reports):
        return 2
    return 0


def _target_problem(target: Path) -> str | None:
    if target.is_symlink() and not target.exists():
        return f"{target} is a link, but the file or folder it points to is missing."
    if not target.exists():
        return f"cannot find {target}."
    if not (target.is_file() or target.is_dir()):
        return f"{target} is not a file or a folder, so it was not checked."
    return None


def to_json(reports: list[FileReport], problems: list[dict], code: int) -> dict:
    return {
        "version": __version__,
        "exit_code": code,
        "summary": summary_counts(reports, problems),
        "files": [r.to_dict() for r in reports],
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="markcheck",
        description="Look for AI labels in a file or folder and explain what they do and don't prove. "
                    "Runs on your machine; nothing is uploaded.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("target", help="a file or a folder")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    ap.add_argument("--version", action="version", version=f"markcheck {__version__}")
    args = ap.parse_args(argv)

    target = Path(args.target)
    reports: list[FileReport] = []
    problems: list[dict] = []

    bad_target = _target_problem(target)
    if bad_target:
        print(f"markcheck: {bad_target}", file=sys.stderr)
        problems.append({"path": str(target), "problem": bad_target})
    else:
        for p in iter_files(target, problems):
            try:
                reports.append(check_file(p))
            except OSError as exc:
                problems.append({"path": str(p), "problem": plain_os_error(exc)})
        for pr in problems:
            print(f"markcheck: could not check {pr['path']}: {pr['problem']}", file=sys.stderr)
        if not reports and not problems:
            print(f"markcheck: no files to check in {target}. Hidden files and tool folders "
                  f"such as .git are skipped.", file=sys.stderr)

    code = exit_code(reports, problems)
    if args.json:
        print(json.dumps(to_json(reports, problems, code), indent=2))
    elif not bad_target:
        print(render_text(reports, problems))
    return code


if __name__ == "__main__":
    sys.exit(main())
