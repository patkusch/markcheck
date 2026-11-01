"""`markcheck text`: estimate the chance that a text was AI-written.

    markcheck text "some text"
    markcheck text --file essay.txt
    cat essay.txt | markcheck text -

Exit codes:
  0  an estimate was handled (scored, or refused because the text is too short)
  2  error: no text, an unreadable file, or the language models are missing

This runs on your computer and never goes online.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import __version__
from . import CAVEAT, TextResult, check_text
from .scorer import MissingModels

EPILOG = """\
exit codes:
  0  an estimate was handled (a number, or "too short to estimate")
  2  error: no text, a file that could not be read, or the language models are missing

The text check needs extra libraries and two small language models:
  pip install 'markcheck[text]'
  python -m markcheck.fetch_models --text
After that it runs on your computer and never goes online.
"""

KIND = "text_estimate"
# Read at most this much. Only the first 512 word pieces are scored anyway.
MAX_BYTES = 1_000_000
CUT_NOTE = "The input was longer than 1 MB; only its first 1 MB was read."


def _plain_os_error(exc: OSError) -> str:
    from ..cli import plain_os_error
    return plain_os_error(exc)


def _decode(raw: bytes, cut: bool) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if cut and exc.start >= len(raw) - 3:  # the 1 MB cut fell inside one character
            return raw[:exc.start].decode("utf-8-sig")
        raise


def read_input(args, stdin=None) -> tuple[str | None, str | None, str | None]:
    """Return (text, problem, note). Exactly one of text and problem is None; note says if the input was cut."""
    stdin = sys.stdin if stdin is None else stdin
    note = None
    if args.file and args.text is not None:
        return None, "give the text or --file, not both.", None
    if args.file:
        try:
            with open(args.file, "rb") as fh:
                raw = fh.read(MAX_BYTES + 1)
        except OSError as exc:
            return None, f"could not read {args.file}: {_plain_os_error(exc)}.", None
        cut = len(raw) > MAX_BYTES
        try:
            text = _decode(raw[:MAX_BYTES], cut)
        except UnicodeDecodeError:
            text = None
        if text is None or "\0" in text:  # NUL bytes mean a binary file, even if it decodes
            return None, f"could not read {args.file}: it is not plain text saved as UTF-8.", None
        note = CUT_NOTE if cut else None
    elif args.text == "-" or (args.text is None and stdin is not None and not stdin.isatty()):
        if stdin is None:
            return None, "no text was given.", None
        text = stdin.read(MAX_BYTES + 1)
        if len(text) > MAX_BYTES:
            text, note = text[:MAX_BYTES], CUT_NOTE
    elif args.text is None:
        return None, "no text was given. Pass the text, use --file, or pipe it in with '-'.", None
    else:
        text = args.text
    if not text.strip():
        return None, "the text is empty.", None
    return text, None, note


def _rate_lines(rates: dict, indent: str = "") -> list[str]:
    out = []
    for key, value in rates.items():
        if isinstance(value, bool):  # flags such as "placeholder" are for tools, not readers
            continue
        if isinstance(value, dict):
            sub = _rate_lines(value, indent + "  ")
            if sub:
                out.append(f"{indent}- {key}:")
                out += sub
        elif isinstance(value, (str, int, float)):
            out.append(f"{indent}- {key}: {value}")
    return out


def render_text(res: TextResult) -> str:
    lines = [res.headline]
    if res.status == "scored":
        if res.reasons:
            lines += ["", "Why:"]
            lines += [f"- {r}" for r in res.reasons]
        if res.sentences:
            lines += ["", "Sentences that moved the estimate most:"]
            for s in res.sentences:
                lines.append(f"- \"{s['text']}\"")
                lines.append(f"  {s['effect'][:1].upper()}{s['effect'][1:]}: {s['why']}.")
        lines += ["", "How often this check is right and wrong (held-out tests):"]
        lines += _rate_lines(res.error_rates) or ["- Not measured yet."]
        if res.notes:
            lines += [""]
            lines += [f"Note: {n}" for n in res.notes]
    lines += ["", CAVEAT]
    return "\n".join(lines)


def to_json(res: TextResult) -> dict:
    return {"version": __version__, "kind": KIND, **res.to_dict()}


def error_json(message: str) -> dict:
    return {"version": __version__, "kind": KIND, "status": "error", "error": message,
            "probability_ai": None, "caveat": CAVEAT}


class _ArgError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """Argument mistakes still give JSON when --json was asked for."""
    want_json = False

    def error(self, message: str):
        if self.want_json:
            raise _ArgError(message)
        super().error(message)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = _Parser(
        prog="markcheck text",
        description="Estimate the chance that a text was AI-written, with reasons and how often "
                    "this check is wrong. It is an estimate, not a finding about a person.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("text", nargs="?", help="the text to check, or '-' to read it from standard input")
    ap.add_argument("--file", help="read the text from this file (plain text, UTF-8)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    ap.want_json = "--json" in argv
    try:
        args = ap.parse_args(argv)
    except _ArgError as exc:
        return _fail(str(exc), True)

    text, problem, note = read_input(args)
    if problem:
        return _fail(problem, args.json)
    try:
        res = check_text(text)
    except MissingModels as exc:
        return _fail(str(exc), args.json)
    except (ImportError, OSError):
        return _fail("The language models for the text check could not be loaded. Install the extra "
                     "libraries with: pip install 'markcheck[text]', then fetch the models once with: "
                     "python -m markcheck.fetch_models --text", args.json)
    except Exception as exc:  # noqa: BLE001 - one plain line (and JSON) instead of a traceback
        return _fail(f"the text could not be scored ({type(exc).__name__}: {exc}).", args.json)
    if note:
        res.notes.insert(0, note)
    if args.json:
        print(json.dumps(to_json(res), indent=2))
    else:
        print(render_text(res))
    return 0


def _fail(message: str, as_json: bool) -> int:
    print(f"markcheck text: {message}", file=sys.stderr)
    if as_json:
        print(json.dumps(error_json(message), indent=2))
    return 2


if __name__ == "__main__":
    sys.exit(main())
