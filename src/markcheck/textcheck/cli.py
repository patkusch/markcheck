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


def _plain_os_error(exc: OSError) -> str:
    from ..cli import plain_os_error
    return plain_os_error(exc)


def read_input(args, stdin=None) -> tuple[str | None, str | None]:
    """Return (text, problem). Exactly one of them is None."""
    stdin = sys.stdin if stdin is None else stdin
    if args.file and args.text is not None:
        return None, "give the text or --file, not both."
    if args.file:
        try:
            raw = Path(args.file).read_bytes()
        except OSError as exc:
            return None, f"could not read {args.file}: {_plain_os_error(exc)}."
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None, f"could not read {args.file}: it is not plain text saved as UTF-8."
    elif args.text == "-" or (args.text is None and stdin is not None and not stdin.isatty()):
        if stdin is None:
            return None, "no text was given."
        text = stdin.read()
    elif args.text is None:
        return None, "no text was given. Pass the text, use --file, or pipe it in with '-'."
    else:
        text = args.text
    if not text.strip():
        return None, "the text is empty."
    return text, None


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="markcheck text",
        description="Estimate the chance that a text was AI-written, with reasons and how often "
                    "this check is wrong. It is an estimate, not a finding about a person.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("text", nargs="?", help="the text to check, or '-' to read it from standard input")
    ap.add_argument("--file", help="read the text from this file (plain text, UTF-8)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    args = ap.parse_args(argv)

    text, problem = read_input(args)
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
