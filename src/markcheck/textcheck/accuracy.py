"""The README's "How accurate it is" numbers, made from the shipped calibration file.

    python -m markcheck.textcheck.accuracy            # print the block
    python -m markcheck.textcheck.accuracy --write-readme

Every number in that README block comes from src/markcheck/data/text_calibration.json,
so the README cannot drift from the stored results. tests/test_text_numbers.py checks
the README holds exactly this block.

Keys read (all but the first are optional, and missing ones are left out):
  heldout.how_often_wrong   {label: "N in 100"}; shown verbatim and in order, as the CLI does
  uncertain_band            [low, high], chances between 0 and 1
  min_words                 texts shorter than this get no number
  fairness.groups           {name: {"status", "flagged": RATE, "cant_tell": RATE}}
  fairness.pooled           {"native" | "non-native": same shape as a group}
  fairness.band_tradeoff    [{"band_top", "non_native_flagged", "native_flagged", "ai_caught": {set: RATE}}]
  fairness.computed_with    the band and weights the fairness numbers were worked out with
A RATE is {"k", "n", "rate", "ci95": [low, high]}, written by scripts/text_fairness.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from . import _pcts, load_calibration

START = "<!-- text-accuracy:start -->"
END = "<!-- text-accuracy:end -->"


def _in_100(x: float) -> str:
    if x == 0:
        return "0"
    if x < 0.01:
        return f"{100 * x:.1f}"
    return str(round(100 * x))


def rate_words(r: dict | None) -> str:
    """'3 in 100 (likely 1 to 11)' from a RATE; '-' when there is nothing to show."""
    if not isinstance(r, dict) or not r.get("n") or r.get("rate") is None:
        return "-"
    out = f"{_in_100(r['rate'])} in 100"
    ci = r.get("ci95")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        out += f" (likely {_in_100(ci[0])} to {_in_100(ci[1])})"
    return out


def how_often_rows(cal: dict) -> list[tuple[str, str]]:
    """The same lines the CLI prints, flattened: nested groups become 'group: label'."""
    def walk(d: dict, prefix: str = "") -> list[tuple[str, str]]:
        out = []
        for k, v in d.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, dict):
                out += walk(v, f"{prefix}{k}: ")
            elif isinstance(v, (str, int, float)):
                out.append((f"{prefix}{k}", str(v)))
        return out
    return walk(cal.get("heldout", {}).get("how_often_wrong", {}) or {})


def _cell(s: str) -> str:
    return s.replace("|", "/")


def render(cal: dict | None = None) -> str:
    cal = cal or load_calibration()
    lines = [START]
    lo, hi = cal["uncertain_band"]
    _, slo, shi = _pcts(0.5, lo, hi)
    lines += [f"Measured on texts kept aside, never used for fitting. "
              f"The \"can't tell\" band is {slo}-{shi}. "
              f"Texts under {cal['min_words']} words get no number. "
              f"Full numbers are in `docs/rounds.md`.", "",
              "| What we measured | How often |", "|---|---|"]
    rows = how_often_rows(cal)
    lines += [f"| {_cell(k)} | {_cell(v)} |" for k, v in rows] or ["| Not measured yet. | - |"]

    fair = cal.get("fairness") or {}
    groups = {**(fair.get("groups") or {}),
              **{f"All {k} writers together": v for k, v in (fair.get("pooled") or {}).items()}}
    if groups:
        lines += ["", "**Fairness.** People's writing only, never used for fitting, 100 words or more. "
                  "\"Likely\" is the range the rate is likely to be in (95% confidence); "
                  "it is wide when we had few texts.", "",
                  "| Writers | Texts | Put above the band | Put in the \"can't tell\" band |",
                  "|---|---|---|---|"]
        for name, g in groups.items():
            if not isinstance(g, dict):
                continue
            n = (g.get("flagged") or {}).get("n", "-")
            lines.append(f"| {_cell(name)} | {n} | {rate_words(g.get('flagged'))} | "
                         f"{rate_words(g.get('cant_tell'))} |")
    trade = [t for t in (fair.get("band_tradeoff") or []) if isinstance(t, dict) and "band_top" in t]
    if trade:
        sets = sorted({k for t in trade for k in (t.get("ai_caught") or {})})
        lines += ["", "**If the top of the band were higher.** Fewer people would be put above it, "
                  "and less AI text would be caught:", "",
                  "| Top of band | Non-native writers put above | Native writers put above | "
                  + " | ".join(f"AI caught: {_cell(s)}" for s in sets) + " |",
                  "|---" * (3 + len(sets)) + "|"]
        for t in trade:
            caught = t.get("ai_caught") or {}
            lines.append(f"| {_pcts(t['band_top'])[0]} | {rate_words(t.get('non_native_flagged'))} | "
                         f"{rate_words(t.get('native_flagged'))} | "
                         + " | ".join(rate_words(caught.get(s)) for s in sets) + " |")
    lines.append(END)
    return "\n".join(lines)


def readme_path() -> Path:
    return Path(__file__).resolve().parents[3] / "README.md"


def replace_block(readme: str, block: str) -> str:
    pat = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pat.search(readme):
        raise ValueError("README has no text-accuracy block markers")
    return pat.sub(lambda _: block, readme, count=1)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    block = render()
    if "--write-readme" in argv:
        path = readme_path()
        path.write_text(replace_block(path.read_text(encoding="utf-8"), block), encoding="utf-8")
        print(f"updated {path}")
    else:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
