"""Detector registry.

Each detector module exposes:

    NAME: str                      # machine name
    LABEL: str                     # plain name shown to people
    def check(path, data, kind) -> list[Finding]

`data` is the whole file as bytes. `kind` comes from filetypes.sniff.
A detector must never raise for an ordinary unlabelled file: it returns
a "not_found" or "skipped" Finding. Unexpected exceptions are caught
here and turned into an "error" Finding.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from ..findings import Finding

# Order matters: this is the order results are printed.
MODULES = ["c2pa_check", "metadata", "sd_watermark", "trustmark_check"]


def run_all(path: Path, data: bytes, kind: str, options: dict | None = None) -> list[Finding]:
    options = options or {}
    out: list[Finding] = []
    for modname in MODULES:
        try:
            mod = importlib.import_module(f"{__name__}.{modname}")
        except Exception as exc:  # a broken optional detector must not stop the others
            out.append(Finding(modname, modname, "error", f"Could not load this check: {exc}"))
            continue
        try:
            out.extend(mod.check(path, data, kind, options))
        except Exception as exc:
            out.append(Finding(getattr(mod, "NAME", modname), getattr(mod, "LABEL", modname), "error",
                               f"This check failed on this file: {type(exc).__name__}: {exc}"))
    return out
