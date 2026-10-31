"""Adobe TrustMark, an open hidden mark, read with Adobe's `trustmark` library.

No network: the trustmark library downloads its model files the first time
it is used. markcheck never lets that happen. Before loading anything we
check that every file the decoder needs is already on disk and matches the
library's own checksum. If not, we skip and tell the person to run
`python -m markcheck.fetch_models` once.

Variant: "Q". It is the library's default (TrustMark(model_type='Q')) and
the library notes "most people use this as default". A mark made with a
different variant (B, C or P) may not read back with Q; pass
options["trustmark_variant"] to try another one.
"""
from __future__ import annotations

import io
import os
import pathlib
from hashlib import md5

from ..findings import Finding

NAME = "trustmark"
LABEL = "Adobe TrustMark hidden mark"
VARIANT = "Q"

_MODEL = None  # (variant, TrustMark) once loaded
_VERIFIED: set[str] = set()  # variants whose files passed the checksum this run
SCHEMAS = {0: "BCH_SUPER", 1: "BCH_5", 2: "BCH_4", 3: "BCH_3"}

FETCH_HINT = ("The TrustMark model files are not on this computer, and markcheck does not go online by itself. "
              "To fetch them once, run: python -m markcheck.fetch_models")


def _models_dir() -> pathlib.Path | None:
    import importlib.util

    spec = importlib.util.find_spec("trustmark")
    if spec is None or not spec.origin:
        return None
    return pathlib.Path(spec.origin).parent / "models"


def needed_files(variant: str) -> list[str]:
    # TrustMark.__init__ always loads the decoder and the encoder (we never load the remover).
    return [f"trustmark_{variant}.yaml", f"decoder_{variant}.ckpt", f"encoder_{variant}.ckpt"]


def missing_files(variant: str = VARIANT) -> list[str]:
    """Files that are absent or whose checksum does not match the library's list."""
    if variant in _VERIFIED:
        return []
    from trustmark.trustmark import MODEL_CHECKSUMS

    d = _models_dir()
    missing = []
    for name in needed_files(variant):
        p = d / name if d else None
        if p is None or not p.is_file() or p.stat().st_size == 0:
            missing.append(name)
            continue
        h = md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != MODEL_CHECKSUMS.get(name):
            missing.append(name)
    if not missing:
        _VERIFIED.add(variant)
    return missing


def available(variant: str = VARIANT) -> bool:
    try:
        import trustmark  # noqa: F401
    except Exception:
        return False
    return not missing_files(variant)


def get_model(variant: str = VARIANT):
    """Load once and keep it. Only call after missing_files() came back empty."""
    global _MODEL
    if _MODEL is None or _MODEL[0] != variant:
        if missing_files(variant):  # guard: never let the library download
            raise RuntimeError(FETCH_HINT)
        import contextlib

        from trustmark import TrustMark

        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            tm = TrustMark(verbose=False, model_type=variant, device="cpu", loadRemover=False)
        _MODEL = (variant, tm)
    return _MODEL[1]


def check(path, data, kind, options) -> list[Finding]:
    if kind != "image":
        return [Finding(NAME, LABEL, "skipped", "Only pictures can carry this mark.")]
    try:
        import trustmark  # noqa: F401
    except Exception:
        return [Finding(NAME, LABEL, "skipped",
                        "This check needs an extra library. Install it with: pip install 'markcheck[trustmark]'")]
    variant = (options or {}).get("trustmark_variant", VARIANT)
    if missing_files(variant):
        return [Finding(NAME, LABEL, "skipped", FETCH_HINT)]
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            rgb = im.convert("RGB")
    except Exception:
        return [Finding(NAME, LABEL, "skipped", "Could not open this file as a picture.")]
    if min(rgb.size) < 32:
        return [Finding(NAME, LABEL, "skipped", f"The picture is too small to check ({rgb.size[0]}x{rgb.size[1]}).")]

    import contextlib

    tm = get_model(variant)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        secret, present, version = tm.decode(rgb, MODE="binary")
    details = {"variant": f"TrustMark {variant}"}
    if not present:
        return [Finding(NAME, LABEL, "not_found", "No TrustMark mark found.", details=details)]

    details.update({
        "payload_bits": secret,
        "payload_hex": f"{int(secret, 2):x}" if secret else "",
        "schema": f"{SCHEMAS.get(version, 'unknown')} (version {version})",
    })
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            text, ok, _ = tm.decode(rgb, MODE="text")
        if ok and text and text.isprintable():
            details["payload_text"] = text
    except Exception:
        pass
    return [Finding(
        NAME, LABEL, "found",
        "This image carries an Adobe TrustMark hidden mark; its message passed the error check.",
        details=details,
        proves=["the pixels contain a TrustMark message, and the message read back without errors"],
        does_not_prove=[
            "who made the image: TrustMark is open, so anyone with the free library can add a mark",
            "that the image was made by AI; the message is usually just an ID that points to a signed "
            "label kept somewhere else, which this check does not look up",
            "anything when it is missing: many tools never add it",
        ],
    )]
