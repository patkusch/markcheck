"""Adobe TrustMark, an open hidden mark, read with Adobe's `trustmark` library.

No network: the trustmark library downloads its model files the first time
it is used. markcheck never lets that happen. Before loading anything we
check that every file the decoder needs is already on disk and matches the
library's own checksum. If not, we skip and tell the person to run
`python -m markcheck.fetch_models` once.

How sure is "found"? The library calls a mark present whenever the 96
message bits it reads pass the error check of the schema named by the last
two version bits. For the weaker schemas that check is easy to pass by
luck: in our measurements (round 2, scripts/measure.py and
tests/test_watermark.py) about 1 in 25 ordinary unmarked pictures "passed",
nearly all as BCH_3, which fixes up to 3 wrong bits with only 21 check bits.
So, like the Stable Diffusion check, we only say "found" when a reading
that good would come from a random picture less than once in a million
times. We count how many bits the error check had to fix (by rebuilding
the clean message from the decoded data and comparing), require the two
unused version bits to be zero as the encoder writes them, and work out the
chance. In practice: BCH_SUPER with any number it can fix (8), BCH_5 up to 2,
BCH_4 up to 1, BCH_3 only with none.

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
# Check bits per schema (96 message bits = data + check bits; 4 version bits follow).
CHECK_BITS = {0: 56, 1: 35, 2: 28, 3: 21}
CODE_BITS = 96
# Same budget as the Stable Diffusion check: under one in a million unmarked
# pictures may reach "found", split evenly over the four schemas.
FALSE_ALARM_BUDGET = 1e-6


def chance_by_luck(version: int, fixed: int) -> float:
    """Chance a random reading lands on this schema, has zero unused version bits,
    and is within `fixed` wrong bits of a valid message."""
    from math import comb

    near = sum(comb(CODE_BITS, i) for i in range(fixed + 1))
    return (1 / 4) * (1 / 4) * near / 2 ** CHECK_BITS[version]


def max_fixed_bits(version: int, budget: float = FALSE_ALARM_BUDGET / 4) -> int:
    """Most fixed bits we accept for a schema; -1 means never."""
    k = -1
    while k + 1 <= CODE_BITS and chance_by_luck(version, k + 1) < budget:
        k += 1
    return k

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


def read(tm, rgb) -> dict:
    """Read the 100 raw bits the way TrustMark.decode() does (no rotation, no
    box search), then run the library's own error check on them.

    Returns what the library decided plus the facts we need to judge it:
    which schema, how many bits the check had to fix, and whether the two
    unused version bits are zero. A test checks this gives the same answer
    as tm.decode() on our files.
    """
    import contextlib

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from trustmark.datalayer import DataLayer

    crop = tm.get_the_image_for_processing(rgb)
    crop = crop.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
    x = transforms.ToTensor()(crop).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
    with torch.no_grad():
        bits = (tm.decoder.decoder(x) > 0).cpu().numpy()
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        secret, ok, version = tm.ecc.decode_bitstream(bits, "binary")[0]
        text, text_ok, _ = tm.ecc.decode_bitstream(bits, "text")[0]
    out = {"passed_library_check": bool(ok), "version": version, "secret": secret,
           "text": text if text_ok else "", "bits_fixed": None, "unused_bits_zero": None}
    if ok and version in CHECK_BITS:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            clean = DataLayer(len(bits[0]), verbose=False, encoding_mode=version).process_encode(secret)
        raw = bits[0].astype(np.float32)
        out["bits_fixed"] = int(np.sum(clean[:CODE_BITS] != raw[:CODE_BITS]))
        out["unused_bits_zero"] = not raw[CODE_BITS:CODE_BITS + 2].any()
    return out


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

    tm = get_model(variant)
    try:
        r = read(tm, rgb)
    except Exception as exc:  # library internals changed: say so rather than guess
        return [Finding(NAME, LABEL, "error", f"The TrustMark library could not read this picture ({exc}).")]
    details = {"variant": f"TrustMark {variant}"}
    if not r["passed_library_check"]:
        return [Finding(NAME, LABEL, "not_found", f"No TrustMark mark found (variant {variant} checked; marks made with "
                        f"{', '.join(v for v in ('B', 'C', 'P', 'Q') if v != variant)} are not read).",
                        details=details)]

    version, fixed = r["version"], r["bits_fixed"]
    chance = chance_by_luck(version, fixed) if r["unused_bits_zero"] and version in CHECK_BITS else 1.0
    why = (f"{fixed} bits fixed by the error check; chance of this by luck about {chance:.0e}"
           if r["unused_bits_zero"] else "the unused version bits are not zero, which the encoder never writes")
    details["schema"] = f"{SCHEMAS.get(version, 'unknown')} (version {version}); {why}"
    if chance >= FALSE_ALARM_BUDGET / 4:
        return [Finding(NAME, LABEL, "not_found",
                        "No reliable TrustMark mark found. The reading passed only a weak error check that "
                        "pictures with no mark also pass now and then, so it is not counted.",
                        details=details)]
    secret = r["secret"]
    details.update({
        "payload_bits": secret,
        "payload_hex": f"{int(secret, 2):x}" if secret else "",
    })
    if r["text"] and r["text"].isprintable():
        details["payload_text"] = r["text"]
    if fixed == 0:
        how = "the message read back without errors"
    else:
        how = (f"the message read back after its error check fixed {fixed} bits, which a picture with "
               f"no mark would match by chance less than once in a million times")
    return [Finding(
        NAME, LABEL, "found",
        "This image carries an Adobe TrustMark hidden mark; its message passed the error check"
        + ("." if fixed == 0 else f" ({fixed} bits fixed)."),
        details=details,
        proves=[f"the pixels contain a TrustMark message, and {how}"],
        does_not_prove=[
            "who made the image: TrustMark is open, so anyone with the free library can add a mark",
            "that the image was made by AI; the message is usually just an ID that points to a signed "
            "label kept somewhere else, which this check does not look up",
            "anything when it is missing: many tools never add it",
        ],
    )]
