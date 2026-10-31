"""Stable Diffusion's open hidden mark (invisible-watermark, method "dwtDct").

What the official Stable Diffusion tools write into the pixels:

- SD v1 (CompVis/stable-diffusion scripts/txt2img.py): the text
  "StableDiffusionV1" as UTF-8 bytes, 17 bytes = 136 bits.
- SD v2 (Stability-AI/stablediffusion scripts/txt2img.py): the text
  "SDV2" as UTF-8 bytes, 4 bytes = 32 bits.
- SDXL (diffusers StableDiffusionXLWatermarker, copied from
  Stability-AI/generative-models): the 48-bit number 0xB3EC907BB19E,
  set with set_watermark("bits", ...), most significant bit first.

All three use WatermarkEncoder.encode(bgr_image, "dwtDct") with the
library's default strength (scales [0, 36, 36], block 4) on a BGR image.

How we read it: the decoder looks at 4x4 blocks of one colour channel and
reads one bit per block. Bit k of the message is repeated in every block
whose index i has i % message_length == k, and the decoder takes a vote.
We compute the per-block bits once (the same arithmetic as the library's
EmbedMaxDct.decode, written with numpy so big photos stay fast; a test
checks it gives the same bits as imwatermark's own WatermarkDecoder), then
vote for each of the three message lengths and count how many bits match.

A random image matches about half the bits by chance, so we need a high
bit-match rate before saying "found". The threshold below was picked from
measured data (scripts/measure.py and tests/test_watermark.py).
"""
from __future__ import annotations

import io

import numpy as np

from ..findings import Finding

NAME = "sd_watermark"
LABEL = "Stable Diffusion hidden mark"

# Same defaults as imwatermark.maxDct.EmbedMaxDct.
_SCALES = (0, 36, 36)
_BLOCK = 4

SDXL_MESSAGE = 0b101100111110110010010000011110111011000110011110  # == 0xB3EC907BB19E


def _bytes_to_bits(b: bytes) -> list[int]:
    return [int(x) for x in np.unpackbits(np.frombuffer(b, dtype=np.uint8))]


PAYLOADS: dict[str, list[int]] = {
    "SDXL": [int(c) for c in bin(SDXL_MESSAGE)[2:]],          # 48 bits
    "SD v1": _bytes_to_bits(b"StableDiffusionV1"),           # 136 bits
    "SD v2": _bytes_to_bits(b"SDV2"),                        # 32 bits
}

# How many bits must match before we say "found".
# A picture with no mark matches each bit about half the time, like a coin.
# We ask for so many matching bits that a coin-toss reading would reach it
# less than once in a million pictures, counting all six readings we make
# (three messages, two read modes). That gives 30 of 32, 42 of 48 and
# 98 of 136. Measured on our committed marked fixtures (checked by
# tests/test_watermark.py::test_sd_comment_numbers_match_measurements):
# 47-48 of 48 (SDXL), 31-32 of 32 (SD v2) and 120-121 of 136 (SD v1).
# Unmarked fixtures plus 80 generated unmarked pictures (drawn scenes and
# pure noise) stayed at or below 69% of bits for every message.
FALSE_ALARM_BUDGET = 1e-6
N_READINGS = 2 * len(PAYLOADS)


def needed_bits(n: int, budget: float = FALSE_ALARM_BUDGET / N_READINGS) -> int:
    from math import comb

    tail = 0.0
    for k in range(n, -1, -1):
        tail += comb(n, k) / 2 ** n
        if tail > budget:
            return k + 1
    return 0


NEEDED = {name: needed_bits(len(bits)) for name, bits in PAYLOADS.items()}

MIN_PIXELS = 256 * 256  # the library refuses smaller images


def block_bits(bgr: np.ndarray, float_mode: bool = False) -> list[np.ndarray]:
    """Per-block bit readings, in the decoder's block order, for each used channel.

    Mirrors imwatermark.maxDct.EmbedMaxDct.decode/infer_dct_matrix.

    float_mode: diffusers (SDXL) hands the encoder a float32 picture in 0..255,
    not uint8. OpenCV then centres the colour channels on 0.5 instead of 128,
    so the mark sits on a different grid. Reading a uint8 file the uint8 way
    gives a mix of right and flipped bits; reading it the float way recovers
    the message. We therefore read both ways.
    """
    import cv2
    import pywt

    row, col = bgr.shape[:2]
    src = bgr.astype(np.float32) if float_mode else bgr
    yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    out = []
    for channel in range(2):  # the library only looks at channels 0 and 1
        scale = _SCALES[channel]
        if scale <= 0:
            continue
        ca1, _ = pywt.dwt2(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar")
        r, c = ca1.shape
        nr, nc = r // _BLOCK, c // _BLOCK
        blocks = ca1[: nr * _BLOCK, : nc * _BLOCK].reshape(nr, _BLOCK, nc, _BLOCK).swapaxes(1, 2)
        flat = blocks.reshape(nr * nc, _BLOCK * _BLOCK)
        pos = np.argmax(np.abs(flat[:, 1:]), axis=1) + 1
        val = np.abs(flat[np.arange(flat.shape[0]), pos])
        out.append(((val % scale) > 0.5 * scale).astype(np.float64))
    return out


def vote(per_block: list[np.ndarray], length: int) -> np.ndarray:
    """Turn per-block readings into `length` message bits, as the library does."""
    sums = np.zeros(length)
    counts = np.zeros(length)
    for scores in per_block:
        idx = np.arange(scores.shape[0]) % length
        sums += np.bincount(idx, weights=scores, minlength=length)
        counts += np.bincount(idx, minlength=length)
    avg = np.divide(sums, counts, out=np.zeros(length), where=counts > 0)
    return avg * 255 > 127


def match_rates(bgr: np.ndarray) -> dict[str, tuple[int, int]]:
    """For each known payload: (bits that match, total bits), best of the two read modes."""
    readings = [block_bits(bgr, False), block_bits(bgr, True)]
    res = {}
    for name, bits in PAYLOADS.items():
        want = np.array(bits, dtype=bool)
        res[name] = (max(int(np.sum(vote(pb, len(bits)) == want)) for pb in readings), len(bits))
    return res


def load_bgr(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)
        rgb = np.array(im.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


DOES_NOT_PROVE = [
    "that AI made every part of the picture: it only shows the picture went through a Stable Diffusion "
    "tool that adds this mark, and it may have been edited since",
    "who made the image: the mark is public and the same for everyone, so anyone can add it to any picture",
    "anything when it is missing: many Stable Diffusion tools leave it off, and re-saving or resizing "
    "often wipes it",
]


def check(path, data, kind, options) -> list[Finding]:
    if kind != "image":
        return [Finding(NAME, LABEL, "skipped", "Only pictures can carry this mark.")]
    try:
        import cv2  # noqa: F401
        import pywt  # noqa: F401
    except ImportError:
        return [Finding(NAME, LABEL, "skipped",
                        "This check needs an extra library. Install it with: pip install 'markcheck[sd]'")]
    try:
        bgr = load_bgr(data)
    except Exception:
        return [Finding(NAME, LABEL, "skipped", "Could not open this file as a picture.")]
    h, w = bgr.shape[:2]
    if h * w < MIN_PIXELS:
        return [Finding(NAME, LABEL, "skipped",
                        f"The picture is too small to check ({w}x{h}); this mark needs at least 256x256 pixels.")]

    rates = match_rates(bgr)
    hits = [k for k, (m, n) in rates.items() if m >= NEEDED[k]]
    # report the strongest hit, else the reading closest to its bar
    best = max(hits or rates, key=lambda k: (rates[k][0] - NEEDED[k]) / rates[k][1])
    m, n = rates[best]
    details = {
        "best_match": f"{best} payload, {m} of {n} bits match ({m / n:.0%})",
        "bit_match": {k: f"{a}/{b} (needs {NEEDED[k]})" for k, (a, b) in rates.items()},
    }
    if hits:
        return [Finding(
            NAME, LABEL, "found",
            f"This image carries Stable Diffusion's open hidden mark ({best} payload, {m} of {n} bits match).",
            details=details,
            proves=["the pixels contain the fixed pattern that Stable Diffusion's own tools add"],
            does_not_prove=list(DOES_NOT_PROVE),
        )]
    return [Finding(NAME, LABEL, "not_found",
                    f"No Stable Diffusion mark found (closest: {best} payload, {m} of {n} bits match; "
                    f"{NEEDED[best]} needed).",
                    details=details)]
