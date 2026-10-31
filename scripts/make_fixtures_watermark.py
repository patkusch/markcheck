"""Make the hidden-mark test pictures in tests/fixtures/watermark/.

Every mark is added with the library's own encoder, the same way the real
tools do it:

- SD v1: CompVis/stable-diffusion scripts/txt2img.py put_watermark():
  wm = "StableDiffusionV1"; WatermarkEncoder().set_watermark('bytes', wm.encode('utf-8'));
  RGB -> BGR (cv2.cvtColor), encode(img, 'dwtDct'), BGR -> RGB (img[:, :, ::-1]).
- SD v2: Stability-AI/stablediffusion scripts/txt2img.py, same code with wm = "SDV2".
- SDXL: diffusers StableDiffusionXLWatermarker.apply_watermark(): 48 bits of
  0b101100111110110010010000011110111011000110011110 via set_watermark("bits", ...),
  applied to a float32 array in 0..255 (not uint8), RGB -> BGR by [:, :, ::-1],
  encode(image, "dwtDct"), back to RGB, clamp, then the pipeline rounds to uint8.
- TrustMark: trustmark.TrustMark(model_type='Q').encode(image, text). Needs the
  model files on disk (python -m markcheck.fetch_models); skipped otherwise.

The base pictures are drawn with Pillow and numpy from fixed seeds (three
kinds of scene: landscape, city, still life; sizes 256 to 1024) with soft
texture and grain, so they look like ordinary pictures rather than flat
colours. Every run makes the same pixels. Small marked pictures are PNG;
larger ones are high-quality JPEG (quality 95), as many generation tools
hand files out. Each unmarked base picture is saved too, as a control
(JPEG quality 85, to keep the folder small).

Note: a quality-95 JPEG keeps the Stable Diffusion mark only just. On our
pictures SD v2 read back at 30-32 of 32 bits (30 needed) and SDXL at 40-48
of 48 (42 needed), so some quality-95 SDXL saves lose it. The fixtures use
the seeds listed in SD_SET; none was re-rolled to make a mark pass.

    .venv/bin/python scripts/make_fixtures_watermark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "watermark"
sys.path.insert(0, str(ROOT / "src"))

from markcheck.detectors import trustmark_check  # noqa: E402

SDXL_MESSAGE = 0b101100111110110010010000011110111011000110011110
GRAIN = 4.0
CLOUD = 0.6
SDXL_BITS = [int(bit) for bit in bin(SDXL_MESSAGE)[2:]]


SCENES = ("landscape", "city", "still_life")


def _paint_landscape(d: ImageDraw.ImageDraw, rng, w: int, h: int) -> None:
    cx, cy, r = rng.integers(60, w - 60), rng.integers(40, h // 3), rng.integers(25, 50)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(v) for v in rng.integers([230, 190, 90], [256, 240, 160])))
    for layer in range(3):  # hills, back to front
        base = h * (0.45 + 0.17 * layer)
        xs = np.arange(0, w + 8, 8)
        phase, amp, freq = rng.uniform(0, 6.3), rng.uniform(15, 45), rng.uniform(1.5, 4)
        ys = base + amp * np.sin(xs / w * freq * np.pi + phase)
        col = rng.integers([30, 70, 20], [90, 140, 70]) + 25 * (2 - layer)
        d.polygon([(0, h)] + list(zip(xs.tolist(), ys.tolist())) + [(w, h)], fill=tuple(int(v) for v in col))
    for _ in range(rng.integers(4, 9)):  # trees / rocks
        x, yy = rng.integers(0, w), rng.integers(int(h * 0.6), h)
        s = rng.integers(8, 22)
        d.ellipse([x - s, yy - 2 * s, x + s, yy], fill=tuple(int(v) for v in rng.integers([10, 40, 10], [50, 90, 40])))


def _paint_city(d: ImageDraw.ImageDraw, rng, w: int, h: int) -> None:
    x = 0
    while x < w:  # buildings with lit windows
        bw, bh = int(rng.integers(w // 12, w // 5)), int(rng.integers(h // 4, int(h * 0.8)))
        col = tuple(int(v) for v in rng.integers([50, 50, 60], [120, 110, 130]))
        d.rectangle([x, h - bh, x + bw, h], fill=col)
        for wy in range(h - bh + 8, h - 10, 14):
            for wx in range(x + 5, x + bw - 8, 11):
                if rng.random() < 0.55:
                    d.rectangle([wx, wy, wx + 5, wy + 7], fill=tuple(int(v) for v in rng.integers([200, 180, 90], [256, 230, 160])))
        x += bw + int(rng.integers(0, 6))
    d.rectangle([0, int(h * 0.92), w, h], fill=(60, 60, 64))  # street


def _paint_still_life(d: ImageDraw.ImageDraw, rng, w: int, h: int) -> None:
    d.rectangle([0, int(h * 0.62), w, h], fill=tuple(int(v) for v in rng.integers([110, 70, 40], [170, 120, 80])))  # table
    for _ in range(rng.integers(3, 6)):  # fruit and cups
        cx, cy = int(rng.integers(w // 8, w - w // 8)), int(rng.integers(int(h * 0.5), int(h * 0.8)))
        r = int(rng.integers(w // 16, w // 7))
        col = tuple(int(v) for v in rng.integers([120, 20, 20], [250, 200, 90]))
        if rng.random() < 0.5:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        else:
            d.rectangle([cx - r, cy - int(1.4 * r), cx + r, cy + r // 2], fill=col)
        d.ellipse([cx - r // 3, cy - r // 2, cx, cy - r // 5], fill=(250, 250, 240))  # highlight


def base_image(seed: int, size=(512, 512), scene: str = "landscape") -> Image.Image:
    """A drawn picture with photo-like texture. Same seed, size and scene -> same pixels."""
    rng = np.random.default_rng(seed)
    w, h = size
    y = np.linspace(0, 1, h)[:, None, None]
    top = rng.uniform([40, 90, 160], [110, 160, 230])
    bottom = rng.uniform([170, 180, 200], [250, 230, 240])
    if scene == "still_life":  # indoor wall instead of sky
        top, bottom = rng.uniform([150, 130, 100], [210, 190, 160]), rng.uniform([90, 80, 70], [150, 130, 110])
    sky = (top * (1 - y) + bottom * y) * np.ones((1, w, 1))
    im = Image.fromarray(sky.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    {"landscape": _paint_landscape, "city": _paint_city, "still_life": _paint_still_life}[scene](d, rng, w, h)
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    # cloud-like colour texture (soft, mid-sized blobs) plus fine colour grain,
    # so the picture has the small-scale detail real photos and generated images have
    clouds = rng.normal(0, 1, (h // 8, w // 8, 3))
    clouds = np.array(Image.fromarray(np.clip(clouds * 40 + 128, 0, 255).astype(np.uint8))
                      .resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(3)), dtype=np.float64) - 128
    grain = rng.normal(0, GRAIN, (h, w, 3))
    arr = np.clip(np.asarray(im, dtype=np.float64) + CLOUD * clouds + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def sd_script_mark(img: Image.Image, text: str) -> Image.Image:
    """Exactly put_watermark() from the SD v1 / v2 txt2img scripts."""
    import cv2
    from imwatermark import WatermarkEncoder

    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark("bytes", text.encode("utf-8"))
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    arr = wm_encoder.encode(arr, "dwtDct")
    return Image.fromarray(arr[:, :, ::-1])


def sdxl_mark(img: Image.Image) -> Image.Image:
    """diffusers StableDiffusionXLWatermarker.apply_watermark on one image, then the pipeline's uint8 step."""
    from imwatermark import WatermarkEncoder

    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", SDXL_BITS)
    image = np.asarray(img, dtype=np.float32)            # 255 * (x / 2 + 0.5), x in [-1, 1]
    image = image[:, :, ::-1]                            # RGB -> BGR
    image = encoder.encode(image, "dwtDct")[:, :, ::-1]  # encode, BGR -> RGB
    x = np.clip(2 * (image / 255 - 0.5), -1.0, 1.0)      # torch.clamp(...)
    x = (x / 2 + 0.5).clip(0, 1)                         # image_processor.denormalize
    return Image.fromarray((x * 255).round().astype(np.uint8))  # numpy_to_pil


def trustmark_mark(img: Image.Image, text: str) -> Image.Image:
    tm = trustmark_check.get_model()
    return tm.encode(img, text, MODE="text")


# (file name, seed, size, scene, mark, payload text for TrustMark)
# Large pictures are saved as high-quality JPEG (quality 95), the way many
# generation tools and web front-ends hand files out; small ones as PNG.
SD_SET = [
    ("sd_v1_marked.png", 1, (512, 512), "landscape", "SD v1"),
    ("sd_v2_marked.png", 2, (512, 512), "landscape", "SD v2"),
    ("sdxl_marked.png", 3, (512, 512), "landscape", "SDXL"),
    ("sd_v1_city_768x512.jpg", 11, (768, 512), "city", "SD v1"),
    ("sd_v2_still_life_768x768.jpg", 12, (768, 768), "still_life", "SD v2"),
    ("sdxl_city_1024x1024.jpg", 13, (1024, 1024), "city", "SDXL"),
    ("sdxl_still_life_896x640.jpg", 14, (896, 640), "still_life", "SDXL"),
]
TM_SET = [
    ("trustmark_q_marked.png", 4, (512, 512), "landscape", "markchk"),
    ("trustmark_city_256x256.png", 21, (256, 256), "city", "mc-0002"),
    ("trustmark_still_life_384x384.png", 22, (384, 384), "still_life", "mc-0003"),
    ("trustmark_landscape_1024x768.jpg", 23, (1024, 768), "landscape", "mc-0004"),
    ("trustmark_city_640x480.jpg", 24, (640, 480), "city", "mc-0005"),
]
SD_NOTES = {
    "SD v1": "SD v1 text 'StableDiffusionV1' added with invisible-watermark exactly as CompVis txt2img.py does",
    "SD v2": "SD v2 text 'SDV2' added with invisible-watermark exactly as Stability-AI txt2img.py does",
    "SDXL": "SDXL 48-bit payload 0xB3EC907BB19E added exactly as diffusers StableDiffusionXLWatermarker does",
}


def mark_sd(img: Image.Image, kind: str) -> Image.Image:
    if kind == "SDXL":
        return sdxl_mark(img)
    return sd_script_mark(img, "StableDiffusionV1" if kind == "SD v1" else "SDV2")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        if old.suffix in (".png", ".jpg"):
            old.unlink()
    files = []

    def save(im: Image.Image, name: str, expect: list[str], note: str, quality: int = 95, **extra):
        if name.endswith(".jpg"):
            im.save(OUT / name, "JPEG", quality=quality)
            note += f" (saved as a JPEG at quality {quality})"
        else:
            im.save(OUT / name, optimize=True)
        files.append({"file": name, "expect": expect, "note": note, **extra})

    def unmarked_name(name: str) -> str:
        # controls are saved as smaller JPEGs (quality 85) to keep the folder small
        return "unmarked_" + name.replace("_marked", "").rsplit(".", 1)[0] + ".jpg"

    for name, seed, size, scene, kind in SD_SET:
        base = base_image(seed, size, scene)
        save(mark_sd(base, kind), name, ["sd_watermark"], f"{SD_NOTES[kind]}; {scene}, {size[0]}x{size[1]}",
             payload=kind)
        save(base, unmarked_name(name), [], f"the same {scene} picture before marking; no mark", quality=85)
    have_tm = trustmark_check.available()
    if not have_tm:
        print("TrustMark model files not on disk; skipping the TrustMark fixtures. "
              "Run: python -m markcheck.fetch_models")
    for name, seed, size, scene, text in TM_SET:
        base = base_image(seed, size, scene)
        if have_tm:
            save(trustmark_mark(base, text), name, ["trustmark"],
                 f"TrustMark variant Q, text '{text}' (BCH_5 schema), added with trustmark's own encoder; "
                 f"{scene}, {size[0]}x{size[1]}", payload=text)
        save(base, unmarked_name(name), [], f"the same {scene} picture before marking; no mark", quality=85)

    (OUT / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    total = sum((OUT / f["file"]).stat().st_size for f in files)
    print(f"Wrote {len(files)} files to {OUT} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
