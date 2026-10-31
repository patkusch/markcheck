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

The base pictures are drawn with Pillow (sky, hills, sun, soft texture) so they
look like ordinary pictures rather than flat colours. Unmarked versions are
saved too, as controls. All files are PNG so nothing is lost on save.

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

SDXL_MESSAGE = 0b101100111110110010010000011110111011000110011110
GRAIN = 4.0
CLOUD = 0.6
SDXL_BITS = [int(bit) for bit in bin(SDXL_MESSAGE)[2:]]


def base_image(seed: int, size=(512, 512)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    y = np.linspace(0, 1, h)[:, None, None]
    top = rng.uniform([40, 90, 160], [110, 160, 230])
    bottom = rng.uniform([170, 180, 200], [250, 230, 240])
    sky = (top * (1 - y) + bottom * y) * np.ones((1, w, 1))
    im = Image.fromarray(sky.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    # sun
    cx, cy, r = rng.integers(60, w - 60), rng.integers(40, h // 3), rng.integers(25, 50)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(v) for v in rng.integers([230, 190, 90], [256, 240, 160])))
    # hills, back to front
    for layer in range(3):
        base = h * (0.45 + 0.17 * layer)
        xs = np.arange(0, w + 8, 8)
        phase, amp, freq = rng.uniform(0, 6.3), rng.uniform(15, 45), rng.uniform(1.5, 4)
        ys = base + amp * np.sin(xs / w * freq * np.pi + phase)
        col = rng.integers([30, 70, 20], [90, 140, 70]) + 25 * (2 - layer)
        d.polygon([(0, h)] + list(zip(xs.tolist(), ys.tolist())) + [(w, h)], fill=tuple(int(v) for v in col))
    # a few trees / rocks
    for _ in range(rng.integers(4, 9)):
        x, yy = rng.integers(0, w), rng.integers(int(h * 0.6), h)
        s = rng.integers(8, 22)
        d.ellipse([x - s, yy - 2 * s, x + s, yy], fill=tuple(int(v) for v in rng.integers([10, 40, 10], [50, 90, 40])))
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


def trustmark_mark(img: Image.Image, text: str) -> Image.Image | None:
    from markcheck.detectors import trustmark_check

    if not trustmark_check.available():
        print("TrustMark model files not on disk; skipping the TrustMark fixture. "
              "Run: python -m markcheck.fetch_models")
        return None
    tm = trustmark_check.get_model()
    return tm.encode(img, text, MODE="text")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = []

    def save(im: Image.Image, name: str, expect: list[str], note: str):
        im.save(OUT / name, optimize=True)
        files.append({"file": name, "expect": expect, "note": note})

    save(sd_script_mark(base_image(1), "StableDiffusionV1"), "sd_v1_marked.png", ["sd_watermark"],
         "SD v1 text 'StableDiffusionV1' added with invisible-watermark exactly as CompVis txt2img.py does")
    save(sd_script_mark(base_image(2), "SDV2"), "sd_v2_marked.png", ["sd_watermark"],
         "SD v2 text 'SDV2' added with invisible-watermark exactly as Stability-AI txt2img.py does")
    save(sdxl_mark(base_image(3)), "sdxl_marked.png", ["sd_watermark"],
         "SDXL 48-bit payload 0xB3EC907BB19E added exactly as diffusers StableDiffusionXLWatermarker does")
    tm = trustmark_mark(base_image(4), "markchk")
    if tm is not None:
        save(tm, "trustmark_q_marked.png", ["trustmark"],
             "TrustMark variant Q, text 'markchk' (BCH_5 schema), added with trustmark's own encoder")
    for seed in (1, 5):
        save(base_image(seed), f"unmarked_{seed}.png", [],
             "drawn picture with no mark; same kind of picture as the marked ones")

    (OUT / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    total = sum((OUT / f["file"]).stat().st_size for f in files)
    print(f"Wrote {len(files)} files to {OUT} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
