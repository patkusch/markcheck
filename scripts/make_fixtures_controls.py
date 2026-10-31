"""Make varied unlabelled control files in tests/fixtures/controls_varied/.

Controls carry no label of any kind. Every check must say "not found" on
every one of them; scripts/measure.py counts any "found" as a false alarm.

The set covers the kinds of picture people really check: photo-like
textures at several sizes and JPEG qualities, flat graphics, a chart,
screenshot-like pictures with text, grayscale, a picture with see-through
parts, tiny pictures, a GIF, a very wide and a very tall picture, pure
noise, and two files whose ordinary text mentions AI tools.

Everything is drawn from fixed seeds, so every run makes the same pixels.

    .venv/bin/python scripts/make_fixtures_controls.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "controls_varied"


def photo_like(seed: int, size: tuple[int, int]) -> Image.Image:
    """Soft 1/f noise in colour: the uneven, blotchy texture of real photos."""
    rng = np.random.default_rng(seed)
    w, h = size
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    falloff = 1.0 / np.maximum(np.hypot(fx, fy), 1.0 / max(w, h)) ** 1.6
    chans = []
    for _ in range(3):
        spec = (rng.normal(size=(h, w // 2 + 1)) + 1j * rng.normal(size=(h, w // 2 + 1))) * falloff
        c = np.fft.irfft2(spec, s=(h, w))
        c = (c - c.mean()) / (c.std() + 1e-9)
        chans.append(c)
    mix = np.stack(chans, -1) @ rng.uniform(-0.4, 1.0, (3, 3))
    arr = 128 + 40 * mix / (mix.std() + 1e-9) + rng.normal(0, 3, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def flat_graphic(seed: int, size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", size, (245, 245, 240))
    d = ImageDraw.Draw(im)
    w, h = size
    for _ in range(8):
        x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        x1, y1 = x0 + int(rng.integers(40, w // 3)), y0 + int(rng.integers(40, h // 3))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x1, y1], fill=col)
    return im


def bar_chart(seed: int, size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    w, h = size
    d.line([(50, 20), (50, h - 40), (w - 20, h - 40)], fill="black", width=2)
    n = 7
    bw = (w - 90) // n
    for i in range(n):
        bh = int(rng.integers(30, h - 80))
        x = 60 + i * bw
        d.rectangle([x, h - 40 - bh, x + bw - 12, h - 41], fill=(40, 90, 170))
        d.text((x + 4, h - 34), f"Q{i + 1}", fill="black")
    d.text((60, 4), "Visitors per quarter", fill="black")
    return im


def screenshot(seed: int, size: tuple[int, int]) -> Image.Image:
    """A desktop-window look: title bar, sidebar, lines of text."""
    rng = np.random.default_rng(seed)
    w, h = size
    im = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    d.rectangle([0, 0, w, 32], fill=(230, 230, 232))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + 20 * i, 10, 24 + 20 * i, 22], fill=c)
    d.rectangle([0, 32, 200, h], fill=(240, 240, 244))
    for i, item in enumerate(["Inbox", "Drafts", "Sent", "Archive", "Garden club", "Recipes"]):
        d.text((20, 50 + 28 * i), item, fill=(60, 60, 60), font=font)
    words = ("the meeting moved to thursday please bring chairs tea will be provided "
             "minutes from last week are attached and the new rota starts in march").split()
    y = 50
    while y < h - 30:
        line = " ".join(rng.choice(words, int(rng.integers(6, 14))))
        d.text((220, y), line.capitalize() + ".", fill=(20, 20, 20), font=font)
        y += 24
    return im


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        if old.name != "manifest.json":
            old.unlink()
    files: list[dict] = []

    def add(name: str, note: str):
        files.append({"file": name, "expect": [], "note": note})

    for seed, size, q in ((1, (1024, 768), 70), (2, (640, 480), 95), (3, (512, 512), 40), (4, (800, 600), 85)):
        name = f"photo_like_{size[0]}x{size[1]}_q{q}.jpg"
        photo_like(seed, size).save(OUT / name, "JPEG", quality=q)
        add(name, f"photo-like texture, JPEG quality {q}")
    photo_like(5, (600, 400)).save(OUT / "photo_like_600x400.webp", "WEBP", quality=75)
    add("photo_like_600x400.webp", "photo-like texture, lossy WebP")
    photo_like(6, (384, 384)).save(OUT / "photo_like_384x384.png", optimize=True)
    add("photo_like_384x384.png", "photo-like texture, PNG")

    flat_graphic(7, (800, 600)).save(OUT / "flat_graphic_800x600.png", optimize=True)
    add("flat_graphic_800x600.png", "flat shapes on a plain background, like a logo or slide")
    flat_graphic(8, (512, 512)).save(OUT / "flat_graphic_512.jpg", "JPEG", quality=90)
    add("flat_graphic_512.jpg", "flat shapes saved as JPEG")
    bar_chart(9, (640, 400)).save(OUT / "bar_chart_640x400.png", optimize=True)
    add("bar_chart_640x400.png", "a simple bar chart with text")

    screenshot(10, (1280, 800)).save(OUT / "screenshot_1280x800.png", optimize=True)
    add("screenshot_1280x800.png", "screenshot-like picture of a window with text")
    screenshot(11, (900, 600)).save(OUT / "screenshot_900x600.jpg", "JPEG", quality=80)
    add("screenshot_900x600.jpg", "screenshot-like picture with text, saved as JPEG")

    photo_like(12, (600, 400)).convert("L").save(OUT / "grayscale_600x400.jpg", "JPEG", quality=88)
    add("grayscale_600x400.jpg", "grayscale (one channel) JPEG")
    flat_graphic(13, (400, 400)).convert("L").save(OUT / "grayscale_400.png", optimize=True)
    add("grayscale_400.png", "grayscale PNG")

    rgba = photo_like(14, (512, 512)).convert("RGBA")
    a = Image.new("L", rgba.size, 0)
    ImageDraw.Draw(a).ellipse([40, 40, 472, 472], fill=255)
    rgba.putalpha(a.filter(ImageFilter.GaussianBlur(6)))
    rgba.save(OUT / "alpha_512.png", optimize=True)
    add("alpha_512.png", "picture with see-through edges (alpha channel)")

    photo_like(15, (16, 16)).save(OUT / "tiny_16x16.png")
    add("tiny_16x16.png", "a tiny 16x16 picture, like an icon")
    photo_like(16, (100, 80)).save(OUT / "small_100x80.jpg", "JPEG", quality=85)
    add("small_100x80.jpg", "a small thumbnail-sized JPEG")

    flat_graphic(17, (320, 240)).quantize(64).save(OUT / "palette_320x240.gif")
    add("palette_320x240.gif", "GIF with a 64-colour palette")
    photo_like(18, (1600, 400)).save(OUT / "panorama_1600x400.jpg", "JPEG", quality=80)
    add("panorama_1600x400.jpg", "very wide picture")
    photo_like(19, (480, 960)).save(OUT / "tall_480x960.jpg", "JPEG", quality=82)
    add("tall_480x960.jpg", "very tall picture, like a phone photo")

    noise = np.random.default_rng(20).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(noise, "RGB").save(OUT / "noise_256.png")
    add("noise_256.png", "pure random noise, the hardest case for hidden-mark checks")

    info = PngInfo()
    info.add_text("Description", "Slides from the Stable Diffusion and DALL-E talk at the library, 4 October.")
    info.add_text("Author", "Garden club")
    flat_graphic(21, (480, 360)).save(OUT / "caption_mentions_sd.png", pnginfo=info)
    add("caption_mentions_sd.png", "PNG whose caption names AI tools; ordinary text, not a label")

    (OUT / "notes_about_c2pa.txt").write_text(
        "Reading list: C2PA content credentials, IPTC digital source type,\n"
        "trainedAlgorithmicMedia, TrustMark, invisible-watermark (dwtDct).\n")
    add("notes_about_c2pa.txt", "text file that names labels and watermarks; mentions are not labels")

    (OUT / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    total = sum((OUT / f["file"]).stat().st_size for f in files)
    print(f"Wrote {len(files)} controls to {OUT} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
