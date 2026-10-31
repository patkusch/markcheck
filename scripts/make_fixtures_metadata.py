"""Build the metadata test files and the shared unlabelled controls.

Run:  .venv/bin/python scripts/make_fixtures_metadata.py

Every labelled file here is our own generated picture with a real XMP or
text tag written into it by this script. No tag is copied from anyone
else's file. The controls carry no AI tag at all.
"""
from __future__ import annotations

import json
import random
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
META = ROOT / "metadata"
CTRL = ROOT / "controls"

IPTC_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
DST = "cv.iptc.org/newscodes/digitalsourcetype/"
TC260_NS = "http://www.tc260.org.cn/ns/AIGC/1.0/"


def base_image(seed: int, size=(320, 240)) -> Image.Image:
    """A natural-looking picture: sky gradient, hills, a sun, mild noise."""
    rnd = random.Random(seed)
    w, h = size
    img = Image.new("RGB", size)
    top = (rnd.randint(60, 140), rnd.randint(120, 190), rnd.randint(200, 255))
    bot = (rnd.randint(200, 255), rnd.randint(170, 220), rnd.randint(120, 180))
    px = img.load()
    for y in range(h):
        t = y / (h - 1)
        c = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        for x in range(w):
            px[x, y] = c
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.65, h * 0.1, w * 0.85, h * 0.37], fill=(255, 236, 170))
    for k in range(3):
        base = int(h * (0.55 + 0.12 * k))
        pts = [(0, h)] + [(x, base + int(18 * rnd.random() - 9 + 14 * ((x / w * (3 + k)) % 1))) for x in range(0, w + 20, 20)] + [(w, h)]
        g = (40 + 25 * k, 90 + 20 * k + rnd.randint(0, 20), 50 + 10 * k)
        d.polygon(pts, fill=g)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    noise = Image.effect_noise(size, 12).convert("RGB")
    return Image.blend(img, noise, 0.06)


def xmp_packet(description_inner: str, attrs: str = "", ns: str = "") -> bytes:
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'  <rdf:Description rdf:about="" xmlns:Iptc4xmpExt="{IPTC_NS}" '
        f'xmlns:xmp="http://ns.adobe.com/xap/1.0/" {ns} {attrs}>\n'
        f'{description_inner}'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    ).encode("utf-8")


def jpeg_with_app1(img: Image.Image, segments: list[bytes]) -> bytes:
    """Save a JPEG, then insert APP1 segments right after the start marker."""
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    raw = buf.getvalue()
    extra = b"".join(b"\xff\xe1" + struct.pack(">H", len(s) + 2) + s for s in segments)
    return raw[:2] + extra + raw[2:]


def xmp_app1(packet: bytes) -> bytes:
    return b"http://ns.adobe.com/xap/1.0/\x00" + packet


AIGC_SAMPLE = {
    "Label": "1",
    "ContentProducer": "001191330000EXAMPLE0",
    "ProduceID": "markcheck-fixture-0001",
    "ReservedCode1": "",
    "ContentPropagator": "001191330000EXAMPLE0",
    "PropagateID": "markcheck-fixture-0001",
    "ReservedCode2": "",
}


def build_metadata() -> list[dict]:
    META.mkdir(parents=True, exist_ok=True)
    out = []

    # 1. JPEG, attribute form, full http URI
    p = xmp_packet("", attrs=f'Iptc4xmpExt:DigitalSourceType="http://{DST}trainedAlgorithmicMedia"')
    (META / "iptc_trained_uri_attr.jpg").write_bytes(jpeg_with_app1(base_image(1), [xmp_app1(p)]))
    out.append({"file": "iptc_trained_uri_attr.jpg", "expect": ["metadata"],
                "note": "IPTC DigitalSourceType trainedAlgorithmicMedia, full http URI, attribute form"})

    # 2. JPEG, element form, short code
    p = xmp_packet("   <Iptc4xmpExt:DigitalSourceType>compositeWithTrainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>\n")
    (META / "iptc_composite_short_element.jpg").write_bytes(jpeg_with_app1(base_image(2), [xmp_app1(p)]))
    out.append({"file": "iptc_composite_short_element.jpg", "expect": ["metadata"],
                "note": "IPTC compositeWithTrainedAlgorithmicMedia, short code, element form"})

    # 3. PNG iTXt XMP, https URI as rdf:resource
    p = xmp_packet(f'   <Iptc4xmpExt:DigitalSourceType rdf:resource="https://{DST}trainedAlgorithmicMedia"/>\n')
    info = PngInfo()
    info.add_itxt("XML:com.adobe.xmp", p.decode("utf-8"), zip=False)
    base_image(3).save(META / "iptc_trained_https.png", pnginfo=info)
    out.append({"file": "iptc_trained_https.png", "expect": ["metadata"],
                "note": "PNG iTXt XMP, https URI as rdf:resource"})

    # 4. WebP with XMP chunk
    p = xmp_packet("", attrs=f'Iptc4xmpExt:DigitalSourceType="http://{DST}compositeSynthetic"')
    base_image(4).save(META / "iptc_composite_synthetic.webp", "WEBP", quality=85, xmp=p)
    out.append({"file": "iptc_composite_synthetic.webp", "expect": ["metadata"],
                "note": "WebP XMP chunk, compositeSynthetic"})

    # 5. AIGC field in XMP (JSON value under a TC260-namespace AIGC property)
    p = xmp_packet(f"   <TC260:AIGC>{json.dumps(AIGC_SAMPLE)}</TC260:AIGC>\n", ns=f'xmlns:TC260="{TC260_NS}"')
    (META / "aigc_xmp.jpg").write_bytes(jpeg_with_app1(base_image(5), [xmp_app1(p)]))
    out.append({"file": "aigc_xmp.jpg", "expect": ["metadata"],
                "note": "China AIGC label as JSON in an XMP property (layout from secondary sources)"})

    # 6. AIGC field in a PNG text chunk
    info = PngInfo()
    info.add_text("AIGC", json.dumps(AIGC_SAMPLE))
    base_image(6).save(META / "aigc_png_text.png", pnginfo=info)
    out.append({"file": "aigc_png_text.png", "expect": ["metadata"],
                "note": "China AIGC label as JSON in a PNG tEXt chunk named AIGC"})

    # 7. Stable Diffusion style generation settings in PNG "parameters"
    params = ("a quiet valley at sunset, soft light, wide angle\n"
              "Negative prompt: blurry, low quality\n"
              "Steps: 25, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1234567890, "
              "Size: 320x240, Model hash: 0123abcd, Model: example_model_v1")
    info = PngInfo()
    info.add_text("parameters", params)
    base_image(7).save(META / "sd_parameters.png", pnginfo=info)
    out.append({"file": "sd_parameters.png", "expect": ["metadata"],
                "note": "AUTOMATIC1111-style PNG 'parameters' text (a hint: anyone can type this)"})

    (META / "manifest.json").write_text(json.dumps({"files": out}, indent=2) + "\n")
    return out


def tiny_pdf() -> bytes:
    content = b"BT /F1 18 Tf 40 100 Td (A short test page.) Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


def build_controls() -> list[dict]:
    CTRL.mkdir(parents=True, exist_ok=True)
    out = []
    base_image(101).save(CTRL / "plain.jpg", "JPEG", quality=88)
    out.append({"file": "plain.jpg", "expect": [], "note": "plain JPEG, no metadata"})
    base_image(102).save(CTRL / "plain.png")
    out.append({"file": "plain.png", "expect": [], "note": "plain PNG, no metadata"})
    base_image(103).save(CTRL / "plain.webp", "WEBP", quality=85)
    out.append({"file": "plain.webp", "expect": [], "note": "plain WebP, no metadata"})

    p = xmp_packet(f'   <Iptc4xmpExt:DigitalSourceType>http://{DST}digitalCapture</Iptc4xmpExt:DigitalSourceType>\n'
                   '   <xmp:CreatorTool>Adobe Lightroom Classic 13.0</xmp:CreatorTool>\n')
    (CTRL / "xmp_digital_capture.jpg").write_bytes(jpeg_with_app1(base_image(104), [xmp_app1(p)]))
    out.append({"file": "xmp_digital_capture.jpg", "expect": [],
                "note": "ordinary XMP with IPTC digitalCapture; must not be flagged"})

    img = base_image(105)
    exif = Image.Exif()
    exif[0x010F] = "Canon"            # Make
    exif[0x0110] = "Canon EOS R6"     # Model
    exif[0x0131] = "Firmware Version 1.8.1"  # Software
    exif[0x0132] = "2026:10:01 09:30:00"     # DateTime
    img.save(CTRL / "camera_exif.jpg", "JPEG", quality=90, exif=exif)
    out.append({"file": "camera_exif.jpg", "expect": [], "note": "JPEG with camera EXIF"})

    (CTRL / "note.txt").write_text(
        "Meeting notes, 3 October.\nWe agreed to move the garden club to Thursdays.\n"
        "Bring chairs. Tea will be provided.\n")
    out.append({"file": "note.txt", "expect": [], "note": "plain text file"})

    (CTRL / "tiny.pdf").write_bytes(tiny_pdf())
    out.append({"file": "tiny.pdf", "expect": [], "note": "tiny one-page PDF, no XMP"})

    info = PngInfo()
    info.add_text("Description", "Photo of the AI lab team at the open day; AI talks ran all afternoon.")
    info.add_text("Software", "GIMP 2.10.36")
    base_image(106).save(CTRL / "caption_mentions_ai.png", pnginfo=info)
    out.append({"file": "caption_mentions_ai.png", "expect": [],
                "note": "PNG whose caption mentions 'AI'; must not be flagged"})

    (CTRL / "manifest.json").write_text(json.dumps({"files": out}, indent=2) + "\n")
    return out


if __name__ == "__main__":
    a = build_metadata()
    b = build_controls()
    print(f"wrote {len(a)} metadata fixtures to {META}")
    print(f"wrote {len(b)} controls to {CTRL}")
