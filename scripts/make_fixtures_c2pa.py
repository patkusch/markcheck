"""Make the signed-label (C2PA) test files in tests/fixtures/c2pa/.

What it does:
1. Draws a small natural-looking picture (sky gradient, hills, sun, mild noise).
2. Makes a throwaway test certificate authority and a signing certificate.
   These are TEST certificates. They are not on any trust list, and the
   checker must say so.
3. Signs copies of the picture with c2pa-python, saying different things:
   made by an AI model (JPEG and PNG), and an ordinary edit with no AI.
4. Changes a few pixel bytes in one signed copy, so the checker can show
   that it reports "changed after signing".
5. Saves an unsigned copy as a control.
6. Signs an AI label onto more file types the c2pa Builder supports: WebP,
   GIF, TIFF, WAV sound, MP4 video and PDF (each with an unsigned control).
   A type the library or this computer cannot make is skipped with a message.

Pictures and test certificates come from fixed seeds, key numbers and
dates, so every run makes the same pictures and certificates. Signed files
still differ byte for byte between runs (each signature and label ID is
fresh), but they say the same things.

Run from the repo root:  .venv/bin/python scripts/make_fixtures_c2pa.py
The output is committed, so tests never need to run this.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import random
import struct
import tempfile
import wave
from pathlib import Path

import c2pa
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "c2pa"
CONTROLS = OUT / "controls"

AI_SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
TEST_MODEL = "markcheck-test-model 1.0 (fictional, for tests)"
GENERATOR = {"name": "markcheck fixture maker", "version": "0.1.0"}


# ---------- picture ----------

def base_image(size: int = 512, seed: int = 7) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        t = y / size
        sky = (int(90 + 120 * t), int(150 + 70 * t), int(230 - 40 * t))
        for x in range(size):
            px[x, y] = sky
    d = ImageDraw.Draw(img)
    d.ellipse((350, 60, 440, 150), fill=(255, 226, 140))
    for i, (base, col) in enumerate([(330, (70, 120, 70)), (380, (50, 95, 55)), (430, (35, 75, 40))]):
        pts = [(0, size)]
        for x in range(0, size + 32, 32):
            pts.append((x, base + rnd.randint(-40, 25) - i * 5))
        pts.append((size, size))
        d.polygon(pts, fill=col)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    px = img.load()
    for y in range(size):
        for x in range(size):
            r, g, b = px[x, y]
            n = rnd.randint(-6, 6)
            px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
    return img


def encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, "JPEG", quality=88)
    else:
        img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def tiny_wav() -> bytes:
    """Half a second of a quiet 440 Hz tone, 8 kHz mono."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / 8000)))
                               for i in range(4000)))
    return buf.getvalue()


def tiny_mp4(tmp: Path) -> bytes | None:
    """One second of 64x64 video (10 frames fading from dark to light), via OpenCV."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    p = tmp / "tiny.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    if not vw.isOpened():
        return None
    for i in range(10):
        vw.write(np.full((64, 64, 3), 20 + i * 20, np.uint8))
    vw.release()
    return p.read_bytes() if p.is_file() and p.stat().st_size else None


def tiny_pdf() -> bytes:
    """A one-page PDF with one line of text, written by hand (no PDF library needed)."""
    text = b"BT /F1 18 Tf 40 100 Td (markcheck test page) Tj ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(text), text),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, o)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return out


# ---------- test certificates ----------

# Fixed key numbers, serials and dates so every run makes the same test
# certificates. These keys are public on purpose: they only sign test files.
CA_KEY_NUMBER = 0x6D61726B636865636B2D746573742D726F6F742D6361  # "markcheck-test-root-ca"
SIGNER_KEY_NUMBER = 0x6D61726B636865636B2D746573742D7369676E6572  # "markcheck-test-signer"
CERT_START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def make_test_certs() -> tuple[str, str, str]:
    """Return (signing chain PEM, private key PEM, root PEM). Test use only."""
    now = CERT_START
    ca_key = ec.derive_private_key(CA_KEY_NUMBER, ec.SECP256R1())
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "markcheck test CA (not trusted)"),
        x509.NameAttribute(NameOID.COMMON_NAME, "markcheck Test Root CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(0x4D43_0001)
        .not_valid_before(now).not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.derive_private_key(SIGNER_KEY_NUMBER, ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "markcheck test signer"),
            x509.NameAttribute(NameOID.COMMON_NAME, "markcheck Test Signer"),
        ]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(0x4D43_0002)
        .not_valid_before(now).not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    pem = lambda c: c.public_bytes(serialization.Encoding.PEM).decode()
    chain = pem(leaf) + pem(ca_cert)
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode()
    return chain, key_pem, pem(ca_cert)


# ---------- signing ----------

# Signing makes no network calls either: no timestamp server, no OCSP.
OFFLINE = {"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False, "verify_trust": False},
           "builder": {"certificate_status_fetch": None}}


def sign(src: bytes, mime: str, manifest: dict, chain: str, key: str,
         ingredient: tuple[dict, bytes] | None = None, thumbnail: bool = True) -> bytes:
    """Sign with the c2pa library's own Builder. thumbnail=False leaves out the small
    preview picture the library would otherwise add, which keeps test files small."""
    info = c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, chain.encode(), key.encode(), None)
    signer = c2pa.Signer.from_info(info)
    settings: dict = {"verify": OFFLINE["verify"]}
    if not thumbnail:
        settings["builder"] = {"thumbnail": {"enabled": False}}
    ctx = c2pa.Context.from_dict(settings)
    builder = c2pa.Builder(manifest, ctx)
    if ingredient:
        builder.add_ingredient(ingredient[0], mime, io.BytesIO(ingredient[1]))
    out = io.BytesIO()
    builder.sign(signer, mime, io.BytesIO(src), out)
    return out.getvalue()


def ai_manifest(title: str) -> dict:
    return {
        "claim_generator_info": [GENERATOR],
        "title": title,
        "assertions": [{
            "label": "c2pa.actions",
            "data": {"actions": [{
                "action": "c2pa.created",
                "digitalSourceType": AI_SOURCE,
                "softwareAgent": {"name": TEST_MODEL},
            }]},
        }],
    }


def edit_manifest(title: str) -> dict:
    return {
        "claim_generator_info": [GENERATOR],
        "title": title,
        "assertions": [{
            "label": "c2pa.actions",
            "data": {"actions": [
                {"action": "c2pa.opened", "parameters": {"ingredientIds": ["parent"]}},
                {"action": "c2pa.color_adjustments", "softwareAgent": {"name": "markcheck test photo editor 1.0"}},
            ]},
        }],
    }


def tamper(data: bytes) -> bytes:
    """Flip a few bytes in the middle of the JPEG picture data (after the label)."""
    b = bytearray(data)
    sos = b.rfind(b"\xff\xda")
    assert sos > 0
    start = sos + 2 + int.from_bytes(b[sos + 2:sos + 4], "big") + 4000
    for i in range(start, start + 8):
        if b[i] not in (0xFF, 0x00):
            b[i] ^= 0x10
    return bytes(b)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CONTROLS.mkdir(parents=True, exist_ok=True)
    img = base_image()
    jpg, png = encode(img, "JPEG"), encode(img, "PNG")
    chain, key, root = make_test_certs()
    (OUT / "test-root-ca.pem").write_text(root)  # public cert only, so tests can show the chain is not trusted

    files = []

    def put(name, data, ai, note, expect=("c2pa",)):
        (OUT / name).write_bytes(data)
        files.append({"file": name, "expect": list(expect), "ai": ai, "note": note})

    put("ai_generated.jpg", sign(jpg, "image/jpeg", ai_manifest("ai_generated.jpg"), chain, key), True,
        "Signed with a test certificate; label says made by a (fictional) AI model.")
    put("ai_generated.png", sign(png, "image/png", ai_manifest("ai_generated.png"), chain, key), True,
        "PNG version of the AI label.")
    parent = ({"title": "unsigned.jpg", "relationship": "parentOf", "label": "parent"}, jpg)
    put("ordinary_edit.jpg", sign(jpg, "image/jpeg", edit_manifest("ordinary_edit.jpg"), chain, key, parent), False,
        "Signed with a test certificate; label says opened and colour-adjusted, no AI source type.")
    signed = sign(jpg, "image/jpeg", ai_manifest("changed_after_signing.jpg"), chain, key)
    put("changed_after_signing.jpg", tamper(signed), True,
        "AI label, then 8 bytes of picture data changed after signing. Must report changed after signing.")

    (CONTROLS / "unsigned.jpg").write_bytes(jpg)
    (CONTROLS / "unsigned.png").write_bytes(png)
    files.append({"file": "controls/unsigned.jpg", "expect": [], "ai": False, "note": "Same picture, never signed."})
    files.append({"file": "controls/unsigned.png", "expect": [], "ai": False, "note": "Same picture, never signed."})

    # More file types the c2pa Builder can sign. A smaller picture and no
    # preview thumbnail keep these files small.
    small = base_image(256, seed=8)
    others: list[tuple[str, str, bytes | None, str]] = []
    for ext, fmt, mime, kw in (("webp", "WEBP", "image/webp", {"quality": 85}),
                               ("gif", "GIF", "image/gif", {}),
                               ("tif", "TIFF", "image/tiff", {"compression": "tiff_deflate"})):
        buf = io.BytesIO()
        small.save(buf, fmt, **kw)
        others.append((ext, mime, buf.getvalue(), f"{fmt} picture"))
    others.append(("wav", "audio/wav", tiny_wav(), "half a second of WAV audio (a plain tone)"))
    with tempfile.TemporaryDirectory() as tmp:
        others.append(("mp4", "video/mp4", tiny_mp4(Path(tmp)), "one second of MP4 video (64x64, made with OpenCV)"))
    others.append(("pdf", "application/pdf", tiny_pdf(), "one-page PDF"))
    for ext, mime, data, what in others:
        if data is None:
            print(f"skipped {ext}: could not make a source file here")
            continue
        try:
            signed_other = sign(data, mime, ai_manifest(f"ai_generated.{ext}"), chain, key, thumbnail=False)
        except Exception as exc:  # report honestly and move on
            print(f"skipped {ext}: the c2pa Builder could not sign it ({exc})")
            continue
        put(f"ai_generated.{ext}", signed_other, True, f"AI label on a {what}, signed with a test certificate.")
        (CONTROLS / f"unsigned.{ext}").write_bytes(data)
        files.append({"file": f"controls/unsigned.{ext}", "expect": [], "ai": False,
                      "note": f"The same {what}, never signed."})

    (OUT / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(f"wrote {len(files)} files to {OUT}")


if __name__ == "__main__":
    main()
