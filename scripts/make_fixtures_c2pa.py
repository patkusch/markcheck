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

Run from the repo root:  .venv/bin/python scripts/make_fixtures_c2pa.py
The output is committed, so tests never need to run this.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import random
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


# ---------- test certificates ----------

def make_test_certs() -> tuple[str, str, str]:
    """Return (signing chain PEM, private key PEM, root PEM). Test use only."""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "markcheck test CA (not trusted)"),
        x509.NameAttribute(NameOID.COMMON_NAME, "markcheck Test Root CA"),
    ])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name).issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now).not_valid_after(now + dt.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "markcheck test signer"),
            x509.NameAttribute(NameOID.COMMON_NAME, "markcheck Test Signer"),
        ]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
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
         ingredient: tuple[dict, bytes] | None = None) -> bytes:
    info = c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, chain.encode(), key.encode(), None)
    signer = c2pa.Signer.from_info(info)
    ctx = c2pa.Context.from_dict({"verify": OFFLINE["verify"]})
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

    (OUT / "manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(f"wrote {len(files)} files to {OUT}")


if __name__ == "__main__":
    main()
