"""Make test text files that carry a signed label (C2PA) hidden after the words.

    python scripts/make_fixtures_text_label.py

The label is signed with the same fictional test certificates as the other C2PA
fixtures, then hidden after the words the way C2PA's plain-text method does.
Writes tests/fixtures/c2pa/text/: one good label and one whose signature was broken.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import c2pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fixtures_c2pa as m  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from markcheck.textcheck.hidden import MAGIC, MARKER  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "c2pa" / "text"
WORDS = "The committee met on Tuesday and agreed the new budget for the library.\n"


def hide(payload: bytes) -> str:
    framed = MAGIC + bytes([1]) + len(payload).to_bytes(4, "big") + payload
    return MARKER + "".join(chr(0xFE00 + b) if b < 0x10 else chr(0xE0100 + b - 0x10) for b in framed)


def label_bytes() -> bytes:
    chain, key = m.make_test_certs()[:2]
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, chain.encode(), key.encode(), None))
    ctx = c2pa.Context.from_dict({"verify": m.OFFLINE["verify"], "builder": {"thumbnail": {"enabled": False}}})
    builder = c2pa.Builder(m.ai_manifest("ai_text_label.txt"), ctx)
    builder.set_no_embed()
    # The c2pa library cannot sign plain text yet, so the label is signed over a tiny picture.
    # Its signature still checks out; markcheck never claims the words are unchanged for text.
    src = (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "c2pa" / "controls" / "unsigned.jpg").read_bytes()
    return builder.sign(signer, "image/jpeg", io.BytesIO(src), io.BytesIO())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = label_bytes()
    (OUT / "ai_text_label.txt").write_text(WORDS + hide(payload), encoding="utf-8")
    broken = bytearray(payload)
    broken[-40] ^= 0xFF  # flips bits inside the signature
    (OUT / "broken_signature.txt").write_text(WORDS + hide(bytes(broken)), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
