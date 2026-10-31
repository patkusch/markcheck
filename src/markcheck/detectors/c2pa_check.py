"""Signed label check (C2PA, "Content Credentials").

Reads the signed label with c2pa-python and explains it in plain English:
who signed it, which tool or AI model it names, whether it says AI was used,
whether the signature checks out, whether the file changed after signing, and
whether the signer is on the official C2PA trust list (bundled, checked offline).

This check never goes online: fetching labels stored on the web and
certificate-status (OCSP) lookups are both switched off.
"""
from __future__ import annotations

import io
import json
import re
from functools import lru_cache
from pathlib import Path

from ..findings import Finding

NAME = "c2pa"
LABEL = "Signed label (C2PA)"

TRUST_LIST = Path(__file__).resolve().parent.parent / "data" / "C2PA-TRUST-LIST.pem"

# Kinds of file that can carry a C2PA label.
MEDIA_KINDS = {"image", "video", "audio", "pdf"}

# Settings that keep every read offline.
OFFLINE_VERIFY = {
    "remote_manifest_fetch": False,  # never download a label stored on the web
    "ocsp_fetch": False,             # never ask a certificate server about revocation
}

MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif", ".tif": "image/tiff", ".tiff": "image/tiff", ".heic": "image/heic",
    ".heif": "image/heif", ".avif": "image/avif", ".svg": "image/svg+xml", ".dng": "image/x-adobe-dng",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".m4v": "video/mp4", ".avi": "video/x-msvideo",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".flac": "audio/flac",
    ".pdf": "application/pdf",
}

# IPTC digital source types: what each one says, and whether it means AI.
SOURCE_TYPES = {
    "trainedalgorithmicmedia": ("made by an AI model", "made"),
    "compositewithtrainedalgorithmicmedia": ("mixes AI-made parts with other material", "edited"),
    "algorithmicmedia": ("made by a computer program that is not a trained AI model", None),
    "algorithmicallyenhanced": ("touched up by software", None),
    "compositesynthetic": ("mixes computer-made parts with other material", None),
    "compositecapture": ("combines several camera captures", None),
    "composite": ("combines several pieces of material", None),
    "digitalcapture": ("captured by a camera or recorder", None),
    "computationalcapture": ("captured by a camera with on-device processing", None),
    "digitalcreation": ("made by a person with software", None),
    "humanedits": ("edited by a person", None),
    "screencapture": ("a screenshot or screen recording", None),
    "datadrivenmedia": ("made from data, for example a chart", None),
    "virtualrecording": ("recorded inside a virtual scene", None),
    "negativefilm": ("scanned from film", None),
    "positivefilm": ("scanned from film", None),
    "print": ("scanned from a print", None),
}

FILE_HASH_CODES = re.compile(r"^assertion\.(dataHash|bmffHash|boxesHash|collectionHash)\.mismatch$")
LABEL_CHANGED_CODES = {"assertion.hashedURI.mismatch", "assertion.missing", "claim.missing",
                       "assertion.cloudData.hardBinding"}
SIGNATURE_BAD_CODES = {"claimSignature.mismatch", "claimSignature.missing", "signingCredential.invalid",
                       "signingCredential.revoked"}


# ---------- reading ----------

@lru_cache(maxsize=4)
def _context(extra_anchors: str | None = None):
    """A c2pa Context that never goes online and checks against the bundled trust list."""
    import c2pa

    anchors = TRUST_LIST.read_text() if TRUST_LIST.is_file() else ""
    if extra_anchors:
        anchors = anchors + "\n" + extra_anchors
    cfg = {"verify": dict(OFFLINE_VERIFY, verify_trust=bool(anchors))}
    if anchors:
        cfg["trust"] = {"trust_anchors": anchors}
    return c2pa.Context.from_dict(cfg)


def _has_c2pa_bytes(data: bytes) -> bool:
    return b"jumb" in data and b"c2pa" in data


def _mime(path: Path) -> str | None:
    return MIME.get(Path(path).suffix.lower())


def check(path, data, kind, options) -> list[Finding]:
    return [inspect_bytes(Path(path), data, kind, (options or {}).get("c2pa_extra_anchors"))]


def inspect_bytes(path: Path, data: bytes, kind: str, extra_anchors: str | None = None) -> Finding:
    if kind not in MEDIA_KINDS and not _has_c2pa_bytes(data):
        return Finding(NAME, LABEL, "skipped", "Signed labels (C2PA) are not used in this kind of file.")
    try:
        import c2pa
    except Exception as exc:  # library missing or broken
        return Finding(NAME, LABEL, "skipped", f"The C2PA library is not available, so this was not checked ({exc}).")

    ctx = _context(extra_anchors)
    store = None
    last_exc: Exception | None = None
    for fmt in (_mime(path), None):  # first the file name's type, then let c2pa work it out
        try:
            reader = c2pa.Reader.try_create(fmt, io.BytesIO(data), None, ctx)
            if reader is None:
                return _not_found()
            with reader:
                store = json.loads(reader.json())
                try:
                    cr = json.loads(reader.crjson())
                except Exception:
                    cr = {}
            break
        except Exception as exc:
            last_exc = exc
            if fmt is None:
                break
    if store is None:
        return _from_read_error(last_exc, data)
    return _explain(store, cr, trust_checked=TRUST_LIST.is_file() or bool(extra_anchors))


def _not_found() -> Finding:
    return Finding(NAME, LABEL, "not_found", "No signed label (C2PA) found.")


def _from_read_error(exc: Exception | None, data: bytes) -> Finding:
    msg = str(exc or "")
    if "remote manifest" in msg.lower():
        url = re.search(r"https?://\S+", msg)
        where = f" at {url.group(0)}" if url else ""
        return Finding(
            NAME, LABEL, "found",
            f"This file points to a signed label stored online{where}. markcheck never goes online, so it was not checked.",
            details={"label_location": "online" + where, "signature_valid": "not checked"},
            proves=["The file says a signed label for it exists online."],
            does_not_prove=["Who signed it, what it says, or whether it matches this file: it was not downloaded."],
        )
    if "type is unsupported" in msg or "NotSupported" in msg:
        if _has_c2pa_bytes(data):
            return _damaged(msg)
        return Finding(NAME, LABEL, "skipped", "Signed labels (C2PA) cannot be read from this file type yet.")
    if _has_c2pa_bytes(data):
        return _damaged(msg)
    # The file itself could not be parsed and shows no sign of a label.
    return Finding(NAME, LABEL, "not_found",
                   "No signed label (C2PA) found. The file looks damaged or incomplete, so it could not be fully read.",
                   details={"read_problem": msg[:200]})


def _damaged(msg: str) -> Finding:
    return Finding(
        NAME, LABEL, "found",
        "This file seems to carry a signed label (C2PA), but the label is damaged and could not be read.",
        details={"label_readable": "no", "read_problem": msg[:200]},
        proves=["Part of the file is marked as a signed label."],
        does_not_prove=["Who signed it or what it says: the label could not be read.",
                        "That the content is accurate."],
    )


# ---------- explaining ----------

def _codes(store: dict) -> list[dict]:
    res = (store.get("validation_results") or {}).get("activeManifest") or {}
    fails = list(res.get("failure") or [])
    if not fails and not store.get("validation_results"):
        fails = [s for s in store.get("validation_status") or [] if isinstance(s, dict)]
    return fails


def _walk(obj, key: str):
    """Yield every value stored under `key` anywhere inside obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _agent_name(agent) -> str | None:
    if isinstance(agent, str):
        return agent.strip() or None
    if isinstance(agent, dict) and agent.get("name"):
        v = agent.get("version")
        return f"{agent['name']} {v}".strip() if v and str(v) not in agent["name"] else str(agent["name"])
    return None


def _source_types(manifest: dict) -> list[str]:
    out = []
    for key in ("digitalSourceType", "Iptc4xmpExt:DigitalSourceType", "source_type"):
        for v in _walk(manifest.get("assertions") or [], key):
            if isinstance(v, str) and v not in out:
                out.append(v)
    return out


def _agents(manifest: dict) -> list[str]:
    names: list[str] = []
    for key in ("softwareAgent", "softwareAgents"):
        for v in _walk(manifest.get("assertions") or [], key):
            for a in (v if isinstance(v, list) else [v]):
                n = _agent_name(a)
                if n and n not in names:
                    names.append(n)
    return names


def _generator(manifest: dict) -> str | None:
    parts = []
    for g in manifest.get("claim_generator_info") or []:
        n = _agent_name(g)
        if n:
            parts.append(n)
    if not parts and manifest.get("claim_generator"):
        parts.append(str(manifest["claim_generator"]))
    return ", ".join(parts) or None


def _source_meaning(uri: str) -> tuple[str, str | None]:
    key = uri.rstrip("/").rsplit("/", 1)[-1].lower()
    return SOURCE_TYPES.get(key, (f"source type '{uri}'", None))


def _cert_info(cr: dict, label: str) -> dict:
    for m in cr.get("manifests") or []:
        if m.get("label") == label:
            return ((m.get("signature") or {}).get("certificateInfo")) or {}
    return {}


def _name(d: dict) -> str:
    cn, org = d.get("CN"), d.get("O")
    if cn and org and org != cn:
        return f"{cn} ({org})"
    return cn or org or ""


def _explain(store: dict, cr: dict, trust_checked: bool) -> Finding:
    label = store.get("active_manifest")
    manifests = store.get("manifests") or {}
    m = manifests.get(label) or {}
    sig = m.get("signature_info") or {}
    cert = _cert_info(cr, label)

    subject = _name(cert.get("subject") or {}) or _name({"CN": sig.get("common_name"), "O": sig.get("issuer")})
    issuer = _name(cert.get("issuer") or {})
    signer = subject or "an unnamed signer"
    self_signed = bool(cert.get("subject")) and cert.get("subject") == cert.get("issuer")
    test_cert = "test" in f"{subject} {issuer}".lower()

    fails = _codes(store)
    codes = [f.get("code", "") for f in fails]
    file_changed = any(FILE_HASH_CODES.match(c) for c in codes)
    label_changed = any(c in LABEL_CHANGED_CODES for c in codes)
    sig_bad = any(c in SIGNATURE_BAD_CODES or c.startswith("claimSignature.") and c != "claimSignature.insideValidity"
                  for c in codes)
    expired = "signingCredential.expired" in codes

    state = store.get("validation_state")
    if not trust_checked:
        on_list = "not checked (the trust list is not installed)"
    elif state == "Trusted":
        on_list = "yes"
    elif "signingCredential.untrusted" in codes:
        on_list = "no"
    else:
        on_list = "could not be decided"

    # What the label says about AI, from this file's own label.
    types = _source_types(m)
    meanings = [_source_meaning(t) for t in types]
    ai_kind = next((k for _, k in meanings if k == "made"), None) or next((k for _, k in meanings if k), None)
    if ai_kind == "made":
        ai_text = "yes: it says this was made by an AI model"
    elif ai_kind == "edited":
        ai_text = "yes: it says this mixes AI-made parts with other material"
    else:
        ai_text = "no: the label does not say AI was used"
    # Earlier labels (ingredients) may say AI was used even if this one does not.
    earlier_ai = any(_source_meaning(t)[1] for k, mm in manifests.items() if k != label for t in _source_types(mm))

    agents = _agents(m)
    tool = _generator(m)

    details: dict = {
        "signed_by": signer,
        "certificate_issued_by": issuer or None,
        "signed_at": sig.get("time"),
        "made_with_tool": tool,
        "software_or_ai_named": ", ".join(agents) if agents else None,
        "source_type": "; ".join(t for t, _ in meanings) if meanings else None,
        "says_made_with_ai": ai_text,
        "earlier_label_says_ai": "yes" if earlier_ai else None,
        "signature_valid": "no" if sig_bad else "yes",
        "changed_since_signing": ("yes: the file's content no longer matches the label" if file_changed
                                  else "yes: parts of the label itself were changed" if label_changed
                                  else "cannot tell: the signature does not check out" if sig_bad
                                  else "no"),
        "on_c2pa_trust_list": on_list,
        "test_or_self_signed_certificate": "yes" if (self_signed or test_cert) else None,
        "earlier_labels": len(manifests) - 1 if len(manifests) > 1 else None,
        "problems": "; ".join(f"{f.get('code')} ({f.get('explanation')})" for f in fails
                              if f.get("code") != "signingCredential.untrusted") or None,
    }
    details = {k: v for k, v in details.items() if v is not None}

    # Summary sentence.
    if sig_bad:
        summary = f"A signed label naming {signer} is present, but its signature does not check out."
    elif file_changed:
        summary = f"Signed by {signer}, but the file was changed after signing."
    elif label_changed:
        summary = f"Signed by {signer}, but the label was changed after signing."
    else:
        summary = f"Signed by {signer}; unchanged since signing."
    if ai_kind == "made":
        summary += " The label says it was made with AI."
    elif ai_kind == "edited":
        summary += " The label says AI-made material was mixed in."

    proves: list[str] = []
    not_proves: list[str] = [
        "That the content is accurate or that the events shown happened. "
        "A signature shows who signed the label, not that what it shows is real.",
    ]
    if sig_bad:
        proves.append("The file carries a signed label, but the signature does not match it.")
        not_proves.append("Who made the label: with a broken signature the signer's name cannot be relied on.")
    else:
        proves.append(f"Whoever holds the signing key for the certificate named '{signer}' signed this label.")
        if file_changed or label_changed:
            proves.append("The file or its label was changed after signing, so the label no longer "
                          "describes these exact bytes.")
            not_proves.append("What was changed or whether it matters. Even re-saving a file can do this.")
        else:
            proves.append("The file has not changed since it was signed.")
        if ai_kind:
            proves.append("The signer stated in the label that AI was used.")
    if on_list != "yes":
        not_proves.append("That the signer is who the certificate name says. This certificate is not on the "
                          "official C2PA trust list, so anyone could have made a certificate with this name.")
    if self_signed or test_cert:
        not_proves.append("Anything about the real maker: this is a test or self-signed certificate, "
                          "which anyone can create.")
    if not ai_kind:
        not_proves.append("That no AI was used. The label only records what the signing tool chose to say.")
    if expired:
        not_proves.append("That the certificate was still in date when it was used.")

    return Finding(NAME, LABEL, "found", summary, details=details, proves=proves, does_not_prove=not_proves)
