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
# The older Content Credentials interim trust list (ITL), frozen on 1 January 2026.
# Real makers such as Adobe still sign with certificates that are on it but not on the
# new official list. C2PA asks checkers to keep the two apart, so we report it separately.
INTERIM_ANCHORS = TRUST_LIST.parent / "INTERIM-TRUST-ANCHORS.pem"
INTERIM_ALLOWED = TRUST_LIST.parent / "INTERIM-TRUST-ALLOWED.pem"

# Soft-binding algorithm names (C2PA "c2pa.soft-binding" assertion) in plain words.
SOFT_BINDING_NAMES = {
    "com.adobe.trustmark": "Adobe TrustMark",
    "com.digimarc": "Digimarc",
    "ai.steg": "Steg.AI",
    "io.iscc": "ISCC content code (a fingerprint, not a mark)",
}

# Kinds of file that can carry a C2PA label.
MEDIA_KINDS = {"image", "video", "audio", "pdf"}

# Settings that keep every read offline.
OFFLINE_VERIFY = {
    "remote_manifest_fetch": False,  # never download a label stored on the web
    "ocsp_fetch": False,             # never ask a certificate server about revocation
}

# Certificate purposes (EKUs) a C2PA signing certificate may carry. Without this list the
# library rejects certificates that use the C2PA claim-signing purpose, which real signers
# do (OpenAI's 2026 certificate from Trufo's C2PA CA, for one), and reports them as invalid.
# Same list as the Content Credentials verify site's store.cfg (contentauth/verify-site, Apache-2.0).
SIGNING_PURPOSES = "\n".join([
    "//id-kp-emailProtection", "1.3.6.1.5.5.7.3.4",
    "//id-kp-documentSigning", "1.3.6.1.5.5.7.3.36",
    "//id-kp-timeStamping", "1.3.6.1.5.5.7.3.8",
    "//id-kp-OCSPSigning", "1.3.6.1.5.5.7.3.9",
    "// MS C2PA Signing", "1.3.6.1.4.1.311.76.59.1.9",
    "// c2pa-claim_signing OID", "1.3.6.1.4.1.62558.2.1",
])

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
# Success codes we need before saying "unchanged since signing": the library must have
# confirmed both the signature and a hash that ties the label to the file's bytes.
SIGNATURE_OK_CODE = "claimSignature.validated"
FILE_HASH_MATCH = re.compile(r"^assertion\.(dataHash|bmffHash|boxesHash|collectionHash)\.match$")


# ---------- reading ----------

@lru_cache(maxsize=4)
def _context(extra_anchors: str | None = None):
    """A c2pa Context that never goes online and checks against the bundled trust list."""
    import c2pa

    anchors = TRUST_LIST.read_text() if TRUST_LIST.is_file() else ""
    if extra_anchors:
        anchors = anchors + "\n" + extra_anchors
    cfg = {"verify": dict(OFFLINE_VERIFY, verify_trust=bool(anchors)),
           "trust": {"trust_config": SIGNING_PURPOSES}}
    if anchors:
        cfg["trust"]["trust_anchors"] = anchors
    return c2pa.Context.from_dict(cfg)


@lru_cache(maxsize=1)
def _interim_context():
    """Offline context that trusts only the frozen interim list, or None if it is not installed."""
    import c2pa

    if not INTERIM_ANCHORS.is_file():
        return None
    trust = {"trust_anchors": INTERIM_ANCHORS.read_text(), "trust_config": SIGNING_PURPOSES}
    if INTERIM_ALLOWED.is_file():
        trust["allowed_list"] = INTERIM_ALLOWED.read_text()
    return c2pa.Context.from_dict({"verify": dict(OFFLINE_VERIFY, verify_trust=True), "trust": trust})


def _on_interim_list(fmt: str | None, data: bytes) -> bool | None:
    """True/False if the signer chains to the interim list; None if that could not be checked."""
    import c2pa

    ctx = _interim_context()
    if ctx is None:
        return None
    try:
        reader = c2pa.Reader.try_create(fmt, io.BytesIO(data), None, ctx)
        if reader is None:
            return None
        with reader:
            st = json.loads(reader.json())
    except Exception:
        return None
    codes = [f.get("code") for f in _codes(st)]
    if "signingCredential.untrusted" in codes:
        return False
    return st.get("validation_state") == "Trusted" or None


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
    used_fmt = None
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
            used_fmt = fmt
            break
        except Exception as exc:
            last_exc = exc
            if fmt is None:
                break
    if store is None:
        return _from_read_error(last_exc, data)
    trust_checked = TRUST_LIST.is_file() or bool(extra_anchors)
    interim = None
    if trust_checked and "signingCredential.untrusted" in [f.get("code") for f in _codes(store)]:
        interim = _on_interim_list(used_fmt, data)
    return _explain(store, cr, trust_checked=trust_checked, interim=interim)


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

def _success_codes(store: dict) -> list[str]:
    res = (store.get("validation_results") or {}).get("activeManifest") or {}
    return [s.get("code", "") for s in res.get("success") or [] if isinstance(s, dict)]


def _explained(code: str) -> bool:
    """Failure codes whose meaning we already report. Anything else blocks 'unchanged'."""
    return bool(FILE_HASH_CODES.match(code) or code in LABEL_CHANGED_CODES or code in SIGNATURE_BAD_CODES
                or code.startswith(("claimSignature.", "signingCredential.", "timeStamp.")))


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


def _soft_bindings(manifest: dict) -> list[str]:
    """Hidden marks (watermarks) the label says were put into the content ("soft bindings")."""
    out: list[str] = []
    for a in manifest.get("assertions") or []:
        if not str(a.get("label", "")).startswith("c2pa.soft-binding"):
            continue
        alg = str((a.get("data") or {}).get("alg") or "").strip()
        if not alg:
            continue
        name, variant = SOFT_BINDING_NAMES.get(alg), ""
        if name is None and "." in alg:
            base, _, variant = alg.rpartition(".")
            name = SOFT_BINDING_NAMES.get(base)
        text = (f"{name} (variant {variant})" if name and variant else name) or f"a hidden mark called '{alg}'"
        if text not in out:
            out.append(text)
    return out


def _explain(store: dict, cr: dict, trust_checked: bool, interim: bool | None = None) -> Finding:
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
    success = _success_codes(store)
    sig_ok = not sig_bad and SIGNATURE_OK_CODE in success
    hash_ok = any(FILE_HASH_MATCH.match(c) for c in success)
    unexplained = [c for c in codes if not _explained(c)]
    unchanged = sig_ok and hash_ok and not unexplained and not file_changed and not label_changed

    state = store.get("validation_state")
    if not trust_checked:
        on_list = "not checked (the trust list is not installed)"
    elif state == "Trusted" or "signingCredential.trusted" in success:
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
    marks = _soft_bindings(m)

    details: dict = {
        "signed_by": signer,
        "certificate_issued_by": issuer or None,
        "signed_at": sig.get("time"),
        "made_with_tool": tool,
        "software_or_ai_named": ", ".join(agents) if agents else None,
        "source_type": "; ".join(t for t, _ in meanings) if meanings else None,
        "says_made_with_ai": ai_text,
        "earlier_label_says_ai": "yes" if earlier_ai else None,
        "signature_valid": "no" if sig_bad else "yes" if sig_ok else "not confirmed",
        "changed_since_signing": ("yes: the file's content no longer matches the label" if file_changed
                                  else "yes: parts of the label itself were changed" if label_changed
                                  else "cannot tell: the signature does not check out" if sig_bad
                                  else "no" if unchanged
                                  else "not confirmed: the checks that tie the label to this file did not all pass"),
        "on_c2pa_trust_list": on_list,
        "on_interim_trust_list": (None if on_list != "no" or interim is None
                                  else "yes: on the older interim list of Content Credentials signers, "
                                       "which C2PA froze on 1 January 2026" if interim else "no"),
        "hidden_mark_named_in_label": ", ".join(marks) if marks else None,
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
    elif unchanged:
        summary = f"Signed by {signer}; unchanged since signing."
    else:
        summary = (f"A signed label naming {signer} is present, but markcheck could not confirm "
                   f"that the signature and the file still match it.")
    if ai_kind == "made":
        summary += " The label says it was made with AI."
    elif ai_kind == "edited":
        summary += " The label says AI-made material was mixed in."

    proves: list[str] = []
    not_proves: list[str] = [
        "That the content is accurate or that the events shown happened. "
        "A signature shows who signed the label, not that what it shows is correct.",
    ]
    if sig_bad:
        proves.append("The file carries a signed label, but the signature does not match it.")
        not_proves.append("Who made the label: with a broken signature the signer's name cannot be relied on.")
    elif not unchanged and not (file_changed or label_changed):
        proves.append("The file carries a signed label.")
        not_proves.append("Who signed it or that the file is unchanged since signing: markcheck could not "
                          "confirm the signature and the file's match with the label.")
    else:
        if sig_ok:
            proves.append(f"Whoever holds the signing key for the certificate named '{signer}' signed this label.")
        else:
            not_proves.append("Who signed it: markcheck could not confirm the signature.")
        if file_changed or label_changed:
            proves.append("The file or its label was changed after signing, so the label no longer "
                          "describes these exact bytes.")
            not_proves.append("What was changed or whether it matters. Even re-saving a file can do this.")
        else:
            proves.append("The file has not changed since it was signed.")
        if ai_kind:
            proves.append("The signer stated in the label that AI was used.")
    if on_list == "no" and interim:
        not_proves.append("That the signer passed the current C2PA checks. The certificate is not on the official "
                          "C2PA trust list. It is on the older interim list, which C2PA froze on 1 January 2026 "
                          "and asks checkers to keep apart from the official one.")
    elif on_list == "no":
        not_proves.append("That the signer is who the certificate name says. This certificate is not on the "
                          "official C2PA trust list, so anyone could have made a certificate with this name.")
    elif on_list != "yes":
        not_proves.append("That the signer is who the certificate name says. markcheck could not check this "
                          "certificate against the official C2PA trust list.")
    if marks:
        not_proves.append("That the hidden mark the label names is in the pixels. This line only repeats the "
                          "label's claim; markcheck's separate hidden-mark checks say whether a mark was read.")
    if self_signed or test_cert:
        not_proves.append("Anything about who actually made the file: this is a test or self-signed certificate, "
                          "which anyone can create.")
    if not ai_kind:
        not_proves.append("That no AI was used. The label only records what the signing tool chose to say.")
    if expired:
        not_proves.append("That the certificate was still in date when it was used.")

    return Finding(NAME, LABEL, "found", summary, details=details, proves=proves, does_not_prove=not_proves)
