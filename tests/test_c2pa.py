"""Signed label (C2PA) check. Fully offline: every file used here is committed."""
from __future__ import annotations

import io
import json
import re
import socket
import sys
import threading
from pathlib import Path

import pytest

from markcheck.cli import check_file, main
from markcheck.detectors import c2pa_check
from markcheck.filetypes import sniff

FIX = Path(__file__).parent / "fixtures"
C2PA_DIR = FIX / "c2pa"
MANIFEST = json.loads((C2PA_DIR / "manifest.json").read_text())["files"]
TEST_ROOT = (C2PA_DIR / "test-root-ca.pem").read_text()

BANNED = re.compile(r"\b(authentic|genuine|human-made|made by a human|not ai|true)\b", re.I)


def run(path: Path, extra_anchors: str | None = None):
    data = path.read_bytes()
    return c2pa_check.inspect_bytes(path, data, sniff(path, data[:64]), extra_anchors)


def run_bytes(name: str, data: bytes):
    p = Path(name)
    return c2pa_check.inspect_bytes(p, data, sniff(p, data[:64]))


# ---------- our own signed fixtures ----------

@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_fixture_matches_manifest(entry):
    f = run(C2PA_DIR / entry["file"])
    if "c2pa" in entry["expect"]:
        assert f.status == "found", f.summary
        assert f.details["says_made_with_ai"].startswith("yes" if entry["ai"] else "no")
        assert f.details["signed_by"].startswith("markcheck Test Signer")
        assert f.details["certificate_issued_by"].startswith("markcheck Test Root CA")
        assert f.details["on_c2pa_trust_list"] == "no"
        assert f.details["test_or_self_signed_certificate"] == "yes"
    else:
        assert f.status == "not_found"


def test_ai_label_names_model_and_tool():
    f = run(C2PA_DIR / "ai_generated.jpg")
    assert "markcheck-test-model" in f.details["software_or_ai_named"]
    assert "markcheck fixture maker" in f.details["made_with_tool"]
    assert f.details["source_type"] == "made by an AI model"
    assert f.details["signature_valid"] == "yes"
    assert f.details["changed_since_signing"] == "no"
    assert f.summary.startswith("Signed by markcheck Test Signer")
    assert "unchanged since signing" in f.summary
    assert "made with AI" in f.summary


def test_ordinary_edit_not_labelled_ai():
    f = run(C2PA_DIR / "ordinary_edit.jpg")
    assert f.status == "found"
    assert f.details["says_made_with_ai"].startswith("no")
    assert "AI" not in f.summary
    assert any("no AI was used" in s for s in f.does_not_prove)
    assert "earlier_label_says_ai" not in f.details


def test_changed_after_signing_is_reported():
    f = run(C2PA_DIR / "changed_after_signing.jpg")
    assert f.status == "found"
    assert f.details["changed_since_signing"].startswith("yes")
    assert "changed after signing" in f.summary
    assert "unchanged" not in f.summary
    assert "assertion.dataHash.mismatch" in f.details["problems"]
    assert not any("has not changed" in s for s in f.proves)


def test_test_certificate_is_not_on_trust_list_and_says_anyone_could():
    f = run(C2PA_DIR / "ai_generated.png")
    assert f.details["on_c2pa_trust_list"] == "no"
    assert any("anyone" in s for s in f.does_not_prove)
    assert any("not that what it shows is correct" in s for s in f.does_not_prove)


def test_trust_check_really_runs():
    """Same file, but with our test root added as a trust anchor: now it is on the list.
    This shows the 'no' above comes from a real chain check, not a fixed answer."""
    assert c2pa_check.TRUST_LIST.is_file()
    f = run(C2PA_DIR / "ai_generated.png", extra_anchors=TEST_ROOT)
    assert f.details["on_c2pa_trust_list"] == "yes"


def test_bundled_trust_list_has_licence_note():
    readme = (c2pa_check.TRUST_LIST.parent / "README.md").read_text()
    assert "CC BY 4.0" in readme
    assert c2pa_check.TRUST_LIST.read_text().count("BEGIN CERTIFICATE") > 10


# ---------- public C2PA test files ----------

PUBLIC = [
    ("adobe-20220124-A.jpg", "not_found", None, None),
    ("adobe-20220124-CA.jpg", "found", "yes", "no"),
    ("adobe-20220124-E-dat-CA.jpg", "found", "yes", "yes: the file"),
    ("adobe-20220124-E-sig-CA.jpg", "found", "no", "cannot tell"),
    ("adobe-20220124-E-uri-CA.jpg", "found", "yes", "yes: parts of the label"),
]


@pytest.mark.parametrize("name,status,sig,changed", PUBLIC, ids=[p[0] for p in PUBLIC])
def test_public_files(name, status, sig, changed):
    f = run(C2PA_DIR / "public" / name)
    assert f.status == status
    if status == "found":
        assert f.details["signature_valid"] == sig
        assert f.details["changed_since_signing"].startswith(changed)
        assert f.details["on_c2pa_trust_list"] == "no"


def test_bad_signature_does_not_vouch_for_signer():
    f = run(C2PA_DIR / "public" / "adobe-20220124-E-sig-CA.jpg")
    assert "does not check out" in f.summary
    assert not any(s.startswith("Whoever holds") for s in f.proves)


# ---------- controls and odd files ----------

CONTROLS = sorted(p for d in (FIX / "controls", C2PA_DIR / "controls") if d.is_dir()
                  for p in d.iterdir() if p.is_file() and not p.name.startswith(".")
                  and p.suffix.lower() not in {".json", ".md"})


@pytest.mark.parametrize("path", CONTROLS, ids=[p.name for p in CONTROLS])
def test_controls_have_no_label(path):
    assert run(path).status in ("not_found", "skipped")


def test_text_file_is_skipped():
    f = run_bytes("note.txt", b"just some words")
    assert f.status == "skipped"


@pytest.mark.parametrize("name,data", [
    ("garbage.jpg", b"\xff\xd8\xff\xe0" + b"garbage" * 50),
    ("random.png", bytes(range(256)) * 4),
    ("empty.jpg", b""),
    ("fake.png", b"\x89PNG\r\n\x1a\n" + b"\0" * 50),
], ids=["garbage-jpeg", "random-bytes", "empty", "broken-png"])
def test_garbage_is_not_an_error(name, data):
    f = run_bytes(name, data)
    assert f.status in ("not_found", "skipped"), f.summary


def test_truncated_signed_file_reports_damaged_label():
    data = (C2PA_DIR / "ai_generated.jpg").read_bytes()[:30000]
    f = run_bytes("cut.jpg", data)
    assert f.status == "found"
    assert "damaged" in f.summary
    assert f.details["label_readable"] == "no"


# ---------- never goes online ----------

def test_offline_settings():
    assert c2pa_check.OFFLINE_VERIFY == {"remote_manifest_fetch": False, "ocsp_fetch": False}


def test_label_stored_online_is_not_fetched(tmp_path):
    """Sign a file whose label lives at a URL on a local server we control,
    then check it. The server must receive no request at all."""
    import c2pa
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import make_fixtures_c2pa as mk

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(0.2)
    port = srv.getsockname()[1]
    hits: list[bytes] = []
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            hits.append(conn.recv(200))
            conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        chain, key, _ = mk.make_test_certs()
        signer = c2pa.Signer.from_info(
            c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, chain.encode(), key.encode(), None))
        builder = c2pa.Builder(mk.ai_manifest("remote.jpg"), c2pa.Context.from_dict({"verify": mk.OFFLINE["verify"]}))
        builder.set_no_embed()
        builder.set_remote_url(f"http://127.0.0.1:{port}/label.c2pa")
        out = io.BytesIO()
        builder.sign(signer, "image/jpeg", io.BytesIO((C2PA_DIR / "controls" / "unsigned.jpg").read_bytes()), out)
        p = tmp_path / "remote.jpg"
        p.write_bytes(out.getvalue())

        f = run(p)
        assert f.status == "found"
        assert "online" in f.summary and "never goes online" in f.summary
        assert f.details["signature_valid"] == "not checked"
    finally:
        stop.set()
        t.join(2)
        srv.close()
    assert hits == [], f"markcheck contacted the network: {hits!r}"


# ---------- wording and CLI ----------

ALL_FILES = [C2PA_DIR / e["file"] for e in MANIFEST] + [C2PA_DIR / "public" / p[0] for p in PUBLIC]


@pytest.mark.parametrize("path", ALL_FILES, ids=[p.name for p in ALL_FILES])
def test_wording_rules(path):
    f = run(path)
    for text in [f.summary, *f.proves, *f.does_not_prove, *map(str, f.details.values())]:
        assert not BANNED.search(text), text


def test_cli_exit_codes(capsys):
    assert main([str(C2PA_DIR / "ai_generated.jpg")]) == 1
    out = capsys.readouterr().out
    assert "Signed label (C2PA)" in out and "unchanged since signing" in out
    assert main([str(C2PA_DIR / "controls" / "unsigned.jpg")]) == 0


def test_cli_json_has_c2pa_details(capsys):
    main([str(C2PA_DIR / "changed_after_signing.jpg"), "--json"])
    doc = json.loads(capsys.readouterr().out)
    c = next(f for f in doc["files"][0]["findings"] if f["detector"] == "c2pa")
    assert c["found"] is True
    assert c["details"]["changed_since_signing"].startswith("yes")


def test_check_file_runs_all_detectors_without_error():
    rep = check_file(C2PA_DIR / "ai_generated.png")
    assert rep.any_found


# ---------- round 3: claims must match what was actually checked ----------

def _store(path: Path, extra_anchors: str | None = None) -> tuple[dict, dict]:
    import c2pa

    data = path.read_bytes()
    r = c2pa.Reader.try_create(c2pa_check._mime(path), io.BytesIO(data), None,
                               c2pa_check._context(extra_anchors))
    with r:
        return json.loads(r.json()), json.loads(r.crjson())


def test_trusted_signer_on_changed_file_is_not_called_untrusted():
    """A file changed after signing makes the overall state 'Invalid', but the signer
    can still be on the trust list. markcheck must not then say it is not on the list."""
    f = run(C2PA_DIR / "changed_after_signing.jpg", extra_anchors=TEST_ROOT)
    assert f.details["changed_since_signing"].startswith("yes")
    assert f.details["on_c2pa_trust_list"] == "yes"
    assert not any("not on the official C2PA trust list" in s for s in f.does_not_prove)


def test_trust_list_not_decided_is_not_called_not_on_list():
    store, cr = _store(C2PA_DIR / "ai_generated.jpg")
    res = store["validation_results"]["activeManifest"]
    res["failure"] = [c for c in res["failure"] if c["code"] != "signingCredential.untrusted"]
    store["validation_state"] = "Valid"
    f = c2pa_check._explain(store, cr, trust_checked=True)
    assert f.details["on_c2pa_trust_list"] != "no"
    assert not any("is not on the" in s for s in f.does_not_prove)


def test_unchanged_needs_positive_hash_and_signature_checks():
    """'Unchanged since signing' only when the library reported a matching file hash
    and a validated signature, not merely when it reported no failure."""
    store, cr = _store(C2PA_DIR / "ai_generated.jpg")
    store["validation_results"]["activeManifest"]["success"] = []
    f = c2pa_check._explain(store, cr, trust_checked=True)
    assert "unchanged since signing" not in f.summary
    assert f.details["changed_since_signing"] != "no"
    assert f.details["signature_valid"] != "yes"
    assert "The file has not changed since it was signed." not in f.proves


def test_unknown_failure_code_blocks_unchanged():
    store, cr = _store(C2PA_DIR / "ai_generated.jpg")
    store["validation_results"]["activeManifest"]["failure"].append(
        {"code": "general.error", "explanation": "something went wrong"})
    f = c2pa_check._explain(store, cr, trust_checked=True)
    assert "unchanged since signing" not in f.summary
    assert f.details["changed_since_signing"] != "no"


def test_changed_file_without_validated_signature_does_not_vouch_for_signer():
    store, cr = _store(C2PA_DIR / "changed_after_signing.jpg")
    res = store["validation_results"]["activeManifest"]
    res["success"] = [c for c in res["success"] if not c["code"].startswith("claimSignature.")]
    f = c2pa_check._explain(store, cr, trust_checked=True)
    assert f.details["signature_valid"] != "yes"
    assert not any(s.startswith("Whoever holds") for s in f.proves)
