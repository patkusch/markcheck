"""Files labelled by real AI products, not by us. See tests/fixtures/realworld/SOURCES.md.

Each test here started out failing against a real file and was then fixed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from markcheck.cli import check_file
from markcheck.detectors import c2pa_check, trustmark_check
from markcheck.filetypes import sniff

RW = Path(__file__).parent / "fixtures" / "realworld"
MANIFEST = json.loads((RW / "manifest.json").read_text())["files"]
TEST_CERT_FILE = Path(__file__).parent / "fixtures" / "c2pa" / "ai_generated.jpg"


def c2pa_of(path: Path):
    data = path.read_bytes()
    return c2pa_check.inspect_bytes(path, data, sniff(path, data[:64]))


def test_every_realworld_file_has_a_source_entry():
    sources = (RW / "SOURCES.md").read_text()
    for e in MANIFEST:
        assert e["file"] in sources, e["file"]
        assert (RW / e["file"]).is_file()


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_realworld_file_matches_manifest(entry):
    rep = check_file(RW / entry["file"])
    found = {f.detector for f in rep.findings if f.found}
    expect = set(entry["expect"])
    if "trustmark" in expect and not trustmark_check.available():
        pytest.skip("TrustMark model files not installed")
    assert found == expect, [(f.detector, f.status, f.summary) for f in rep.findings]


def test_adobe_signer_on_interim_list_is_not_called_forgeable():
    """Photoshop + Firefly file signed by Adobe's production certificate. It is not on the
    new official C2PA list but is on the older interim list (frozen 1 January 2026).
    markcheck used to say 'anyone could have made a certificate with this name'."""
    f = c2pa_of(RW / "adobe_photoshop_firefly_ghost.png")
    assert f.details["signed_by"].startswith("Adobe C2PA")
    assert f.details["software_or_ai_named"] == "Adobe Firefly"
    assert f.details["says_made_with_ai"].startswith("yes")
    assert f.details["on_c2pa_trust_list"] == "no"
    assert f.details["on_interim_trust_list"].startswith("yes")
    assert not any("anyone could have made" in s for s in f.does_not_prove)
    assert any("interim" in s for s in f.does_not_prove)


def test_test_certificate_is_not_on_interim_list_either():
    f = c2pa_of(TEST_CERT_FILE)
    assert f.details["on_interim_trust_list"] == "no"
    assert any("anyone could have made" in s for s in f.does_not_prove)


def test_label_naming_a_hidden_mark_is_shown():
    """The label carries a 'soft binding': it says a TrustMark (variant Q) mark was put in the
    pixels. markcheck used to drop that line."""
    f = c2pa_of(RW / "adobe_trustmark_softbinding_signed.jpg")
    assert "TrustMark" in f.details["hidden_mark_named_in_label"]
    assert "Q" in f.details["hidden_mark_named_in_label"]
    assert any("hidden mark" in s for s in f.does_not_prove)


def test_trustmark_not_found_says_which_variant_was_checked():
    """Adobe's own samples include variant P marks, which a Q reader cannot see.
    'No TrustMark mark found' alone overstated what was checked."""
    if not trustmark_check.available():
        pytest.skip("TrustMark model files not installed")
    rep = check_file(RW / "adobe_trustmark_unmarked_ufo.jpg")
    f = next(x for x in rep.findings if x.detector == "trustmark")
    assert f.status == "not_found"
    assert "variant Q" in f.summary
    assert "P" in f.summary


def test_interim_list_has_licence_note():
    readme = (c2pa_check.TRUST_LIST.parent / "README.md").read_text()
    assert "INTERIM-TRUST-ANCHORS.pem" in readme and "Apache" in readme


def test_automatic1111_settings_name_the_model():
    """A real SDXL 1.0 output from AUTOMATIC1111. Its settings text names the model
    (sd_xl_base_1.0) and the tool version; markcheck used to show only 'AUTOMATIC1111-style'."""
    rep = check_file(RW / "sdxl_automatic1111_ostris.png")
    f = next(x for x in rep.findings if x.detector == "metadata" and x.found)
    assert f.details["model named"] == "sd_xl_base_1.0"
    assert "sd_xl_base_1.0" in f.summary


def test_automatic1111_sdxl_output_has_no_sd_mark():
    """AUTOMATIC1111 does not add Stability's hidden mark (only the diffusers library does),
    so a real SDXL file from it must not be reported as marked."""
    rep = check_file(RW / "sdxl_automatic1111_ostris.png")
    assert next(x for x in rep.findings if x.detector == "sd_watermark").status == "not_found"


def test_openai_label_signed_with_c2pa_claim_signing_eku_checks_out():
    """A real gpt-image file from Wikimedia Commons, signed by 'OpenAI Media Service' with a
    certificate from Trufo's C2PA CA. That certificate uses the C2PA claim-signing purpose
    (EKU 1.3.6.1.4.1.62558.2.1), which the library does not accept unless told to. markcheck
    used to say the signature 'does not check out' and the signer is not on the trust list."""
    f = c2pa_of(RW / "openai_gpt_image_commons_map.png")
    assert f.details["signed_by"].startswith("OpenAI Media Service")
    assert f.details["signature_valid"] == "yes"
    assert f.details["changed_since_signing"] == "no"
    assert f.details["on_c2pa_trust_list"] == "yes"
    assert "gpt-image" in f.details["software_or_ai_named"]
    assert f.details["says_made_with_ai"].startswith("yes")
    assert "does not check out" not in f.summary


def test_measure_counts_real_files_separately():
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure", Path(__file__).parents[1] / "scripts" / "measure.py")
    m = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["measure"] = m  # dataclasses look the module up while it loads
    spec.loader.exec_module(m)
    res = m.measure(Path(__file__).parent / "fixtures")
    real = res["real_files"]
    assert real["hits"]["c2pa"]["labelled"] >= 4
    assert real["hits"]["c2pa"]["hits"] == real["hits"]["c2pa"]["labelled"]
    assert real["survival"]["c2pa"]["original"]["of"] >= 3
    assert "Files labelled by real AI products only" in m.tables(res)


def test_trust_list_answer_is_not_contradicted():
    """A signer reported as 'not on the list' must not also be reported as 'could not check'."""
    from markcheck.cli import check_file
    for name in ("adobe_photoshop_firefly_ghost.png", "openai_gpt_image_commons_map.png"):
        rep = check_file(FIX / name) if "FIX" in globals() else check_file(
            __import__("pathlib").Path(__file__).parent / "fixtures" / "realworld" / name)
        for f in rep.findings:
            if f.detector == "c2pa" and f.found:
                on = f.details.get("on_c2pa_trust_list", "")
                said_unchecked = any("could not check this certificate" in s for s in f.does_not_prove)
                if str(on).startswith(("yes", "no")):
                    assert not said_unchecked, (name, on)
