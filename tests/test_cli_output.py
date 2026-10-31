"""Text and JSON output: plain, scannable, stable."""
import json
import os
import re
import shutil

import pytest

from markcheck.cli import main
from markcheck.detectors import trustmark_check

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
LABELLED = os.path.join(FIX, "c2pa", "ai_generated.jpg")
ORDINARY = os.path.join(FIX, "c2pa", "ordinary_edit.jpg")
PLAIN = os.path.join(FIX, "controls", "plain.jpg")
SD = os.path.join(FIX, "watermark", "sd_v1_marked.png")
TM = os.path.join(FIX, "watermark", "trustmark_q_marked.png")


def run(capsys, *args):
    code = main([*args])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


@pytest.fixture(scope="module")
def folder_text():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main([FIX])
    return buf.getvalue()


from markcheck.report import DETAIL_LABELS

_KEYS = [k for k in DETAIL_LABELS if "_" in k] + ["sd_watermark", "trustmark_check", "c2pa_check", "not_found"]
INTERNAL = re.compile(r"\bTrue\b|\bFalse\b|\bNone\b|\{'|\['|'\]|\}\s*$|\b(" + "|".join(_KEYS) + r")\b", re.M)


def test_text_has_no_python_reprs_or_internal_names(folder_text):
    bad = [l for l in folder_text.splitlines() if INTERNAL.search(l) and not l.startswith("==")]
    assert bad == []


def test_text_hides_raw_validation_codes(folder_text):
    assert "assertion.dataHash.mismatch" not in folder_text
    assert "claimSignature.mismatch" not in folder_text


def test_each_file_starts_with_short_verdict(capsys):
    _, out, _ = run(capsys, LABELLED)
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[0].startswith("==") and "ai_generated.jpg" in lines[0]
    assert lines[1].strip().startswith("Label found")
    _, out, _ = run(capsys, PLAIN)
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[1].strip() == "No label found. This does not prove it was made by a person."


def test_verdict_does_not_call_a_non_ai_label_ai(capsys):
    # ordinary_edit.jpg carries a signed label that does NOT say AI was used.
    _, out, _ = run(capsys, ORDINARY)
    verdict = [l for l in out.splitlines() if l.strip()][1]
    assert "AI" not in verdict


def test_jargon_explained_once_per_run(folder_text):
    for term in ("C2PA", "XMP", "trust list"):
        expl = re.findall(rf"^\s*\({re.escape(term)}[^)]*: ", folder_text, re.M | re.I)
        assert len(expl) == 1, term
    # the explanation comes right after the first use
    first_use = folder_text.find("C2PA")
    first_expl = re.search(r"^\s*\(C2PA", folder_text, re.M).start()
    assert first_use < first_expl
    assert folder_text[first_use:first_expl].count("\n") <= 1


def test_makers_section_printed_once_per_kind(folder_text):
    assert folder_text.count("Only the maker can check") == 1
    assert folder_text.count("synthid.com") == 1


def test_makers_kept_per_file_in_json(capsys):
    _, out, _ = run(capsys, FIX, "--json")
    doc = json.loads(out)
    images = [f for f in doc["files"] if f["kind"] == "image"]
    assert images and all(f["only_the_maker_can_check"] for f in images)


def test_skipped_checks_are_quiet(capsys):
    _, out, _ = run(capsys, os.path.join(FIX, "controls", "note.txt"))
    assert "[skipped]" not in out
    assert out.count("Only pictures can carry this mark") == 0
    assert "Not checked" in out


def test_trustmark_missing_models_told_once(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant="Q": ["decoder_Q.ckpt"])
    for i in range(3):
        shutil.copy(PLAIN, tmp_path / f"p{i}.jpg")
    _, out, _ = run(capsys, str(tmp_path))
    assert out.count("python -m markcheck.fetch_models") == 1


def test_sd_found_text_has_no_dict(capsys):
    _, out, _ = run(capsys, SD)
    assert "{" not in out and "(needs " not in out


def test_trustmark_found_hides_raw_bits(capsys):
    _, out, _ = run(capsys, TM)
    assert not re.search(r"[01]{20,}", out)


def test_json_is_stable(capsys):
    code, out, _ = run(capsys, FIX, "--json")
    doc = json.loads(out)
    assert set(doc) >= {"version", "files", "exit_code", "summary", "problems"}
    assert doc["exit_code"] == code == 1
    s = doc["summary"]
    assert s["files_checked"] == len(doc["files"])
    assert s["label_found_in"] + s["no_label_in"] == s["files_checked"]
    for f in doc["files"]:
        assert set(f) >= {"path", "kind", "label_found", "findings", "only_the_maker_can_check"}
        assert isinstance(f["label_found"], bool)
        for fd in f["findings"]:
            assert set(fd) >= {"detector", "label", "status", "summary", "details", "proves", "does_not_prove", "found"}


def test_json_on_error_still_valid(tmp_path, capsys):
    code, out, _ = run(capsys, str(tmp_path), "--json")
    doc = json.loads(out)
    assert code == 2 and doc["exit_code"] == 2 and doc["files"] == []


def test_json_missing_target_still_valid(tmp_path, capsys):
    code, out, err = run(capsys, str(tmp_path / "nope"), "--json")
    assert code == 2
    doc = json.loads(out)
    assert doc["exit_code"] == 2 and doc["problems"]


def test_failed_check_is_flagged_in_verdict(capsys, monkeypatch):
    from markcheck.detectors import sd_watermark

    def boom(*a):
        raise ValueError("bad pixels")

    monkeypatch.setattr(sd_watermark, "check", boom)
    code, out, _ = run(capsys, PLAIN)
    assert code == 2
    assert "not fully checked" in out
    assert "ValueError" not in out


def test_skipped_check_is_flagged_in_verdict(capsys, monkeypatch):
    """A check that did not run (here: TrustMark model files missing) must not leave the
    file looking fully checked: the verdict says so, and so does the JSON summary."""
    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant="Q": ["decoder_Q.ckpt"])
    code, out, _ = run(capsys, PLAIN)
    assert code == 0
    verdict = out.splitlines()[1:3]
    assert "No label found" in verdict[0]
    assert "not fully checked" in verdict[1], out
    code, out, _ = run(capsys, PLAIN, "--json")
    doc = json.loads(out)
    assert doc["summary"]["files_not_fully_checked"] == 1
    assert "not fully checked" in doc["summary"]["text"]
    assert doc["files"][0]["all_checks_ran"] is False


def test_not_applicable_checks_do_not_count_as_not_run(capsys):
    code, out, _ = run(capsys, os.path.join(FIX, "controls", "note.txt"), "--json")
    doc = json.loads(out)
    assert doc["files"][0]["all_checks_ran"] is True
    assert doc["summary"]["files_not_fully_checked"] == 0
