"""The download helper, with the network and the model libraries replaced by stand-ins.

Nothing here goes online or loads a model; the helper is the only part of markcheck that
is allowed to fetch anything, so what it does before and after a fetch is worth pinning.
"""
import sys
import types

import pytest

from markcheck import fetch_models
from markcheck.detectors import trustmark_check
from markcheck.textcheck import load_calibration

NAMES = load_calibration()["models"]


def fake_hub(monkeypatch, download):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_text_models_are_fetched_once_each_with_only_the_files_the_check_needs(monkeypatch, capsys):
    seen = []

    def download(repo_id, allow_patterns, ignore_patterns):
        seen.append((repo_id, allow_patterns, ignore_patterns))
        return f"/cache/{repo_id}"

    fake_hub(monkeypatch, download)
    assert fetch_models.main(["--text"]) == 0
    assert [s[0] for s in seen] == NAMES
    assert all(s[1] == fetch_models.TEXT_ALLOW and s[2] == fetch_models.TEXT_IGNORE for s in seen)
    out = capsys.readouterr().out
    assert "about 0.7 GB per model" in out
    assert f"{NAMES[0]}: saved in /cache/{NAMES[0]}" in out
    assert "Done. markcheck text can now run without going online." in out


def test_one_failed_download_is_reported_and_does_not_stop_the_other(monkeypatch, capsys):
    def download(repo_id, allow_patterns, ignore_patterns):
        if repo_id == NAMES[0]:
            raise OSError("disk full")
        return f"/cache/{repo_id}"

    fake_hub(monkeypatch, download)
    assert fetch_models.main(["--text"]) == 2
    out = capsys.readouterr().out
    assert f"Could not fetch {NAMES[0]}: disk full" in out
    assert f"{NAMES[1]}: saved in /cache/{NAMES[1]}" in out
    assert f"Some models could not be fetched: {NAMES[0]}" in out
    assert "Done." not in out


def test_text_fetch_says_how_to_install_the_libraries_when_they_are_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # makes the import fail
    assert fetch_models.main(["--text"]) == 2
    assert "pip install 'markcheck[text]'" in capsys.readouterr().out


def test_trustmark_fetch_says_how_to_install_the_library_when_it_is_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "trustmark", None)
    assert fetch_models.main([]) == 2
    assert "pip install 'markcheck[trustmark]'" in capsys.readouterr().out


class FakeTrustMark:
    created = []

    def __init__(self, **kwargs):
        FakeTrustMark.created.append(kwargs)


def fake_trustmark(monkeypatch):
    FakeTrustMark.created = []
    mod = types.ModuleType("trustmark")
    mod.TrustMark = FakeTrustMark
    monkeypatch.setitem(sys.modules, "trustmark", mod)


def test_nothing_is_downloaded_when_the_files_are_already_there(monkeypatch, capsys):
    fake_trustmark(monkeypatch)
    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant=trustmark_check.VARIANT: [])
    assert fetch_models.main([]) == 0
    assert "already on this computer. Nothing to do." in capsys.readouterr().out
    assert FakeTrustMark.created == []


def test_missing_files_are_fetched_through_the_library_on_the_cpu_without_the_remover(monkeypatch, capsys):
    fake_trustmark(monkeypatch)
    answers = iter([["encoder.ckpt", "decoder.ckpt"], []])
    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant=trustmark_check.VARIANT: next(answers))
    assert fetch_models.main([]) == 0
    out = capsys.readouterr().out
    assert "Fetching TrustMark model files (variant Q) from Adobe: encoder.ckpt, decoder.ckpt" in out
    assert "Done. markcheck can now check for TrustMark without going online." in out
    assert FakeTrustMark.created == [{"verbose": False, "model_type": "Q", "device": "cpu", "loadRemover": False}]


def test_a_fetch_that_leaves_files_missing_is_a_failure_with_the_names(monkeypatch, capsys):
    fake_trustmark(monkeypatch)
    answers = iter([["a.ckpt", "b.ckpt"], ["b.ckpt"]])
    monkeypatch.setattr(trustmark_check, "missing_files", lambda variant=trustmark_check.VARIANT: next(answers))
    assert fetch_models.main([]) == 2
    out = capsys.readouterr().out
    assert "Some files could not be fetched: b.ckpt" in out
    assert "Done." not in out


def test_help_names_the_text_option(capsys):
    with pytest.raises(SystemExit) as e:
        fetch_models.main(["--help"])
    assert e.value.code == 0
    assert "--text" in capsys.readouterr().out
