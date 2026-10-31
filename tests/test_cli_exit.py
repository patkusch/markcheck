"""Exit codes: 0 = no label found, 1 = label found, 2 = error. And folder walking."""
import os
import shutil
import sys

import pytest

from markcheck import cli
from markcheck.cli import main

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
LABELLED = os.path.join(FIX, "c2pa", "ai_generated.jpg")
PLAIN = os.path.join(FIX, "controls", "plain.jpg")


def test_labelled_file_exits_1(capsys):
    assert main([LABELLED]) == 1


def test_plain_file_exits_0(capsys):
    assert main([PLAIN]) == 0


def test_empty_folder_is_error_and_says_so(tmp_path, capsys):
    assert main([str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "no files" in err.lower()


def test_folder_of_only_hidden_files_is_error(tmp_path, capsys):
    (tmp_path / ".DS_Store").write_bytes(b"x")
    assert main([str(tmp_path)]) == 2


def test_mixed_folder_with_label_exits_1(tmp_path, capsys):
    shutil.copy(LABELLED, tmp_path / "a.jpg")
    shutil.copy(PLAIN, tmp_path / "b.jpg")
    assert main([str(tmp_path)]) == 1


def test_mixed_folder_without_label_exits_0(tmp_path, capsys):
    shutil.copy(PLAIN, tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "data.bin").write_bytes(os.urandom(5000))
    assert main([str(tmp_path)]) == 0


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, non-root")
def test_unreadable_file_is_error(tmp_path, capsys):
    p = tmp_path / "locked.jpg"
    shutil.copy(PLAIN, p)
    p.chmod(0)
    try:
        assert main([str(tmp_path)]) == 2
        err = capsys.readouterr().err
        assert "locked.jpg" in err
        assert "Errno" not in err and "PermissionError" not in err
    finally:
        p.chmod(0o644)


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, non-root")
def test_unreadable_subfolder_is_reported_not_silently_skipped(tmp_path, capsys):
    shutil.copy(PLAIN, tmp_path / "ok.jpg")
    sub = tmp_path / "private"
    sub.mkdir()
    shutil.copy(PLAIN, sub / "x.jpg")
    sub.chmod(0)
    try:
        assert main([str(tmp_path)]) == 2
        assert "private" in capsys.readouterr().err
    finally:
        sub.chmod(0o755)


def test_unreadable_file_with_label_elsewhere_still_exits_1(tmp_path, capsys, monkeypatch):
    shutil.copy(LABELLED, tmp_path / "a.jpg")
    shutil.copy(PLAIN, tmp_path / "b.jpg")
    real = cli.check_file

    def boom(path, *a, **k):
        if path.name == "b.jpg":
            raise PermissionError(13, "Permission denied")
        return real(path, *a, **k)

    monkeypatch.setattr(cli, "check_file", boom)
    assert main([str(tmp_path)]) == 1
    assert "b.jpg" in capsys.readouterr().err


def test_broken_symlink_target_is_error_with_clear_message(tmp_path, capsys):
    link = tmp_path / "link.jpg"
    link.symlink_to(tmp_path / "gone.jpg")
    assert main([str(link)]) == 2
    err = capsys.readouterr().err.lower()
    assert "link" in err and "missing" in err


def test_broken_symlink_inside_folder_is_reported(tmp_path, capsys):
    shutil.copy(PLAIN, tmp_path / "ok.jpg")
    (tmp_path / "dead.jpg").symlink_to(tmp_path / "gone.jpg")
    assert main([str(tmp_path)]) == 2
    assert "dead.jpg" in capsys.readouterr().err


def test_symlink_loop_does_not_hang(tmp_path, capsys):
    shutil.copy(PLAIN, tmp_path / "ok.jpg")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.count("ok.jpg") == 1


def test_hidden_and_tool_folders_are_skipped(tmp_path, capsys):
    for d in (".git", ".venv", ".cache", "node_modules"):
        (tmp_path / d).mkdir()
        shutil.copy(LABELLED, tmp_path / d / "a.jpg")
    shutil.copy(LABELLED, tmp_path / ".hidden.jpg")
    shutil.copy(PLAIN, tmp_path / "plain.jpg")
    assert main([str(tmp_path)]) == 0


def test_folder_inside_a_folder_named_venv_is_still_scanned(tmp_path, capsys):
    # Skipping must look at the path below the target, not the whole path.
    root = tmp_path / "venv" / "photos"
    root.mkdir(parents=True)
    shutil.copy(LABELLED, root / "a.jpg")
    assert main([str(root)]) == 1


def test_target_inside_hidden_folder_is_still_scanned(tmp_path, capsys):
    root = tmp_path / ".stuff" / "photos"
    root.mkdir(parents=True)
    shutil.copy(LABELLED, root / "a.jpg")
    assert main([str(root)]) == 1


def test_explicit_hidden_file_is_checked(tmp_path, capsys):
    p = tmp_path / ".x.jpg"
    shutil.copy(LABELLED, p)
    assert main([str(p)]) == 1


def test_fifo_target_is_error(tmp_path, capsys):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no fifo")
    f = tmp_path / "pipe"
    os.mkfifo(f)
    assert main([str(f)]) == 2
    assert "not a file" in capsys.readouterr().err.lower()


def test_large_non_media_file_is_not_read_whole(tmp_path, capsys, monkeypatch):
    big = tmp_path / "big.bin"
    with open(big, "wb") as fh:
        fh.truncate(200 * 1024 * 1024)  # sparse: cheap on disk
    seen = {}
    real = cli.run_all

    def spy(path, data, kind, options=None):
        seen["len"] = len(data)
        return real(path, data, kind, options)

    monkeypatch.setattr(cli, "run_all", spy)
    assert main([str(big)]) == 0
    assert seen["len"] <= cli.MAX_NON_MEDIA_BYTES
    assert "first" in capsys.readouterr().out.lower()


def test_not_media_files_are_fine(tmp_path, capsys):
    (tmp_path / "a.csv").write_text("a,b\n1,2\n")
    (tmp_path / "empty.jpg").write_bytes(b"")
    (tmp_path / "notapicture.png").write_text("this is not a picture")
    (tmp_path / "x.docx").write_bytes(b"PK\x03\x04" + os.urandom(100))
    code = main([str(tmp_path)])
    assert code == 0
