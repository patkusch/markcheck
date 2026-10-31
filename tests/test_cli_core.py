import json

from markcheck.cli import main


def test_missing_path_is_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope")]) == 2


def test_plain_text_no_label(tmp_path, capsys):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "does not prove it was made by a person" in out


def test_json_shape(tmp_path, capsys):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    main([str(p), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["files"][0]["label_found"] is False
