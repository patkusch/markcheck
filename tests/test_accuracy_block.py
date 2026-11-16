"""The README accuracy block's helpers, on small made-up calibration data (the shipped file is checked elsewhere)."""
import pytest

from markcheck.textcheck import accuracy
from markcheck.textcheck.accuracy import END, START

CAL = {
    "uncertain_band": [0.4, 0.6],
    "min_words": 100,
    "heldout": {"how_often_wrong": {"people called AI": "3 in 100", "AI called people": "9 in 100", "nested": {"inner": "1 in 100", "flag": True}, "skip": None}},
}


def rate(k, n, r, lo=None, hi=None):
    out = {"k": k, "n": n, "rate": r}
    if lo is not None:
        out["ci95"] = [lo, hi]
    return out


def test_small_chances_keep_a_decimal_so_they_do_not_round_to_zero():
    assert accuracy._in_100(0) == "0"
    assert accuracy._in_100(0.004) == "0.4"
    assert accuracy._in_100(0.0099) == "1.0"
    assert accuracy._in_100(0.01) == "1"
    assert accuracy._in_100(0.125) == "12"
    assert accuracy._in_100(1) == "100"


def test_a_rate_reads_as_chances_in_a_hundred_with_its_range_when_there_is_one():
    assert accuracy.rate_words(rate(3, 100, 0.03, 0.01, 0.08)) == "3 in 100 (likely 1 to 8)"
    assert accuracy.rate_words(rate(3, 100, 0.03)) == "3 in 100"
    assert accuracy.rate_words(rate(0, 50, 0)) == "0 in 100"


def test_nothing_to_show_is_a_dash_not_a_made_up_number():
    for missing in (None, {}, {"n": 0, "rate": 0.1}, {"n": 10}, "text", 5):
        assert accuracy.rate_words(missing) == "-", missing


def test_the_how_often_rows_flatten_nested_groups_and_skip_flags_and_empties():
    assert accuracy.how_often_rows(CAL) == [
        ("people called AI", "3 in 100"),
        ("AI called people", "9 in 100"),
        ("nested: inner", "1 in 100"),
    ]
    assert accuracy.how_often_rows({}) == []
    assert accuracy.how_often_rows({"heldout": {}}) == []


def test_a_bar_inside_a_cell_cannot_break_the_table():
    assert accuracy._cell("a|b") == "a/b"


def test_with_nothing_measured_the_block_says_so_instead_of_an_empty_table():
    block = accuracy.render({"uncertain_band": [0.4, 0.6], "min_words": 50})
    assert block.startswith(START) and block.endswith(END)
    assert "| Not measured yet. | - |" in block
    assert "Fairness" not in block


def test_the_block_names_the_band_and_the_minimum_length():
    block = accuracy.render(CAL)
    assert "Texts under 100 words get no number." in block
    assert "| people called AI | 3 in 100 |" in block


def test_the_fairness_table_lists_each_group_and_the_pooled_ones_with_their_text_counts():
    cal = {**CAL, "fairness": {
        "groups": {"Group A": {"flagged": rate(2, 40, 0.05, 0.01, 0.17), "cant_tell": rate(4, 40, 0.1)}, "odd": "not a dict"},
        "pooled": {"native": {"flagged": rate(1, 100, 0.01), "cant_tell": rate(5, 100, 0.05)}},
    }}
    block = accuracy.render(cal)
    assert "| Group A | 40 | 5 in 100 (likely 1 to 17) | 10 in 100 |" in block
    assert "| All native writers together | 100 | 1 in 100 | 5 in 100 |" in block
    assert "odd" not in block


def test_the_trade_off_table_has_one_column_per_ai_set_and_one_row_per_band_top():
    cal = {**CAL, "fairness": {"band_tradeoff": [
        {"band_top": 0.6, "non_native_flagged": rate(3, 50, 0.06), "native_flagged": rate(1, 50, 0.02), "ai_caught": {"set one": rate(40, 50, 0.8), "set two": rate(30, 50, 0.6)}},
        {"band_top": 0.7, "non_native_flagged": rate(1, 50, 0.02), "native_flagged": rate(0, 50, 0), "ai_caught": {"set one": rate(35, 50, 0.7)}},
        {"no_band_top": True},
    ]}}
    block = accuracy.render(cal)
    assert "AI caught: set one | AI caught: set two |" in block
    rows = [l for l in block.splitlines() if l.startswith("| 60") or l.startswith("| 70")]
    assert len(rows) == 2
    assert rows[1].endswith("| 70 in 100 | - |"), "a missing measurement is a dash"


def test_replacing_the_block_changes_only_what_is_between_the_markers():
    readme = f"before\n{START}\nold numbers\n{END}\nafter\n"
    out = accuracy.replace_block(readme, f"{START}\nnew\n{END}")
    assert out == f"before\n{START}\nnew\n{END}\nafter\n"
    assert accuracy.replace_block(out, f"{START}\nnew\n{END}") == out, "doing it twice changes nothing"


def test_a_replacement_with_backslashes_or_group_references_is_inserted_as_written():
    out = accuracy.replace_block(f"{START}\nx\n{END}", f"{START}\n\\1 \\g<0> C:\\new\n{END}")
    assert "\\1 \\g<0> C:\\new" in out


def test_a_readme_without_the_markers_is_refused_rather_than_guessed_at():
    with pytest.raises(ValueError, match="no text-accuracy block markers"):
        accuracy.replace_block("# README with no markers\n", "x")


def test_the_command_prints_the_block_and_with_write_readme_updates_the_file(tmp_path, monkeypatch, capsys):
    target = tmp_path / "README.md"
    target.write_text(f"intro\n{START}\nold\n{END}\noutro\n", encoding="utf-8")
    monkeypatch.setattr(accuracy, "readme_path", lambda: target)
    monkeypatch.setattr(accuracy, "load_calibration", lambda: CAL)
    assert accuracy.main([]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith(START) and "old" not in printed
    assert target.read_text(encoding="utf-8").count("old") == 1, "printing alone does not touch the file"
    assert accuracy.main(["--write-readme"]) == 0
    assert f"updated {target}" in capsys.readouterr().out
    text = target.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n") and "old" not in text
    assert "| people called AI | 3 in 100 |" in text


def test_the_readme_it_updates_is_the_one_at_the_top_of_the_project():
    assert accuracy.readme_path().name == "README.md"
    assert accuracy.readme_path().exists()
