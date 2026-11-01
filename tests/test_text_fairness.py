"""scripts/text_fairness.py helpers: fast, offline (no data, no models)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
tf = pytest.importorskip("text_fairness")


def test_wilson_known_values():
    lo, hi = tf.wilson(2, 63)
    assert lo == pytest.approx(0.0087, abs=1e-3) and hi == pytest.approx(0.109, abs=1e-3)
    assert tf.wilson(0, 63)[0] == 0.0
    assert tf.wilson(63, 63)[1] == 1.0


@pytest.mark.parametrize("row,status", [
    ({"split": "toefl"}, "non-native"),
    ({"split": "us_students"}, "native"),
    ({"split": "x", "group": "ELL"}, "non-native"),
    ({"split": "x", "group": "non_ell"}, "native"),
    ({"split": "x", "group": "non-ELL grade 8"}, "native"),
    ({"split": "x", "group": "non-native"}, "non-native"),
    ({"split": "x", "group": "native speakers"}, "native"),
    ({"split": "x", "group": "spelling bee"}, None),
    ({"split": "x"}, None),
])
def test_native_status(row, status):
    assert tf.native_status(row) == status


def test_rate_shape():
    r = tf.rate(3, 100)
    assert r["k"] == 3 and r["n"] == 100 and r["rate"] == 0.03 and len(r["ci95"]) == 2
    assert tf.rate(0, 0)["rate"] is None


def test_altered_sets_other_than_rewording_are_left_out():
    assert tf.ai_set_name({"split": "raid", "set": "raid_homoglyph", "attack": "homoglyph"}) is None
    assert tf.ai_set_name({"split": "raid", "set": "raid_paraphrase", "attack": "paraphrase"})
    assert tf.ai_set_name({"split": "raid", "set": "raid_reworded", "attack": "paraphrase"})
    assert tf.ai_set_name({"split": "mage", "set": "mage"}) == "GPT-4 (MAGE)"
