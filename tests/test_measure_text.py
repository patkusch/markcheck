"""scripts/measure_text.py helpers: fast, offline (no data, no models)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
mt = pytest.importorskip("measure_text")


def row(**k):
    r = {"split": "x", "label": 1, "text": "t", "model": "m", "kind": "news", "attack": "none",
         "doc_id": "d1", "source": "raid"}
    r.update(k)
    return r


def test_same_model_one_name():
    assert mt.norm_model("gpt4") == mt.norm_model("gpt-4") == "gpt-4"
    assert mt.norm_model(None) == "unknown"


def test_reworded_sets_are_named_plainly():
    out = mt.assign([row(attack="paraphrase", doc_id=str(i)) for i in range(20)], ("hc3", "mage", "raid"))
    assert {r["set"] for r in out} == {"raid_reworded"}
    # reworded texts are never fitted on unless asked
    assert all(r["side"] in ("test", "unused") for r in out)


def test_newer_models_held_out_unless_asked():
    rows = [row(source="detectrlx", split="newmodels", doc_id=str(i)) for i in range(40)]
    assert all(r["side"] == "test" for r in mt.assign(rows, ("hc3", "mage", "raid")))
    sides = {r["side"] for r in mt.assign(rows, ("hc3", "mage", "raid", "newer"))}
    assert sides == {"train", "test"}


def test_raid_split_keeps_a_document_on_one_side():
    rows = [row(doc_id=str(i // 3), label=i % 2) for i in range(90)]
    out = mt.assign(rows, ("hc3", "mage", "raid"))
    side = {}
    for r in out:
        assert side.setdefault(r["doc_id"], r["side"]) == r["side"]


def _res(overall, kinds, sets=None):
    return {"overall": {"people_above_band": overall, "ai_above_band": 0.5},
            "by_kind": {k: {"people_above_band": v, "n_people": 100} for k, v in kinds.items()},
            "people_only": {k: {"people_above_band": v, "n_people": 50} for k, v in (sets or {}).items()},
            "sets": {}, "by_model": {}, "reworded": {}}


def test_ship_rule_passes_when_nothing_gets_worse():
    old = _res(0.008, {"news": 0.02, "qa": 0.01}, {"toefl": 0.03})
    new = _res(0.006, {"news": 0.02, "qa": 0.0}, {"toefl": 0.01})
    assert mt.compare(old, new)["ship"]


def test_ship_rule_fails_on_one_worse_kind_or_high_overall():
    old = _res(0.008, {"news": 0.02, "qa": 0.01})
    assert mt.compare(old, _res(0.005, {"news": 0.03, "qa": 0.0}))["worse"] == ["kind: news"]
    assert not mt.compare(old, _res(0.02, {"news": 0.0, "qa": 0.0}))["ship"]


def test_ship_rule_overall_limit_is_on_the_fitting_sources():
    old = _res(0.04, {"news": 0.1})
    new = _res(0.05, {"news": 0.1})
    new["overall_fit_sources"] = {"people_above_band": 0.008, "ai_above_band": 0.9}
    out = mt.compare(old, new)
    assert out["ship"] and out["overall_key"] == "overall_fit_sources"


def test_auroc_and_catch_rate_known_values():
    assert mt.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert mt.auroc([0.5, 0.5], [1, 0]) == 0.5
    # 100 people's scores 0..0.99: at 1 in 100 flagged the cut is 0.98, so only AI above 0.98 counts
    s = [i / 100 for i in range(100)] + [0.985, 0.5]
    assert mt.tpr_at_fpr(s, [0] * 100 + [1, 1]) == 0.5


def test_each_source_counts_the_same_in_fitting():
    tr = [row(source="hc3", label=i % 2) for i in range(10)] + [row(source="raid", label=i % 2) for i in range(90)]
    w = mt.weights_for(tr, "source")
    hc3 = sum(w[i] for i, r in enumerate(tr) if r["source"] == "hc3")
    raid = sum(w[i] for i, r in enumerate(tr) if r["source"] == "raid")
    assert hc3 == pytest.approx(raid)
    w2 = mt.weights_for(tr, "source_label_model")
    assert sum(w2) == pytest.approx(len(tr))


def _toy(n=200, seed=0):
    import random
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        lab = i % 2
        m = "human" if not lab else ("a" if i % 4 == 1 else "b")
        out.append(row(label=lab, model=m, kind="news" if i % 3 else "reviews", side="train", set="raid",
                       bino=rnd.gauss(1.0 - 0.3 * lab, 0.1), source="raid"))
    return out


def test_a_fit_separates_toy_data_and_its_band_stays_sane():
    tr = _toy()
    fit = mt.Fit(tr, ["bino"])
    p = fit.predict(tr)
    assert mt.auroc(p, [r["label"] for r in tr]) > 0.9
    lo, hi = fit.band
    assert lo <= 0.35 and 0.65 <= hi <= 0.99


def test_leave_one_model_out_never_fits_on_the_left_out_model(monkeypatch):
    tr, te = _toy(seed=1), _toy(seed=2)
    seen = []
    real_fit = mt.Fit

    class Spy(real_fit):
        def __init__(self, rows, *a, **k):
            seen.append({r["model"] for r in rows})
            super().__init__(rows, *a, **k)

    monkeypatch.setattr(mt, "Fit", Spy)
    exp = mt.Exp.__new__(mt.Exp)
    exp.a = type("A", (), {"weights": "source", "band": "auto", "l2": 1e-2})()
    exp.feats, exp.tr, exp.te = ["bino"], tr, te
    out = exp.leave_one_model_out(min_n=10)
    assert set(out) == {"a", "b"}
    assert seen == [{"human", "b"}, {"human", "a"}]


def test_in_100_words():
    assert mt.in_100(0.123) == "12 in 100"
    assert mt.in_100(0.004) == "fewer than 1 in 100"
    assert mt.in_100(None) == mt.in_100(float("nan")) == "not measured"


def test_clean_writes_null_for_missing_numbers():
    assert mt.clean({"a": float("nan"), "b": [1.0, float("inf")]}) == {"a": None, "b": [1.0, None]}
