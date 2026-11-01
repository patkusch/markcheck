"""`markcheck text` with the actual language models. Slow; skipped by default.

Run with: pytest -m slow
Skips when the text libraries or the model weights are not on this computer.
"""
from __future__ import annotations

import re

import pytest

from markcheck.textcheck import CAVEAT, check_text, load_calibration

pytestmark = pytest.mark.slow

PARAGRAPHS = [
    # Plain, list-like prose of the kind chat models write.
    "Effective time management is essential for success in both personal and professional life. "
    "By prioritizing tasks, setting clear goals, and minimizing distractions, individuals can "
    "significantly improve their productivity. Additionally, taking regular breaks helps maintain "
    "focus and prevents burnout. It is also important to review progress regularly and adjust "
    "strategies as needed. Furthermore, using tools such as calendars and task lists can help "
    "people stay organized. In conclusion, developing strong time management skills can lead to "
    "reduced stress, better work-life balance, and greater overall satisfaction. Ultimately, by "
    "consistently applying these strategies, anyone can make the most of their available time and "
    "achieve their long-term goals more effectively.",
    # Chatty, uneven prose with odd details.
    "So the boiler died on Sunday, which, of course, is the one day nobody answers the phone. "
    "We boiled kettles. Lots of kettles. My brother claimed he could fix it with a coat hanger "
    "and a YouTube video from 2011, and for about forty minutes the house smelled like burnt dust "
    "and optimism. It did not work. By Monday the plumber came, sucked air through his teeth, and "
    "said a word I won't repeat here. Two hundred quid later we had hot water and a cat who now "
    "refuses to walk past the airing cupboard. Worth it? Ask me in January, when the frost is on "
    "the inside of the windows again and my brother is back on YouTube.",
]


@pytest.fixture(scope="module")
def scorer():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from markcheck.textcheck.scorer import MissingModels, PairScorer
    try:
        return PairScorer(*load_calibration()["models"])
    except MissingModels as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize("text", PARAGRAPHS, ids=["plain", "chatty"])
def test_models_score_a_paragraph(scorer, text):
    res = check_text(text, scorer=scorer)
    assert res.status == "scored"
    assert 0.0 <= res.probability <= 1.0
    assert re.match(r"Estimated chance this was AI-written: \d+% ", res.headline)
    assert len(res.reasons) == 3
    assert res.measures["tokens"] > 50
    assert res.caveat == CAVEAT


def test_missing_weights_give_missing_models():
    pytest.importorskip("transformers")
    from markcheck.textcheck.scorer import MissingModels, PairScorer
    with pytest.raises(MissingModels):
        PairScorer("markcheck-tests/no-such-model", "markcheck-tests/no-such-model-either")
