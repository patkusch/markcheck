"""Cheap text measures that need no model: words, sentences, evenness."""
from __future__ import annotations

import re
import statistics

_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|\n\s*\n")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")


def normalise(text: str) -> str:
    """Tidy spacing so stray spaces before punctuation do not change the score."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+([.,;:!?%)])", r"\1", text)
    text = re.sub(r"([(])[ \t]+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences. Simple rules: '.', '!' or '?' then a capital, or a blank line."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    # Trim surrounding spaces.
    out = []
    for a, b in spans:
        seg = text[a:b]
        a2 = a + (len(seg) - len(seg.lstrip()))
        b2 = b - (len(seg) - len(seg.rstrip()))
        out.append((a2, b2))
    return out


def sentence_length_spread(text: str) -> float | None:
    """How uneven sentence lengths are: standard deviation divided by the mean.

    People tend to mix short and long sentences (a high number). Very even
    lengths (a low number) are more typical of AI text. None if fewer than
    three sentences.
    """
    lens = [len(words(text[a:b])) for a, b in sentences(text)]
    lens = [n for n in lens if n > 0]
    if len(lens) < 3:
        return None
    mean = statistics.fmean(lens)
    return statistics.pstdev(lens) / mean if mean else None
