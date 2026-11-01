"""Cheap text measures that need no model: words, sentences, evenness."""
from __future__ import annotations

import re
import statistics

# Places a sentence may end: '.', '!' or '?' (plus closing quotes or brackets) and spaces, a
# blank line, or a line break (kept only before a list item or a heading, or after a heading).
_SENT_END = re.compile(r"[.!?][\"')\]’”]*\s+|\n\s*\n|\n[ \t]*")
# Letters in any alphabet or digits; joined by apostrophes or hyphens; "3.5" and "1,000" are one word.
_WORD = re.compile(r"[^\W_](?:[^\W_]|['’-](?=[^\W_])|(?<=\d)[.,](?=\d))*")
# Short words that end in a full stop without ending the sentence ("Dr. Smith", "No. 5").
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "fig", "no", "gen", "col",
           "lt", "sgt", "capt", "rev", "hon", "approx", "cf", "al"}
_LAST_WORD = re.compile(r"([^\W\d_]+(?:\.[^\W\d_]+)*)\.$")


def normalise(text: str) -> str:
    """Tidy spacing so stray spaces before punctuation do not change the score.

    Only spacing changes: a space before a mark is dropped only when the mark ends a word
    (so ".NET" or "the .bashrc file" keep their space).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+([.,;:!?%)])(?=[\s.,;:!?%)\"'’”]|$)", r"\1", text)
    text = re.sub(r"([(])[ \t]+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text)


_ITEM = re.compile(r"(?:[-*•–]|\d{1,3}[.)]|#{1,6})[ \t]")


def _ends_sentence(text: str, m: re.Match) -> bool:
    gap = m.group(0)
    if gap[0] not in ".!?":
        if gap.count("\n") > 1 or _ITEM.match(text, m.end()):
            return True  # a blank line, or a list item or heading starts the next line
        line = text[text.rfind("\n", 0, m.start()) + 1:m.start()].lstrip()
        return bool(re.match(r"#{1,6}[ \t]", line))  # the line that ends is a heading
    if re.search(r"\n\s*\n", gap) or ("\n" in gap and _ITEM.match(text, m.end())):
        return True
    rest = text[m.end():m.end() + 8].lstrip("\"'([‘“")
    if not rest or not (rest[0].isupper() or rest[0].isdigit()):
        return False
    if gap[0] == ".":
        w = _LAST_WORD.search(text[max(0, m.start() - 40):m.start() + 1])
        if w and (w.group(1).lower() in _ABBREV or all(len(p) == 1 for p in w.group(1).split("."))):
            return False  # "Dr. Smith", "e.g. The", "U.S. Army", "J. Smith"
    return True


def sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences.

    Simple rules: '.', '!' or '?' then a capital or a digit (but not after a short form such
    as "Dr." or "U.S."), a blank line, or a new line that starts a list item or a heading.
    """
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        if not _ends_sentence(text, m):
            continue
        end = m.start() + (len(m.group(0)) - len(m.group(0).lstrip(".!?\"')]’”"))
                           if m.group(0)[:1] in ".!?" else 0)
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
