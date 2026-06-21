"""Shared text/timestamp alignment utilities.

`Qwen3-ForcedAligner` returns timestamps only for spoken units (CJK chars or
Latin words); punctuation is dropped during tokenization. To reason about
"is there punctuation between these two adjacent ts entries?" we walk the
raw transcribed text in sync with the ts list and record each ts's char
span in the text.
"""

from __future__ import annotations

import unicodedata

PUNCT_NATURAL = set("，。！？、；：,.!?;:…—\n")
PUNCT_HARD = set("。！？.!?")
PUNCT_EXTRA = "，。！？、；：,.;:!? \t\n…—"


def is_punct_char(c: str) -> bool:
    """True if `c` is a punctuation/whitespace character we care about."""
    return unicodedata.category(c).startswith("P") or c in PUNCT_EXTRA


def is_natural_punct(c: str) -> bool:
    return c in PUNCT_NATURAL


def is_hard_punct(c: str) -> bool:
    return c in PUNCT_HARD


def strip_punct(s: str) -> str:
    return "".join(c for c in s if not is_punct_char(c))


def align_text_to_ts(text: str, ts) -> list[dict]:
    """Walk text + ts in sync.

    Each ts entry has `.text`, `.start_time`, `.end_time` (seconds). The
    text contains punctuation that the ts list does not.

    Returns a list of records:
        { ts_idx, text, span (start,end) into `text`, start_s, end_s,
          followed_by_punct, punct_after }
    """
    records: list[dict] = []
    cursor = 0
    for i, t in enumerate(ts):
        # Skip any punctuation/whitespace before this token in the text.
        while cursor < len(text) and is_punct_char(text[cursor]):
            cursor += 1
        tok = t.text
        if (cursor + len(tok) <= len(text)
                and text[cursor:cursor + len(tok)].lower() == tok.lower()):
            span = (cursor, cursor + len(tok))
            cursor = span[1]
        else:
            j = text.lower().find(tok.lower(), cursor)
            if j < 0:
                span = (-1, -1)
            else:
                span = (j, j + len(tok))
                cursor = span[1]
        records.append({
            "ts_idx": i,
            "text": tok,
            "span": span,
            "start_s": float(t.start_time),
            "end_s": float(t.end_time),
        })

    # For each record, determine whether punctuation follows it (before the
    # next record, or before the end of `text`).
    for k, rec in enumerate(records):
        _, this_end = rec["span"]
        next_start = (records[k + 1]["span"][0]
                      if k + 1 < len(records) else len(text))
        between = text[this_end:next_start] if this_end >= 0 else ""
        punct = "".join(c for c in between if is_punct_char(c))
        rec["followed_by_punct"] = bool(punct.strip())  # ignore lone whitespace
        rec["punct_after"] = punct
    return records


def find_position_after(text: str, prev_word: str, next_word: str) -> int:
    """Locate `prev_word` followed by `next_word` in `text`; return the
    index just after `prev_word`. Falls back to first occurrence of
    `prev_word`. Returns -1 if not found."""
    if not prev_word:
        return 0
    if next_word:
        i = text.find(prev_word + next_word)
        if i >= 0:
            return i + len(prev_word)
    i = text.find(prev_word)
    if i >= 0:
        return i + len(prev_word)
    return -1
