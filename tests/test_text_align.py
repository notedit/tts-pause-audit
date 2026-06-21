from types import SimpleNamespace

from pause_detector.text_align import (
    align_text_to_ts,
    find_position_after,
    is_punct_char,
    strip_punct,
)


def _ts(text: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def test_is_punct_char():
    assert is_punct_char("。")
    assert is_punct_char(",")
    assert not is_punct_char("刚")
    assert not is_punct_char("a")


def test_strip_punct():
    assert strip_punct("刚收到，发现是。") == "刚收到发现是"


def test_align_text_to_ts_marks_punctuation_boundary():
    text = "刚收到，发现"
    ts = [
        _ts("刚", 0.0, 0.2),
        _ts("收", 0.2, 0.4),
        _ts("到", 0.4, 0.6),
        _ts("发", 0.7, 0.9),
        _ts("现", 0.9, 1.1),
    ]
    recs = align_text_to_ts(text, ts)
    assert [r["text"] for r in recs] == ["刚", "收", "到", "发", "现"]
    # 到 is followed by "，" in text
    assert recs[2]["followed_by_punct"] is True
    assert recs[2]["punct_after"] == "，"
    # the others are not at punct boundaries
    assert recs[0]["followed_by_punct"] is False
    assert recs[3]["followed_by_punct"] is False


def test_find_position_after():
    text = "刚收到，发现是男朋友"
    # 友 is the 9th char (index 9), so position right after it is 10
    assert find_position_after(text, "友", "") == 10
    # disambiguate with next token
    assert find_position_after(text, "到", "发") == 3
    # not found
    assert find_position_after(text, "汪", "") == -1
