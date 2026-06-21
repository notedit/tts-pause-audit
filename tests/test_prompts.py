from pause_detector.prompts import (
    KIND_DESC,
    PAUSE_SYSTEM_PROMPT,
    build_pause_user_prompt,
)


def test_system_prompt_contains_calibrated_samples():
    # All five calibrated samples (A–E) must be present so the LLM has the
    # ground-truth reference points.
    for tag in ["样例 A", "样例 B", "样例 C", "样例 D", "样例 E"]:
        assert tag in PAUSE_SYSTEM_PROMPT


def test_system_prompt_declares_output_schema():
    assert '"natural"' in PAUSE_SYSTEM_PROMPT
    assert '"reason"' in PAUSE_SYSTEM_PROMPT


def test_build_user_prompt_strips_punctuation_and_marks_position():
    text = "刚收到快递，发现是男朋友偷偷寄来的"
    pos = text.find("友") + 1  # position right after 友
    out = build_pause_user_prompt(text, "友", "偷", pos,
                                  duration_ms=350, kind="inter_char")
    # the visible 原文 line must be punctuation-free
    origin_line = next(line for line in out.splitlines() if line.startswith("原文"))
    assert "，" not in origin_line
    # marker is inserted between 友 and 偷
    assert "男朋友⏸偷偷寄来的" in out
    # kind descriptor injected
    assert KIND_DESC["inter_char"] in out
    # duration shown
    assert "350ms" in out


def test_build_user_prompt_handles_boundary_words():
    out = build_pause_user_prompt("一二三", "", "一", 0, kind="leading")
    assert "<句首>" in out
