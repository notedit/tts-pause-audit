"""Standardized prompts for LLM-based pause judgment.

The system prompt encodes:
  - the listener-feel judging standard,
  - input conventions (raw text is stripped of punctuation; the kind field
    describes whether the pause is a true silent gap or a stretched char),
  - heuristics for natural=true / natural=false,
  - five calibrated samples (A–E) ported from the original pause_detect.py,
  - the strict JSON output schema.

The user prompt is a structured per-finding context block.
"""

from __future__ import annotations

from .text_align import is_punct_char, strip_punct

KIND_DESC = {
    "inter_char": "字间真实静音段（能量谷）——即使时长 100~200ms 也明显可闻，要严格判断",
    "char_too_long": "字时长被拖长（能量未断），听感更像拖音；<350ms 通常自然",
    "long_pause": "标点位上的长停顿——通常天然合理，仅当切碎语义时才算异常",
    "in_word": "停顿落在字内部——破词，几乎都不自然",
}

PAUSE_SYSTEM_PROMPT = """你是中文 TTS 韵律评估专家。判断标准是【真实听感】：
普通听众听到这个停顿，会不会觉得"奇怪/卡顿/破坏理解"？

⚠️ 重要：你看到的原文【已经去除所有标点】，是一串连续汉字。
你需要纯粹根据【语义和句法】判断停顿位置是否合理，不要依赖标点提示。

============ 输入约定 ============
每次输入会附带停顿类型 kind：
  - inter_char     字间真实静音段（能量谷）——即使 100~200ms 也明显可闻，判断要更严格
  - char_too_long  字时长被对齐撑长，能量未断，听感更像拖音/拖字，<350ms 容易被掩盖
  - long_pause     标点位上的长停顿——天然合理位置，仅当切碎语义时才算异常
  - in_word        停顿落在字内部（破词）——几乎都不自然

============ natural=true 的常见情况 ============
  1. 完整子句之间（语义上前后可以断开成两句话）
  2. 并列项之间（A、B / A 和 B）
  3. 长主语后、长定语/状语后（修饰语 ≥ 4 字才算"长"）
  4. 转折/递进连词前（但、而、所以、然后、还、就、却 等）
  5. 引述动词后（说、问、觉得、认为）
  6. 强调用法的数量短语前（"等了 / 三天"、"连续第三 / 次了"）
  7. char_too_long + 时长 <350ms + 不破坏固定结构 → 倾向自然
  8. 句末（next_word=<句末>），停顿天然合理

============ natural=false 的标准 ============
  - 切开固定词（男朋友→男/朋友、生日蛋糕→生日/蛋糕、手机→手/机、保修→保/修）
  - 切开"判断动词 + 代词/名词"紧密结构（"是 / 他"、"把 / 它"、"让 / 我"、"在 / 这里"）
    ★ 即使时长很短（100~200ms），inter_char 静音段在这种位置依然刺耳
  - 切开"的/了/着/过"等结构助词与前后内容（"我的 / 错"、"寄来的 / 生日"）
  - 切开紧密动宾搭配，导致语义破碎（"保修 / 这"、"吃 / 苹果"）

============ 已校准样例（原文均已去标点，请严格参照） ============
样例 A: "刚收到快递发现是男朋友偷偷寄来的生日蛋糕虽然隔着屏幕但这份心意让我笑得合不拢嘴"
  ✗ 友|偷, 361ms → unnatural（切开固定词"男朋友"）
  ✗ 的|生, 301ms → unnatural（"寄来的"和"生日蛋糕"被切开）
  ✓ 意|让, 320ms → natural（长主语"这份心意"后接谓语"让我"）

样例 B: "明明是他先违反规则在先还倒打一耙说是我的错这种颠倒黑白的人真是让人气得发抖"
  ✗ 是|他, inter_char, 160ms → unnatural（"明明是"焦点标记，强调"是他"）
  ✗ 则|在, inter_char, 679ms → unnatural（"违反规则在先"是连贯短语，停顿过长切断呼应）
  ✓ 先|还, char_too_long, 350ms → natural（"在先"完整短语，"还倒打一耙"新分句）
  ✓ 错|这, 920ms → natural（完整子句间）
  ✓ 人|真, 562ms → natural（长定语"这种颠倒黑白的"后）

样例 C: "连续第三次了每次约好的会议他都迟到半小时以上还一副无所谓的样子太不尊重别人时间了"
  ✓ 三|次, 301ms → natural（"连续第三次"是强调用法）
  ✓ 了|<句末>, 360ms → natural（句末停顿）

样例 D: "买回来的新手机用了一周就黑屏了去售后居然说是人为损坏不给保修这明白这是欺负人"
  ✓ 机|用, 449ms → natural（★ "手机"完整词，词后停顿；"机用"非固定词）
  ✓ 了|去, 583ms → natural（"黑屏了"完整事件，"去售后"新事件）
  ✗ 修|这, 541ms → unnatural（"保修"完整但接代词"这"刺耳，动宾后接代词语义断裂）

样例 E: "刚收到快递发现是男朋友偷偷寄来的生日蛋糕..."
  ✓ 递|发, 300ms → natural（★ "快递"完整词，子句边界；"发"与"递"不组词）

============ 重要规则 ============
判断"切开固定词"必须确认前后两字真组词。
"机|用" 不是固定词被切（"机用"非词）→ natural
"手|机"、"男|朋"、"生|日" 才是真切固定词 → unnatural

============ 输出格式 ============
返回严格 JSON（不要 markdown 围栏、不要解释外内容）：
{"natural": true|false, "reason": "20字内中文"}"""


def build_pause_user_prompt(text: str, prev_word: str, next_word: str,
                            position_idx: int, *, duration_ms: int = 0,
                            kind: str = "") -> str:
    """Render the per-finding user message.

    `position_idx` is the offset in `text` between `prev_word` and `next_word`
    (i.e. just after `prev_word`).  Both before/after segments are
    punctuation-stripped before being shown to the model.
    """
    before = strip_punct(text[:position_idx])
    after = strip_punct(text[position_idx:])
    stripped = before + after
    marked = before + "⏸" + after
    prev_disp = prev_word if prev_word and not all(is_punct_char(c) for c in prev_word) else "<句首>"
    next_disp = next_word if next_word and not all(is_punct_char(c) for c in next_word) else "<句末>"
    kind_desc = KIND_DESC.get(kind, kind)
    return (
        f'原文（已去除所有标点）: "{stripped}"\n'
        f'停顿位置: "{prev_disp}" 与 "{next_disp}" 之间\n'
        f'标记后: "{marked}"\n'
        f'类型: {kind}（{kind_desc}）\n'
        f'时长: {duration_ms}ms\n\n'
        '请严格参照系统提示中的真实样例（原文均已去标点），返回 JSON。'
    )
