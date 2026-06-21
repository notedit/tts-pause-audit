"""
TTS 不自然停顿检测 + LLM 语义判定 — 单文件版

核心流程：
  Qwen3-ASR-Flash-Filetrans 转写  →  RMS 能量谷分析  →  LLM 语义判定
  (字级时间戳)                     (反向定位停顿位置)   (qwen3-max 判定)

=== 三种用法 ===

  1. 一步到位（推荐）— 转写 + 检测 + 判定一气呵成：
     python3 pause_detect.py run audio.wav --json out.json

  2. 仅检测（不调 LLM，更快）：
     python3 pause_detect.py detect audio.wav --json out.json

  3. 对已检测好的 JSON 跑 LLM 判定：
     python3 pause_detect.py judge out.json [out2.json ...]

=== 检测信号 ===
  S1. 字间能量谷（inter_char）:
      RMS 比整段 90 分位低 ≥12dB、持续 ≥260ms 的字间静音
  S2. 字时长 z-score（char_too_long）:
      字时长 z ≥ +2.0 → 真实静音被 ASR 分摊到字尾，再用能量谷反向定位
  S3. 字内能量谷（in_word）:
      谷深度 ≥18dB / 时长 ≥120ms，落在字内部 → 破词
  S4. 标点位长停顿（long_pause）:
      ≥260ms 的标点位停顿（也送 LLM 判定）
  S5. 首/尾静音（leading/trailing）:
      仅记录，不报警

=== 关键洞察 ===
  所有 ASR 强制对齐都会把"字尾静音"分摊到字 end 时间。
  这里用 RMS 能量谷反向定位，让停顿位置精确到 ±20ms。

=== 依赖 ===
  pip install dashscope numpy soundfile

=== 环境变量 ===
  DASHSCOPE_API_KEY: 阿里云百炼 API Key

=== 用法示例 ===

  # 单条音频
  python3 pause_detect.py run clip.wav --json clip.data.json

  # 批量
  for f in *.wav; do
      python3 pause_detect.py run "$f" --json "${f%.wav}.json"
  done

  # 仅判定（适合阈值调优后只重跑 LLM）
  python3 pause_detect.py judge clip1.json clip2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf


# ============================================================================
# 常量配置
# ============================================================================

PUNCT_NATURAL = set("，。！？、；：,.!?;:…—\n")
PUNCT_HARD = set("。！？.!?")
PUNCT_CHARS = set("，。！？、；：,.!?;:…—\n ")

# 字时长 z-score 阈值
CHAR_DUR_Z = 2.0

# RMS 能量谷参数
VALLEY_DROP_DB = 10        # 比 ref_db 低多少 dB 才算谷
VALLEY_MIN_MS = 80         # 最小谷时长

# 字间停顿触发条件
INTER_CHAR_MIN_MS = 260    # 谷时长 ≥ 此值且 drop ≥ 12dB → 报字间停顿
INTER_CHAR_MIN_DROP_DB = 12

# 字内能量谷
IN_WORD_MIN_MS = 120       # 在字内部出现的谷至少多长
IN_WORD_MIN_DROP_DB = 18

# 标点位长停顿
PUNCT_PAUSE_MIN_MS = 260

# Qwen3 ASR 模型名
ASR_MODEL = "qwen3-asr-flash-filetrans"

# LLM 判定模型名
LLM_MODEL_DEFAULT = "qwen3-max"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Finding:
    start: float
    end: float
    duration: float
    kind: str            # inter_char | char_too_long | in_word | long_pause |
                         # punct_hard | punct_soft | leading | trailing
    label: str
    z: float = 0.0
    depth_db: float = 0.0
    suspicious: bool = False
    reason: str = ""


# ============================================================================
# 工具函数
# ============================================================================

def is_punct(tok: str) -> bool:
    return any(ch in PUNCT_NATURAL for ch in tok)


def strip_punct(s: str) -> str:
    return "".join(c for c in s if c not in PUNCT_CHARS)


# ============================================================================
# Step 1: ASR — Qwen3-ASR-Flash-Filetrans
# ============================================================================

def run_qwen3_asr(wav_path: str, language: str = "zh"):
    """用 Qwen3-ASR 拿字级时间戳。

    流程: Files.upload → Files.get(取签名 URL) → QwenTranscription.async_call

    Qwen3 中文的 words 已经是字粒度，且 punctuation 字段附在前一个字上。
    这里把 punctuation 拆成独立 token，以保持下游接口一致（区分字 / 标点）。

    Returns:
        (text, language, words) — words 元素含 word/start/end/score
    """
    import dashscope
    from dashscope import Files
    from dashscope.audio.qwen_asr import QwenTranscription

    if not dashscope.api_key:
        dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    # 1. 上传 → 拿签名 URL（dashscope filetrans 只接受阿里 OSS URL）
    up = Files.upload(file_path=wav_path, purpose="audio")
    fid = up.output["uploaded_files"][0]["file_id"]
    file_url = Files.get(fid).output["url"]

    # 2. 异步转写（enable_words=True 拿字级时间戳）
    task = QwenTranscription.async_call(
        model=ASR_MODEL,
        file_url=file_url,
        enable_words=True,
        enable_itn=False,
    )
    result = QwenTranscription.wait(task=task.output.task_id)
    out = result.output
    if out.get("task_status") != "SUCCEEDED":
        raise RuntimeError(f"Qwen3 转写失败: {out.get('code')} {out.get('message')}")

    res_url = out["result"]["transcription_url"]
    doc = json.loads(urlopen(res_url, timeout=30).read())
    text = doc["transcripts"][0]["text"]

    words = []
    for s in doc["transcripts"][0]["sentences"]:
        for w in s["words"]:
            t = w["text"].strip()
            if not t:
                continue
            words.append({
                "word": t,
                "start": w["begin_time"] / 1000.0,
                "end": w["end_time"] / 1000.0,
                "score": 0.0,
            })
            # 标点附在前一字 punctuation 字段，转成独立 token
            p = w.get("punctuation", "").strip()
            if p:
                words.append({
                    "word": p,
                    "start": w["end_time"] / 1000.0,
                    "end": w["end_time"] / 1000.0,
                    "score": 0.0,
                })
    return text, language, words


# ============================================================================
# Step 2: RMS 能量包络 + 谷检测
# ============================================================================

def compute_rms_db(wav_path: str, win_ms: int = 30, hop_ms: int = 10):
    """计算 RMS 能量包络。返回 (rms_db, hop_sec, win_sec, ref_db)。

    ref_db = 整段 90 分位 — 用作"正常说话能量"的相对基准。
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    n = (len(audio) - win) // hop + 1
    if n <= 0:
        return np.array([]), hop / sr, win_ms / 1000, -60.0
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n, win),
        strides=(hop * audio.strides[0], audio.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    ref_db = float(np.percentile(rms_db, 90))
    return rms_db, hop / sr, win_ms / 1000, ref_db


def find_valleys(rms_db, hop_s, win_s, ref_db,
                 drop_db=VALLEY_DROP_DB, min_ms=VALLEY_MIN_MS):
    """找出所有 RMS < ref_db - drop_db 的连续段。

    Returns: [(start_s, end_s, depth_db)]
    """
    is_v = rms_db < (ref_db - drop_db)
    out = []
    i = 0
    while i < len(is_v):
        if is_v[i]:
            j = i
            while j < len(is_v) and is_v[j]:
                j += 1
            s_t = i * hop_s
            e_t = j * hop_s + win_s
            if (e_t - s_t) * 1000 >= min_ms:
                depth = ref_db - float(rms_db[i:j].min())
                out.append((s_t, e_t, depth))
            i = j
        else:
            i += 1
    return out


# ============================================================================
# Step 3: 联合分析 — 把能量谷、字时长、标点位关联到具体字位置
# ============================================================================

def analyze(wav_path: str, language: str = "zh"):
    """完整流程：ASR → RMS → 联合分析。

    Returns: (findings, mu, sigma, total_dur, text, ref_db, words, rms_db, hop_s)
    """
    print(f"[1/3] Qwen3-ASR-Flash-Filetrans 转写中...", flush=True)
    text, lang, words = run_qwen3_asr(wav_path, language)
    print(f"   语言={lang}, token={len(words)}")
    print(f"   ASR: {text}")

    print(f"[2/3] RMS 包络 ...", flush=True)
    rms_db, hop_s, win_s, ref_db = compute_rms_db(wav_path)
    valleys = find_valleys(rms_db, hop_s, win_s, ref_db)
    total_dur = len(rms_db) * hop_s + win_s
    print(f"   时长 {total_dur:.2f}s  ref_db={ref_db:.1f}dB  谷数={len(valleys)}")

    print(f"[3/3] 联合分析 ...", flush=True)
    findings: list[Finding] = []
    chars = [w for w in words if not is_punct(w["word"])]
    puncts = [w for w in words if is_punct(w["word"])]
    durs = np.array([w["end"] - w["start"] for w in chars]) if chars else np.array([])
    mu = float(durs.mean()) if len(durs) else 0.0
    sigma = float(durs.std()) if len(durs) > 0 else 0.0

    # leading / trailing 静音
    if words and words[0]["start"] > 0:
        findings.append(Finding(0.0, words[0]["start"], words[0]["start"],
                                "leading", "<BOS>"))
    if words and words[-1]["end"] < total_dur:
        d = total_dur - words[-1]["end"]
        findings.append(Finding(words[-1]["end"], total_dur, d, "trailing", "<EOS>"))

    # ---------- 每个 RMS 谷只报告一次 ----------
    def _overlaps(a0, a1, b0, b1):
        return min(a1, b1) - max(a0, b0)

    for vs, ve, depth in valleys:
        d_ms = (ve - vs) * 1000

        # 1) 谷被标点 token 覆盖（≥50%）→ 跳过（标点位由后续逻辑处理）
        if any(
            _overlaps(vs, ve, p["start"], p["end"]) > 0
            and _overlaps(vs, ve, p["start"], p["end"]) / (ve - vs) >= 0.5
            for p in puncts
        ):
            continue

        # 2) 谷完全落在某字内部 → 破词
        in_word_w = next(
            (w for w in chars
             if vs > w["start"] + 0.06 and ve < w["end"] - 0.06),
            None,
        )
        if in_word_w is not None and d_ms >= IN_WORD_MIN_MS and depth >= IN_WORD_MIN_DROP_DB:
            findings.append(Finding(
                vs, ve, ve - vs, "in_word",
                in_word_w["word"], depth_db=depth, suspicious=True,
                reason=f"字 [{in_word_w['word']}] 内部能量谷 {int(d_ms)}ms (drop={depth:.0f}dB)",
            ))
            continue

        # 3) 字间停顿：找谷起点 vs 附近最近的字对
        # prev_w = 字 end 最接近 vs 的字（允许稍稍越过 vs，因为 ASR 把静音吃进了字 end）
        prev_w, prev_idx = None, -1
        best_dist = 1e9
        for i, w in enumerate(chars):
            if w["start"] < vs + 0.05:
                dist = abs(w["end"] - vs)
                if dist < best_dist:
                    best_dist = dist
                    prev_w, prev_idx = w, i
        next_w = chars[prev_idx + 1] if 0 <= prev_idx < len(chars) - 1 else None
        if prev_w is None or next_w is None:
            continue

        # 谷必须跨过 prev_w 中点（保证停顿真在字尾）
        if vs >= (prev_w["start"] + prev_w["end"]) / 2 - 0.03:
            prev_z = (prev_w["end"] - prev_w["start"] - mu) / sigma if sigma > 1e-6 else 0.0
            cond_long = d_ms >= INTER_CHAR_MIN_MS and depth >= INTER_CHAR_MIN_DROP_DB
            cond_z = prev_z >= CHAR_DUR_Z
            if cond_long or cond_z:
                reasons = [f"字间能量谷 {int(d_ms)}ms (drop={depth:.0f}dB)"]
                if cond_z:
                    reasons.append(f"前字 [{prev_w['word']}] z={prev_z:+.1f} 拖长")
                findings.append(Finding(
                    vs, ve, ve - vs, "inter_char",
                    f"{prev_w['word']}→{next_w['word']}",
                    z=prev_z, depth_db=depth, suspicious=True,
                    reason="; ".join(reasons),
                ))

    # ---------- 字时长 z-score（独立信号，未被 inter_char 覆盖才报）----------
    inter_intervals = [(f.start, f.end) for f in findings if f.kind == "inter_char"]

    def _overlaps_inter(s, e):
        for a, b in inter_intervals:
            if min(e, b) - max(s, a) > 0.05:
                return True
        return False

    for w in chars:
        d = w["end"] - w["start"]
        z = (d - mu) / sigma if sigma > 1e-6 else 0.0
        if z < CHAR_DUR_Z:
            continue

        # 在该字时间区间内找最大能量谷，用谷起止纠正定位
        best_v = None
        for vs, ve, depth in valleys:
            if vs >= w["start"] - 0.05 and vs <= w["end"] + 0.05 \
                    and (ve - vs) >= 0.15 and depth >= 15:
                if best_v is None or depth > best_v[2]:
                    best_v = (vs, ve, depth)

        if best_v is not None:
            vs, ve, depth = best_v
            if _overlaps_inter(vs, ve):
                continue
            # 谷在字的前半还是后半 → 决定 prev/next 标签
            mid_rel = ((vs + ve) / 2 - w["start"]) / max(d, 0.001)
            j = chars.index(w)
            prev_char = chars[j - 1] if j > 0 else None
            next_char = chars[j + 1] if j + 1 < len(chars) else None
            if mid_rel >= 0.5:
                pw_label = w["word"]
                nw_label = next_char["word"] if next_char else "<EOS>"
            else:
                pw_label = prev_char["word"] if prev_char else "<BOS>"
                nw_label = w["word"]
            findings.append(Finding(
                vs, ve, ve - vs, "char_too_long", f"{pw_label}→{nw_label}",
                z=z, depth_db=depth, suspicious=True,
                reason=(f"字 [{w['word']}] 拖长 {int(d*1000)}ms (z={z:.1f})；"
                        f"实际能量谷 {int((ve-vs)*1000)}ms (drop={depth:.0f}dB)"),
            ))
        elif not _overlaps_inter(w["start"], w["end"]):
            findings.append(Finding(
                w["start"], w["end"], d, "char_too_long",
                w["word"], z=z, suspicious=True,
                reason=f"字 [{w['word']}] 时长 {int(d*1000)}ms (z={z:.1f}) 异常拖长",
            ))

    # ---------- 标点位长停顿 ----------
    for w in puncts:
        d = w["end"] - w["start"]
        if d * 1000 >= PUNCT_PAUSE_MIN_MS:
            prev_w = next_w = None
            for cw in chars:
                if cw["end"] <= w["start"] + 0.02:
                    prev_w = cw
                elif next_w is None and cw["start"] >= w["end"] - 0.02:
                    next_w = cw
            label = (
                f"{prev_w['word'] if prev_w else '<句首>'}→"
                f"{next_w['word'] if next_w else '<句末>'}"
            )
            findings.append(Finding(
                w["start"], w["end"], d, "long_pause", label,
                suspicious=True,
                reason=f"停顿 {int(d*1000)}ms（送 LLM 判定）",
            ))
        else:
            kind = "punct_hard" if any(c in PUNCT_HARD for c in w["word"]) else "punct_soft"
            findings.append(Finding(w["start"], w["end"], d, kind,
                                    w["word"], reason="—"))

    findings.sort(key=lambda f: f.start)
    return findings, mu, sigma, total_dur, text, ref_db, words, rms_db, hop_s


# ============================================================================
# Step 4: LLM 语义判定
# ============================================================================

LLM_SYSTEM_PROMPT = """你是中文 TTS 韵律评估专家。判断标准是【真实听感】：
普通听众听到这个停顿，会不会觉得"奇怪/卡顿/破坏理解"？

⚠️ 重要：你看到的原文【已经去除所有标点】，是一串连续汉字。
你需要纯粹根据【语义和句法】判断停顿位置是否合理，不要依赖标点提示。

输入会附带停顿类型：
  - inter_char: 字与字之间出现的【真实静音段】（能量谷），即使时长只有 100~200ms，
    人耳也容易听到——判断要更严格。
  - char_too_long: 字时长被对齐"撑长"，能量没断，听感更像拖音/拖字。
    短的（<350ms）容易被掩盖，倾向自然。

判 natural=true 的常见情况：
  1. 完整子句之间（语义上前后可以断开成两句话）
  2. 并列项之间（A、B / A 和 B）
  3. 长主语后、长定语/状语后（修饰语 ≥ 4 字才算"长"）
  4. 转折/递进连词前（但、而、所以、然后、还、就、却 等）
  5. 引述动词后（说、问、觉得、认为）
  6. 强调用法的数量短语前（"等了 / 三天"、"连续第三 / 次了"）
  7. char_too_long + 时长 <350ms + 不破坏固定结构 → 倾向自然
  8. 句末（next_word=<句末>），停顿天然合理

判 natural=false 的标准：
  - 切开固定词（男朋友→男/朋友、生日蛋糕→生日/蛋糕、手机→手/机、保修→保/修）
  - 切开"判断动词 + 代词/名词"紧密结构（"是 / 他"、"把 / 它"、"让 / 我"、"在 / 这里"）
    ★ 即使时长很短（100~200ms），inter_char 静音段在这种位置依然刺耳
  - 切开"的/了/着/过"等结构助词与前后内容（"我的 / 错"、"寄来的 / 生日"）
  - 切开紧密动宾搭配，导致语义破碎（"保修 / 这"、"吃 / 苹果"）

【已校准的真实样例】（原文均已去标点，请严格参照）：
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

⚠️ 重要规则：判断"切开固定词"必须确认前后两字真组词。
   "机|用" 不是固定词被切（"机用"非词）→ natural
   "手|机"、"男|朋"、"生|日" 才是真切固定词 → unnatural

返回严格 JSON: {"natural": true|false, "reason": "20字内中文"}"""


def _build_user_prompt(text: str, prev_word: str, next_word: str,
                       position_idx: int, duration_ms: int = 0, kind: str = ""):
    """剥掉所有标点，给 LLM 纯语义判断。"""
    before = strip_punct(text[:position_idx])
    after = strip_punct(text[position_idx:])
    stripped = before + after
    marked = before + "⏸" + after
    prev_disp = prev_word if prev_word and prev_word not in PUNCT_CHARS else "<句首>"
    next_disp = next_word if next_word and next_word not in PUNCT_CHARS else "<句末>"
    kind_desc = "字间真实静音" if kind == "inter_char" else "字时长拖长"
    return (
        f'原文（已去除所有标点）: "{stripped}"\n'
        f'停顿位置: "{prev_disp}" 与 "{next_disp}" 之间\n'
        f'标记后: "{marked}"\n'
        f'类型: {kind}（{kind_desc}）\n'
        f'时长: {duration_ms}ms\n\n'
        '严格参照系统提示中的真实样例（原文均已去标点），给出 JSON。'
    )


def _find_position(text: str, prev_word: str, next_word: str):
    """在原文中找 prev_word 紧跟 next_word 的位置，返回 prev_word 之后下标。"""
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


def _find_finding_context(finding: dict, words: list):
    """从 finding 推出 prev_word / next_word。"""
    label = finding["label"]
    kind = finding["kind"]
    end = finding["end"]

    # 新版 label 都是 "前→后" 格式
    if "→" in label:
        return label.split("→", 1)

    # 老版 char_too_long: label 仅一个字，停顿在该字之后
    if kind == "char_too_long":
        prev_word = label
        idx = next((i for i, w in enumerate(words)
                    if w["word"] == label and abs(w["end"] - end) < 0.05), -1)
        next_word = ""
        if idx >= 0 and idx + 1 < len(words):
            next_word = words[idx + 1]["word"]
        return prev_word, next_word

    if kind == "in_word":
        return label, label
    return label, ""


def _call_dashscope_chat(model: str, system: str, user: str,
                         max_tokens=128, timeout=30):
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    req = Request(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    r = urlopen(req, timeout=timeout)
    return json.loads(r.read())["choices"][0]["message"]["content"]


def judge_one(model, text, prev_word, next_word,
              duration_ms=0, kind="", max_retry=2):
    pos = _find_position(text, prev_word, next_word)
    if pos < 0:
        return {"natural": False, "reason": "无法定位"}
    user = _build_user_prompt(text, prev_word, next_word, pos, duration_ms, kind)

    last_err = None
    for attempt in range(max_retry + 1):
        try:
            txt = _call_dashscope_chat(model, LLM_SYSTEM_PROMPT, user).strip()
            # 兼容 ```json ... ``` 包装
            if "```" in txt:
                txt = txt.split("```")[1]
                if txt.startswith("json"):
                    txt = txt[4:]
            j = json.loads(txt.strip())
            return {
                "natural": bool(j.get("natural", False)),
                "reason": str(j.get("reason", "")).strip()[:50],
            }
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    return {"natural": False, "reason": f"LLM 失败: {last_err}"}


def judge_findings_in_data(data: dict, model: str = LLM_MODEL_DEFAULT, verbose=True):
    """对 data['findings'] 中所有 suspicious=true 的项做 LLM 判定，原地修改。"""
    text = data["text"]
    words = data["words"]
    findings = data["findings"]

    sus_idx = [i for i, f in enumerate(findings) if f.get("suspicious")]
    if verbose:
        print(f"  原文: {text}")
        print(f"  可疑停顿: {len(sus_idx)} 处")

    for i in sus_idx:
        f = findings[i]
        prev_w, next_w = _find_finding_context(f, words)
        verdict = judge_one(
            model, text, prev_w, next_w,
            duration_ms=int(f.get("duration_ms", 0)),
            kind=f.get("kind", ""),
        )
        f["llm_natural"] = verdict["natural"]
        f["llm_reason"] = verdict["reason"]
        f["llm_prev"] = prev_w
        f["llm_next"] = next_w
        if verbose:
            flag = "✓自然" if verdict["natural"] else "✗不自然"
            print(f"    [{f['start']:.2f}s] {prev_w}|{next_w}  →  {flag}: {verdict['reason']}")
    return data


# ============================================================================
# 序列化
# ============================================================================

def findings_to_payload(audio_path, findings, mu, sigma, total_dur,
                       text, ref_db, words, rms_db, hop_s):
    """构造可视化页面用的 JSON 数据。"""
    return {
        "audio": os.path.basename(audio_path),
        "duration": float(total_dur),
        "text": text,
        "char_dur_mu_ms": round(mu * 1000, 1),
        "char_dur_sigma_ms": round(sigma * 1000, 1),
        "ref_db": round(float(ref_db), 2),
        "words": [
            {
                "word": w["word"],
                "start": round(w["start"], 4),
                "end": round(w["end"], 4),
                "score": round(w.get("score", 0.0), 3),
                "is_punct": is_punct(w["word"]),
            }
            for w in words
        ],
        "findings": [
            {
                "start": round(f.start, 4),
                "end": round(f.end, 4),
                "duration_ms": round(f.duration * 1000, 1),
                "kind": f.kind,
                "label": f.label,
                "z": round(f.z, 2),
                "depth_db": round(f.depth_db, 2),
                "suspicious": f.suspicious,
                "reason": f.reason,
            }
            for f in findings
        ],
        "rms_db": [round(float(x), 2) for x in rms_db.tolist()],
        "rms_hop_s": round(float(hop_s), 4),
    }


def print_findings(findings, mu, sigma, ref_db):
    print("\n========== 停顿/异常清单 ==========")
    print(f"{'#':>2} {'start':>7} {'end':>7} {'dur(ms)':>8} {'kind':>14} "
          f"{'z':>5} {'drop':>5}  备注")
    for i, f in enumerate(findings):
        flag = "★" if f.suspicious else " "
        print(f"{i:>2} {f.start:>7.3f} {f.end:>7.3f} {int(f.duration*1000):>8d} "
              f"{f.kind:>14} {f.z:>5.2f} {f.depth_db:>5.1f}  "
              f"{flag} {f.label!r:<8} {f.reason}")
    print(f"\n字时长统计: μ={mu*1000:.0f}ms σ={sigma*1000:.0f}ms; ref_db={ref_db:.1f}dB")
    sus = [f for f in findings if f.suspicious]
    print(f"\n>>> 可疑停顿 {len(sus)} 个:")
    for f in sus:
        print(f"   {f.start:.2f}s ~ {f.end:.2f}s  ({int(f.duration*1000)}ms)  "
              f"{f.kind}  {f.label}  {f.reason}")


# ============================================================================
# 命令行入口
# ============================================================================

def cmd_detect(args):
    """仅检测，不调 LLM。"""
    findings, mu, sigma, total_dur, text, ref_db, words, rms_db, hop_s = analyze(
        args.audio, args.language)
    print_findings(findings, mu, sigma, ref_db)

    if args.json:
        payload = findings_to_payload(
            args.audio, findings, mu, sigma, total_dur, text, ref_db,
            words, rms_db, hop_s)
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"\nJSON 已导出: {args.json}")


def cmd_judge(args):
    """对已有 JSON 跑 LLM 判定。"""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("需要 DASHSCOPE_API_KEY", file=sys.stderr)
        sys.exit(1)
    model = args.model or LLM_MODEL_DEFAULT
    print(f"LLM 模型: {model}")
    for path in args.json_files:
        p = Path(path)
        print(f"\n=== {p.name} ===")
        data = json.loads(p.read_text(encoding="utf-8"))
        judge_findings_in_data(data, model=model)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_run(args):
    """完整流程：detect → judge → 写 JSON。"""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("需要 DASHSCOPE_API_KEY", file=sys.stderr)
        sys.exit(1)

    findings, mu, sigma, total_dur, text, ref_db, words, rms_db, hop_s = analyze(
        args.audio, args.language)
    print_findings(findings, mu, sigma, ref_db)

    payload = findings_to_payload(
        args.audio, findings, mu, sigma, total_dur, text, ref_db,
        words, rms_db, hop_s)

    if not args.no_llm:
        print(f"\n[4/4] LLM 判定 (model={args.model or LLM_MODEL_DEFAULT}) ...")
        judge_findings_in_data(payload, model=args.model or LLM_MODEL_DEFAULT)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"\nJSON 已导出: {args.json}")


def main():
    ap = argparse.ArgumentParser(
        description="TTS 不自然停顿检测 + LLM 语义判定",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    # detect: 仅检测
    p_d = sp.add_parser("detect", help="仅做声学/对齐检测，不调 LLM")
    p_d.add_argument("audio")
    p_d.add_argument("--language", default="zh")
    p_d.add_argument("--json", default=None, help="导出 JSON 路径")
    p_d.set_defaults(func=cmd_detect)

    # judge: 仅 LLM 判定
    p_j = sp.add_parser("judge", help="对已有 JSON 跑 LLM 判定")
    p_j.add_argument("json_files", nargs="+", help="一个或多个 detect 输出的 JSON")
    p_j.add_argument("--model", default=None, help=f"LLM 模型（默认 {LLM_MODEL_DEFAULT}）")
    p_j.set_defaults(func=cmd_judge)

    # run: 一步到位
    p_r = sp.add_parser("run", help="检测 + LLM 判定（完整流程）")
    p_r.add_argument("audio")
    p_r.add_argument("--language", default="zh")
    p_r.add_argument("--json", default=None, help="导出 JSON 路径")
    p_r.add_argument("--model", default=None, help=f"LLM 模型（默认 {LLM_MODEL_DEFAULT}）")
    p_r.add_argument("--no-llm", action="store_true", help="跳过 LLM 判定")
    p_r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
