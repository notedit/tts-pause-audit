"""Tool: report_pauses — render detect+judge JSON as a markdown report.

Goal: a one-glance view that highlights *where the pauses look wrong*.

For each audio file we render:
  - the original transcript with every suspicious finding inlined and
    highlighted: ❌ for LLM-judged unnatural, ✓ for LLM-judged natural,
    ⏸ for not-yet-judged.
  - an ASCII timeline showing speech vs. silence vs. flagged regions,
    so problems jump out spatially.
  - a per-finding table (timing / signal / verdict / reason) where the
    "位置" column shows the boundary in its surrounding text context.
  - a per-file LLM-written summary (`--summary` to enable; needs API key).
  - a "🚩 需要关注" focused list of unnatural & pending findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..registry import tool

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

KIND_NAMES = {
    "inter_char": "字间静音",
    "char_too_long": "字时长拖长",
    "long_pause": "标点位长停顿",
    "in_word": "字内能量谷(破词)",
    "punct_hard": "硬标点",
    "punct_soft": "软标点",
    "leading": "首部静音",
    "trailing": "尾部静音",
}

CONTEXT_CHARS = 8  # how many chars of context to show on each side


# Status icon for a finding.
def _status_icon(f: dict) -> str:
    if f.get("llm_natural") is False:
        return "❌"
    if f.get("llm_natural") is True:
        return "✓"
    if f.get("suspicious"):
        return "⏸"
    return " "


def _verdict_cell(f: dict) -> str:
    if "llm_natural" not in f:
        return "⏸ 待判" if f.get("suspicious") else "—"
    return "❌ **不自然**" if not f["llm_natural"] else "✓ 自然"


def _signal_cell(f: dict) -> str:
    bits = []
    z = float(f.get("z", 0.0))
    drop = float(f.get("depth_db", 0.0))
    if abs(z) >= 0.01:
        bits.append(f"z={z:+.1f}")
    if drop >= 0.5:
        bits.append(f"drop={drop:.0f}dB")
    return " ".join(bits) if bits else "—"


def _label_to_pair(label: str) -> tuple[str, str]:
    if "→" in label:
        a, b = label.split("→", 1)
        return a, b
    return label, ""


def _locate_boundary_in_text(text: str, prev_w: str, next_w: str) -> int:
    """Return the index in `text` between prev_w and next_w (i.e. just
    after prev_w). -1 if not found."""
    if not prev_w:
        return 0
    # Use prev_w + next_w to disambiguate when prev_w occurs multiple times.
    if next_w and next_w not in ("<EOS>", "<BOS>"):
        # tolerate punctuation between them
        for i in range(len(text)):
            if text[i:i + len(prev_w)] != prev_w:
                continue
            j = i + len(prev_w)
            k = j
            while k < len(text) and text[k] in "，。！？、；：,.;:!? \t\n":
                k += 1
            if text[k:k + len(next_w)] == next_w:
                return j
    i = text.find(prev_w)
    if i >= 0:
        return i + len(prev_w)
    return -1


def _context_cell(text: str, f: dict) -> str:
    """Render the boundary with surrounding context, marker inserted.

    `…昨晚 收 到 快 递 ⏸ 发 现 是 男 朋…`
    """
    prev_w, next_w = _label_to_pair(f["label"])
    if f["kind"] == "leading":
        snip = text[:CONTEXT_CHARS * 2]
        ellipsis_r = "…" if len(snip) < len(text) else ""
        return f"⏮ {snip}{ellipsis_r}"
    if f["kind"] == "trailing":
        snip = text[-CONTEXT_CHARS * 2:]
        ellipsis_l = "…" if len(snip) < len(text) else ""
        return f"{ellipsis_l}{snip} ⏭"
    if f["kind"] == "in_word":
        i = text.find(prev_w)
        if i < 0:
            return f"`{prev_w}` 内"
        s = max(0, i - CONTEXT_CHARS)
        e = min(len(text), i + len(prev_w) + CONTEXT_CHARS)
        before = ("…" if s > 0 else "") + text[s:i]
        after = text[i + len(prev_w):e] + ("…" if e < len(text) else "")
        return f"{before}『{prev_w}』{after}"

    pos = _locate_boundary_in_text(text, prev_w, next_w)
    if pos < 0:
        return f"`{prev_w}` → `{next_w}`"
    s = max(0, pos - CONTEXT_CHARS)
    e = min(len(text), pos + CONTEXT_CHARS)
    before = ("…" if s > 0 else "") + text[s:pos]
    after = text[pos:e] + ("…" if e < len(text) else "")
    icon = _status_icon(f)
    marker = f" {icon}{int(f['duration_ms'])}ms "
    return f"{before}{marker}{after}"


# ---------------------------------------------------------------------------
# Visual transcript: original text with stop-marker boxes inserted.
# ---------------------------------------------------------------------------

def _annotate_transcript(payload: dict) -> str:
    text: str = payload["text"]
    findings = payload.get("findings", [])
    inserts: list[tuple[int, str, int]] = []

    for f in findings:
        if f["kind"] in ("punct_soft", "punct_hard"):
            continue
        if f["kind"] == "leading":
            inserts.append((0, f"⏮{int(f['duration_ms'])}ms ", -1))
            continue
        if f["kind"] == "trailing":
            inserts.append((len(text), f" ⏭{int(f['duration_ms'])}ms", 99999))
            continue
        if f["kind"] == "in_word":
            i = text.find(f["label"])
            if i < 0:
                continue
            inserts.append((i, f" ❌{int(f['duration_ms'])}ms内 ", i))
            continue

        prev_w, next_w = _label_to_pair(f["label"])
        pos = _locate_boundary_in_text(text, prev_w, next_w)
        if pos < 0:
            continue
        ms = int(f["duration_ms"])
        if f.get("llm_natural") is False:
            sym = f" ❌{ms}ms "
        elif f.get("llm_natural") is True:
            sym = f" ✓{ms}ms "
        elif f.get("suspicious"):
            sym = f" ⏸{ms}ms "
        else:
            sym = f" ·{ms}ms "
        adj = pos
        while adj < len(text) and text[adj] in "，。！？、；：,.!?;: \t\n":
            adj += 1
        inserts.append((adj, sym, pos))

    inserts.sort(key=lambda x: (x[0], x[2]))
    out: list[str] = []
    cursor = 0
    for pos, mark, _key in inserts:
        out.append(text[cursor:pos])
        out.append(mark)
        cursor = pos
    out.append(text[cursor:])
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# ASCII timeline.
# ---------------------------------------------------------------------------

def _ascii_timeline(payload: dict, width: int = 80) -> list[str]:
    dur = float(payload.get("duration", 0.0))
    if dur <= 0:
        return []

    findings = payload.get("findings", [])
    words = payload.get("words", [])

    track = list(" " * width)
    if words:
        for w in words:
            s = max(0, int(w["start"] / dur * width))
            e = min(width, int(w["end"] / dur * width) + 1)
            for k in range(s, e):
                track[k] = "·"

    overlay = list(track)
    legend_used = set()
    prio = {"X": 5, "?": 4, "=": 3, "|": 2, "_": 1, "·": 0, " ": 0}
    for f in findings:
        s = max(0, int(f["start"] / dur * width))
        e = min(width, max(s + 1, int(f["end"] / dur * width)))
        if f["kind"] in ("punct_soft", "punct_hard"):
            ch = "|"
        elif f["kind"] in ("leading", "trailing"):
            ch = "_"
        elif f.get("llm_natural") is False:
            ch = "X"
        elif f.get("llm_natural") is True:
            ch = "="
        elif f.get("suspicious"):
            ch = "?"
        else:
            continue
        legend_used.add(ch)
        for k in range(s, e):
            if prio[ch] >= prio.get(overlay[k], 0):
                overlay[k] = ch

    axis = [" "] * width
    label = [" "] * width
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        idx = min(width - 1, int(frac * (width - 1)))
        axis[idx] = "|"
        txt = f"{frac * dur:.1f}s"
        for k, c in enumerate(txt):
            j = idx + k - len(txt) // 2
            if 0 <= j < width:
                label[j] = c

    legend_parts = []
    if "X" in legend_used:
        legend_parts.append("X=不自然")
    if "?" in legend_used:
        legend_parts.append("?=待判")
    if "=" in legend_used:
        legend_parts.append("==自然")
    if "|" in legend_used:
        legend_parts.append("|=标点")
    if "_" in legend_used:
        legend_parts.append("_=首尾静音")
    legend_parts.append("·=正常发声")

    return [
        "```",
        "时间轴: " + "".join(overlay),
        "       " + "".join(axis),
        "       " + "".join(label),
        "图例: " + " · ".join(legend_parts),
        "```",
    ]


# ---------------------------------------------------------------------------
# Per-file LLM summary (optional).
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """你是 TTS 音频质量审计员。给定一段中文音频的检测结果，请用 1-2 句话总结整句的整体停顿表现：
- 是否有明显的句中异常停顿？发生在哪个位置（具体到 X 与 Y 之间）？
- 总体可不可用（"流畅"/"基本可用，仅 N 处刺耳"/"严重破碎"）？
返回严格 JSON：{"summary":"<≤80 字中文整句总结>","verdict":"smooth|minor|broken"}"""


def _build_summary_user_prompt(payload: dict) -> str:
    findings = payload.get("findings", [])
    sus = [f for f in findings if f.get("suspicious")]
    lines = [f'原文: "{payload["text"]}"', f"时长: {payload.get('duration', 0):.2f}s",
             f"可疑停顿 {len(sus)} 处："]
    for f in sus:
        verdict = "❌不自然" if f.get("llm_natural") is False \
            else ("✓自然" if f.get("llm_natural") is True else "⏸待判")
        reason = f.get("llm_reason") or f.get("reason", "")
        lines.append(f"  · {f['label']} ({f['kind']}, {int(f['duration_ms'])}ms) → {verdict} :: {reason}")
    return "\n".join(lines)


def _generate_per_file_summary(payload: dict, *, client, model: str) -> dict:
    user = _build_summary_user_prompt(payload)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or ""
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        j = json.loads(content.strip())
        return {
            "summary": str(j.get("summary", ""))[:200],
            "verdict": str(j.get("verdict", "")).lower(),
        }
    except Exception as e:  # noqa: BLE001
        return {"summary": f"(总结失败: {e})", "verdict": "unknown"}


# ---------------------------------------------------------------------------
# Per-file rendering
# ---------------------------------------------------------------------------

def _render_file(payload: dict) -> str:
    findings: list[dict] = payload.get("findings", [])
    sus_findings = [f for f in findings if f.get("suspicious")]
    n_unnat = sum(1 for f in sus_findings if f.get("llm_natural") is False)
    n_nat = sum(1 for f in sus_findings if f.get("llm_natural") is True)
    n_pending = sum(1 for f in sus_findings if "llm_natural" not in f)

    if n_unnat:
        health = "🔴 有不自然"
    elif n_pending:
        health = "🟡 待判"
    elif sus_findings:
        health = "🟢 全部自然"
    else:
        health = "⚪ 无可疑"

    text: str = payload["text"]

    lines: list[str] = []
    lines.append(f"## {payload['audio']}  {health}")
    lines.append("")
    lines.append(
        f"**总览**  ·  时长 `{payload.get('duration', 0):.2f}s`  ·  "
        f"语言 `{payload.get('language')}`  ·  "
        f"❌ 不自然 **{n_unnat}**  ·  ✓ 自然 {n_nat}  ·  ⏸ 待判 {n_pending}  "
        f"·  共 {len(sus_findings)} 处可疑 / {len(findings)} 项"
    )
    lines.append("")
    lines.append(
        f"**字时长**  μ={payload['char_dur_mu_ms']}ms  σ={payload['char_dur_sigma_ms']}ms"
        f"  ·  **能量基线** ref_db={payload['ref_db']}dB"
    )
    lines.append("")

    # LLM-generated per-file summary
    if "_summary" in payload:
        s = payload["_summary"]
        verdict_emoji = {"smooth": "🟢 流畅", "minor": "🟡 基本可用",
                         "broken": "🔴 严重破碎"}.get(s.get("verdict", ""), "⚪")
        lines.append("### 🧠 整句总结")
        lines.append("")
        lines.append(f"> **{verdict_emoji}** — {s.get('summary', '')}")
        lines.append("")

    lines.append("### 📝 原文标注（一眼看停顿）")
    lines.append("")
    lines.append(f"> {_annotate_transcript(payload)}")
    lines.append(">")
    lines.append("> 图例：`❌ NNNms` 不自然 ｜ `✓ NNNms` 自然 ｜ "
                 "`⏸ NNNms` 待判 ｜ `⏮/⏭` 首/尾静音")
    lines.append("")

    lines.append("### 🎵 音频时间轴")
    lines.append("")
    lines.extend(_ascii_timeline(payload))
    lines.append("")

    issues = [f for f in sus_findings if f.get("llm_natural") is False]
    pending = [f for f in sus_findings if "llm_natural" not in f]
    if issues or pending:
        lines.append("### 🚩 需要关注")
        lines.append("")
        for f in issues:
            label = f["label"].replace("→", " → ")
            t = f"{f['start']:.2f}s"
            reason = f.get("llm_reason") or f.get("reason", "")
            ctx = _context_cell(text, f)
            lines.append(
                f"- ❌ **{label}** @ {t}  ·  `{f['kind']}`  ·  "
                f"{int(f['duration_ms'])}ms  ·  {reason}"
            )
            lines.append(f"  - 上下文: {ctx}")
        for f in pending:
            label = f["label"].replace("→", " → ")
            t = f"{f['start']:.2f}s"
            reason = f.get("reason", "")
            ctx = _context_cell(text, f)
            lines.append(
                f"- ⏸ **{label}** @ {t}  ·  `{f['kind']}`  ·  "
                f"{int(f['duration_ms'])}ms  ·  {reason}（未跑 LLM）"
            )
            lines.append(f"  - 上下文: {ctx}")
        lines.append("")
    elif sus_findings:
        lines.append("> ✅ 所有可疑停顿都被 LLM 判定为自然。")
        lines.append("")

    lines.append("<details><summary>📊 全部 finding 明细表</summary>")
    lines.append("")
    lines.append("| 状 | 起 | 止 | 时长 | 类型 | 上下文（含位置标记） | 信号 | LLM 判定 | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for f in findings:
        flag = _status_icon(f)
        kind_disp = f"`{f['kind']}` ({KIND_NAMES.get(f['kind'], '?')})"
        ctx = _context_cell(text, f).replace("|", "\\|")
        reason_full = f.get("reason", "")
        llm_reason = f.get("llm_reason", "")
        notes = []
        if reason_full and reason_full != "—":
            notes.append(reason_full)
        if llm_reason:
            notes.append(f"**LLM:** {llm_reason}")
        notes_cell = "<br>".join(notes).replace("|", "\\|") if notes else "—"
        lines.append(
            f"| {flag} | {f['start']:.2f}s | {f['end']:.2f}s | "
            f"{int(f['duration_ms'])}ms | {kind_disp} | {ctx} | "
            f"{_signal_cell(f)} | {_verdict_cell(f)} | {notes_cell} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def render_markdown(payloads: list[dict]) -> str:
    n_files = len(payloads)
    n_sus = 0
    n_unnat = 0
    n_nat = 0
    for p in payloads:
        for f in p["findings"]:
            if not f.get("suspicious"):
                continue
            n_sus += 1
            if f.get("llm_natural") is False:
                n_unnat += 1
            elif f.get("llm_natural") is True:
                n_nat += 1

    out: list[str] = []
    out.append("# 🎯 Qwen3-ASR 停顿检测报告")
    out.append("")
    out.append("## 总览")
    out.append("")
    out.append("| 指标 | 数值 |")
    out.append("|---|---|")
    out.append(f"| 文件数 | {n_files} |")
    out.append(f"| 可疑停顿总数 | {n_sus} |")
    out.append(f"| ❌ 不自然 | **{n_unnat}** |")
    out.append(f"| ✓ 自然 | {n_nat} |")
    out.append(f"| ⏸ 待判 | {n_sus - n_unnat - n_nat} |")
    out.append("")

    out.append("## 文件汇总")
    out.append("")
    out.append("| 文件 | 时长 | ❌ | ✓ | ⏸ | 健康度 | 一句话总结 |")
    out.append("|---|---|---|---|---|---|---|")
    for p in payloads:
        sus = [f for f in p["findings"] if f.get("suspicious")]
        unnat = sum(1 for f in sus if f.get("llm_natural") is False)
        nat = sum(1 for f in sus if f.get("llm_natural") is True)
        pend = sum(1 for f in sus if "llm_natural" not in f)
        if unnat:
            health = "🔴"
        elif pend:
            health = "🟡"
        elif sus:
            health = "🟢"
        else:
            health = "⚪"
        s = (p.get("_summary") or {}).get("summary", "")
        s = s.replace("|", "\\|")
        if not s and sus:
            # quick fallback summary if no LLM summary was generated
            top = next((f for f in sus if f.get("llm_natural") is False), None)
            if top:
                s = (f"在 “{top['label']}” 处出现 ❌不自然停顿 "
                     f"({int(top['duration_ms'])}ms)").replace("|", "\\|")
            else:
                s = f"{len(sus)} 处可疑（待人工/LLM 判定）"
        out.append(
            f"| {p['audio']} | {p.get('duration', 0):.2f}s | **{unnat}** | "
            f"{nat} | {pend} | {health} | {s} |"
        )
    out.append("")
    out.append("---")
    out.append("")

    for p in payloads:
        out.append(_render_file(p))
        out.append("---")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("json_files", nargs="+",
                   help="detect_pauses / run_pauses 输出的 JSON")
    p.add_argument("--md", default=None,
                   help="写入指定 markdown 文件（缺省打印到 stdout）")
    p.add_argument("--summary", action="store_true",
                   help="为每个文件调 LLM 生成 1-2 句整句总结")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)


@tool("report_pauses",
      "将 detect_pauses / run_pauses 输出的 JSON 渲染为可视化 markdown 报告。"
      "可选 --summary 让 LLM 给每个文件写一句整句总结。",
      _add_args)
def report_pauses_cmd(args: argparse.Namespace) -> int:
    payloads: list[dict] = []
    for path in args.json_files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            payloads.extend(data)
        else:
            payloads.append(data)
    if not payloads:
        print("no payloads", file=sys.stderr)
        return 1

    if args.summary:
        from ..llm import build_client, resolve_config
        api_key, base_url, model = resolve_config(api_key=args.api_key,
                                                  base_url=args.base_url,
                                                  model=args.model)
        if not api_key:
            print("ERROR: --summary 需要 OPENAI_API_KEY/DASHSCOPE_API_KEY 或 --api-key",
                  file=sys.stderr)
            return 1
        print(f"[llm] base_url={base_url}  model={model} (writing per-file summaries)",
              file=sys.stderr)
        client = build_client(api_key=args.api_key, base_url=args.base_url)
        for p in payloads:
            p["_summary"] = _generate_per_file_summary(p, client=client, model=model)

    md = render_markdown(payloads)
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
        print(f"[md] {args.md}")
    else:
        print(md)
    return 0

