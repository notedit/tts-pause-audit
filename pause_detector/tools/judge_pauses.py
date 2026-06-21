"""Tool: judge_pauses — LLM judgment for pause findings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..llm import build_client, judge_pause, resolve_config
from ..prompts import PAUSE_SYSTEM_PROMPT, build_pause_user_prompt
from ..registry import tool
from ..text_align import find_position_after


def _split_label(label: str) -> tuple[str, str]:
    """Findings use 'prev→next' labels; fall back to (label, '')."""
    if "→" in label:
        a, b = label.split("→", 1)
        return a, b
    return label, ""


def _resolve_prev_next(finding: dict, words: list) -> tuple[str, str]:
    prev_word, next_word = _split_label(finding["label"])
    kind = finding["kind"]
    if prev_word and next_word:
        return prev_word, next_word

    # legacy single-label char_too_long: stitch with neighbour from words list
    if kind == "char_too_long" and prev_word:
        idx = -1
        for i, w in enumerate(words):
            if w["word"] == prev_word and abs(w["end"] - finding["end"]) < 0.05:
                idx = i
                break
        if idx >= 0 and idx + 1 < len(words):
            return prev_word, words[idx + 1]["word"]
        return prev_word, ""
    if kind == "in_word":
        return prev_word, prev_word
    return prev_word, next_word


def judge_findings_in_payload(payload: dict, *, client, model: str,
                              verbose: bool = True) -> dict:
    text = payload["text"]
    words = payload["words"]
    findings: list[dict] = payload["findings"]
    sus_idx = [i for i, f in enumerate(findings) if f.get("suspicious")]
    if verbose:
        print(f"  原文: {text}")
        print(f"  可疑停顿: {len(sus_idx)} 处")

    for i in sus_idx:
        f = findings[i]
        prev_w, next_w = _resolve_prev_next(f, words)
        pos = find_position_after(text, prev_w, next_w)
        if pos < 0:
            f["llm_natural"] = False
            f["llm_reason"] = "无法定位"
            f["llm_prev"] = prev_w
            f["llm_next"] = next_w
            continue
        user = build_pause_user_prompt(
            text, prev_w, next_w, pos,
            duration_ms=int(f.get("duration_ms", 0)),
            kind=f.get("kind", ""),
        )
        verdict = judge_pause(client, model, PAUSE_SYSTEM_PROMPT, user)
        f["llm_natural"] = bool(verdict["natural"])
        f["llm_reason"] = verdict["reason"]
        f["llm_prev"] = prev_w
        f["llm_next"] = next_w
        if verbose:
            flag = "✓自然" if verdict["natural"] else "✗不自然"
            print(f"   [{f['start']:.2f}s] {prev_w}|{next_w}  →  "
                  f"{flag}: {verdict['reason']}")
    return payload


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("json_files", nargs="+", help="detect_pauses 输出的 JSON 文件")
    p.add_argument("--api-key", default=None,
                   help="覆盖 OPENAI_API_KEY/DASHSCOPE_API_KEY")
    p.add_argument("--base-url", default=None,
                   help="覆盖 OPENAI_BASE_URL（默认 DashScope OpenAI 兼容模式）")
    p.add_argument("--model", default=None,
                   help="覆盖 PAUSE_LLM_MODEL（默认 qwen-plus）")


@tool("judge_pauses",
      "Send each suspicious finding in a detect_pauses JSON to an OpenAI-"
      "compatible LLM and write back llm_natural / llm_reason in place.",
      _add_args)
def judge_pauses_cmd(args: argparse.Namespace) -> int:
    api_key, base_url, model = resolve_config(api_key=args.api_key,
                                              base_url=args.base_url,
                                              model=args.model)
    if not api_key:
        print("ERROR: 未设置 OPENAI_API_KEY/DASHSCOPE_API_KEY 也未传 --api-key",
              file=sys.stderr)
        return 1
    print(f"[llm] base_url={base_url}  model={model}")
    client = build_client(api_key=args.api_key, base_url=args.base_url)

    for path in args.json_files:
        p = Path(path)
        print(f"\n=== {p.name} ===")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                judge_findings_in_payload(item, client=client, model=model)
        else:
            judge_findings_in_payload(payload, client=client, model=model)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print(f"[json] 已更新: {p}")
    return 0
