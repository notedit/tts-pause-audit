"""Tool: run_pauses — composite detect → judge in one call."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from ..llm import build_client, resolve_config
from ..registry import tool
from .detect_pauses import _print_payload, analyze_pauses
from .judge_pauses import judge_findings_in_payload


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio", nargs="+", help="audio path(s) or globs")
    p.add_argument("--language", default=None,
                   help="forced language (default: auto-detect)")
    p.add_argument("--json", default=None,
                   help="write payload(s) to this JSON path")
    p.add_argument("--no-llm", action="store_true",
                   help="skip LLM judgment")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)


@tool("run_pauses",
      "Detect pauses on audio and (unless --no-llm) judge each suspicious "
      "finding with an OpenAI-compatible LLM.",
      _add_args)
def run_pauses_cmd(args: argparse.Namespace) -> int:
    paths: list[str] = []
    for a in args.audio:
        if any(ch in a for ch in "*?["):
            paths.extend(sorted(glob.glob(a)))
        else:
            paths.append(a)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("no audio files", file=sys.stderr)
        return 1

    client = None
    model = None
    if not args.no_llm:
        api_key, base_url, model = resolve_config(api_key=args.api_key,
                                                  base_url=args.base_url,
                                                  model=args.model)
        if not api_key:
            print("ERROR: 未设置 OPENAI_API_KEY/DASHSCOPE_API_KEY 也未传 --api-key",
                  file=sys.stderr)
            return 1
        print(f"[llm] base_url={base_url}  model={model}")
        client = build_client(api_key=args.api_key, base_url=args.base_url)

    payloads = []
    for p in paths:
        payload = analyze_pauses(p, language=args.language)
        _print_payload(payload)
        if not args.no_llm:
            print("\n[LLM] judging suspicious findings ...")
            judge_findings_in_payload(payload, client=client, model=model)
        payloads.append(payload)

    if args.json:
        out = payloads[0] if len(payloads) == 1 else payloads
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False, indent=2)
        print(f"\n[json] {args.json}")
    return 0
