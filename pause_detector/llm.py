"""OpenAI-compatible LLM client for pause judgment.

Resolves api_key / base_url / model with priority CLI > env > defaults.
Defaults aim at DashScope's OpenAI-compatible endpoint + qwen-plus.

Usage:
    client, model = build_client(api_key=..., base_url=..., model=...)
    verdict = judge_pause(client, model, system_prompt, user_prompt)
    # verdict -> {"natural": bool, "reason": str}
"""

from __future__ import annotations

import json
import os
import time

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def resolve_config(api_key: str | None = None,
                   base_url: str | None = None,
                   model: str | None = None
                   ) -> tuple[str | None, str, str]:
    """Pick api_key / base_url / model with priority: arg > env > default."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    model = model or os.environ.get("PAUSE_LLM_MODEL") or DEFAULT_MODEL
    return api_key, base_url, model


def build_client(api_key: str | None = None,
                 base_url: str | None = None,
                 timeout: float = 30.0):
    """Construct an openai.OpenAI client. Raises if no api_key is resolved."""
    from openai import OpenAI  # imported lazily so detect_pauses works without openai

    key, url, _ = resolve_config(api_key=api_key, base_url=base_url)
    if not key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY (or DASHSCOPE_API_KEY) "
            "or pass --api-key.")
    return OpenAI(api_key=key, base_url=url, timeout=timeout)


def _strip_json_fence(s: str) -> str:
    """Tolerate ```json ... ``` fences."""
    s = s.strip()
    if "```" in s:
        parts = s.split("```")
        # take the longest middle segment
        body = max(parts[1:-1] or parts, key=len)
        if body.startswith("json"):
            body = body[4:]
        s = body.strip()
    return s


def judge_pause(client, model: str, system: str, user: str, *,
                max_retry: int = 3, max_tokens: int = 128) -> dict:
    """Call the LLM once for a single finding. Returns {"natural", "reason"}.

    Always returns a dict; on persistent failure: natural=False with reason
    "LLM 失败: <err>".
    """
    last_err: Exception | None = None
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content or ""
            j = json.loads(_strip_json_fence(content))
            return {
                "natural": bool(j.get("natural", False)),
                "reason": str(j.get("reason", "")).strip()[:50],
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    return {"natural": False, "reason": f"LLM 失败: {last_err}"}
