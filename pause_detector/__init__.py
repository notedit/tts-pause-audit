"""Lightweight tool skeleton.

Adding a capability = create a new file in `pause_detector/tools/<name>.py`
and decorate the entry function with `@tool(...)`. The CLI auto-registers
every tool by importing the `tools` package on demand (see `__main__.py`).

We intentionally do NOT import `tools/` from this `__init__.py` — importing
the tools triggers heavy imports (torch, qwen-asr) that aren't needed for
pure utility usage (e.g. `from pause_detector.text_align import ...`).
"""

from .registry import all_tools, get_tool, tool  # noqa: F401
