"""Lightweight agent skeleton.

Adding a capability = create a new file in `agent/tools/<name>.py` and
decorate the entry function with `@tool(...)`. Importing the package
auto-registers every tool, and the CLI exposes them as subcommands.
"""

# Importing the tools package triggers @tool registration.
from . import tools  # noqa: F401  (side effects)
from .registry import all_tools, get_tool, tool  # noqa: F401
