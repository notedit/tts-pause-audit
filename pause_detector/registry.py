"""Tool registry. Each tool is a callable + metadata."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    summary: str
    add_arguments: Callable[[Any], None] | None = None  # add argparse args


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, summary: str, add_arguments: Callable | None = None):
    """Decorator: register a function as a CLI-invocable agent tool."""
    def _wrap(fn: Callable):
        if name in REGISTRY:
            raise ValueError(f"tool {name!r} already registered")
        REGISTRY[name] = ToolSpec(name=name, fn=fn, summary=summary, add_arguments=add_arguments)
        return fn
    return _wrap


def get_tool(name: str) -> ToolSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def all_tools() -> list[ToolSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]
