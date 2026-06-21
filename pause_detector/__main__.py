"""Agent CLI entry point.

Usage:
  python -m agent <tool> [args...]
  python -m agent --list
"""

import argparse
import sys

from . import all_tools, get_tool


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent", description="Qwen3-ASR audio analysis agent")
    sub = parser.add_subparsers(dest="cmd")
    parser.add_argument("--list", action="store_true", help="list registered tools and exit")

    # Register every tool as a subcommand.
    for spec in all_tools():
        sp = sub.add_parser(spec.name, help=spec.summary)
        if spec.add_arguments:
            spec.add_arguments(sp)

    args = parser.parse_args()
    if args.list or args.cmd is None:
        print("Available tools:")
        for spec in all_tools():
            print(f"  {spec.name:<20} {spec.summary}")
        return 0

    spec = get_tool(args.cmd)
    return int(spec.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
