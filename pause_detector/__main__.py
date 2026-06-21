"""tts-pause-audit CLI entry point.

Usage:
  tts-pause-audit <tool> [args...]
  tts-pause-audit --list
  python -m pause_detector <tool> [args...]
"""

import argparse
import sys

from . import all_tools, get_tool


def main() -> int:
    # Trigger @tool registration. We do this lazily here (not in
    # pause_detector/__init__.py) so importing helpers like
    # `pause_detector.text_align` does not pull in torch / qwen-asr.
    from . import tools  # noqa: F401  (side effects)

    parser = argparse.ArgumentParser(
        prog="tts-pause-audit",
        description="Qwen3-ASR-based TTS pause auditor.",
    )
    sub = parser.add_subparsers(dest="cmd")
    parser.add_argument("--list", action="store_true",
                        help="list registered tools and exit")

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
