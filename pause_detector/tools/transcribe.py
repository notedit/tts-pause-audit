"""Tool: transcribe — Qwen3-ASR transcription, optionally with timestamps."""

import argparse
import json
import os

from ..models import get_asr
from ..registry import tool


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio", nargs="+", help="audio path(s)")
    p.add_argument("--language", default=None, help="forced language (e.g. Chinese, English) or omit for auto")
    p.add_argument("--timestamps", action="store_true", help="return character/word time stamps")
    p.add_argument("--json", action="store_true", help="emit JSON")


@tool("transcribe", "Transcribe audio file(s) with Qwen3-ASR.", _add_args)
def transcribe_cmd(args: argparse.Namespace) -> int:
    asr = get_asr(with_aligner=args.timestamps)
    paths = list(args.audio)
    results = asr.transcribe(
        audio=paths,
        language=[args.language] * len(paths) if args.language else None,
        return_time_stamps=args.timestamps,
    )

    out = []
    for p, r in zip(paths, results):
        item = {"file": os.path.basename(p), "language": r.language, "text": r.text}
        if args.timestamps and r.time_stamps:
            item["time_stamps"] = [
                {"text": t.text, "start": t.start_time, "end": t.end_time}
                for t in r.time_stamps
            ]
        out.append(item)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for item in out:
            print(f"\n===== {item['file']}")
            print(f"  language: {item['language']!r}")
            print(f"  text    : {item['text']!r}")
            if "time_stamps" in item:
                print(f"  units   : {len(item['time_stamps'])}")
    return 0
