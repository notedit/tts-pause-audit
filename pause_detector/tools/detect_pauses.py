"""Tool: detect_pauses — TTS-style abnormal pause detection.

Pipeline:
  1. Qwen3-ASR + Qwen3-ForcedAligner-0.6B → text + char-level timestamps.
     (Qwen3-FA's tokens are spoken units only; punctuation is dropped.)
  2. RMS-dB envelope (p90 baseline) + valley detection.
  3. Joint analysis emits 5 signal classes:
       S1 inter_char     字间静音 (能量谷 + 文本无标点)
       S2 char_too_long  字时长 z-score 异常 (z ≥ 2.0)
       S3 in_word        谷完全落在字内部 → 破词
       S4 long_pause     标点位上的长停顿 (≥260ms gap)
       S5 leading/trailing 首尾静音 (suspicious=False)

The output JSON shape mirrors pause_detect.py::findings_to_payload so any
existing visualization can consume it; LLM judgment is added by the
companion `judge_pauses` tool.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

from ..audio_utils import compute_rms_db_p90, find_valleys_p90
from ..models import get_asr
from ..registry import tool
from ..text_align import (
    align_text_to_ts,
    is_hard_punct,
    is_punct_char,
)

# ---- Calibrated thresholds (mirror pause_detect.py constants) ---------------
CHAR_DUR_Z = 2.0
VALLEY_DROP_DB = 10
VALLEY_MIN_MS = 80
INTER_CHAR_MIN_MS = 260
INTER_CHAR_MIN_DROP_DB = 12
IN_WORD_MIN_MS = 120
IN_WORD_MIN_DROP_DB = 18
PUNCT_PAUSE_MIN_MS = 260


# ---- Finding ----------------------------------------------------------------
@dataclass
class Finding:
    start: float
    end: float
    duration: float
    kind: str
    label: str
    z: float = 0.0
    depth_db: float = 0.0
    suspicious: bool = False
    reason: str = ""


# ---- Helpers ----------------------------------------------------------------
def _overlaps(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def _punct_between(text: str, l_end: int, r_start: int) -> str:
    """Return the punctuation chars (whitespace-stripped) between
    character spans [l_end, r_start) in the transcribed text."""
    if l_end < 0 or r_start < 0 or r_start <= l_end:
        return ""
    return "".join(c for c in text[l_end:r_start]
                   if is_punct_char(c) and not c.isspace())


def _annotated_text(text: str, records: list[dict],
                    valleys: list[tuple[float, float, float]],
                    findings: list[Finding]) -> str:
    """Render the transcript with valley markers inserted inline.

    <NNNms>     pause at a punctuation boundary (normal)
    <!NNNms!>   abnormal in-sentence pause (no punctuation between)
    """
    abnormal_spans = [(f.start, f.end) for f in findings
                      if f.suspicious and f.kind in ("inter_char", "char_too_long")]

    inserts: list[tuple[int, str]] = []
    for vs, ve, _depth in valleys:
        dur_ms = (ve - vs) * 1000.0
        # locate bracketing records by time
        left = None
        for r in records:
            if r["start_s"] <= vs:
                left = r
            else:
                break
        right = None
        for r in records:
            if r["end_s"] >= ve:
                right = r
                break
        if left is None or right is None:
            continue
        if left is right:
            i = records.index(left)
            if i + 1 >= len(records):
                continue
            right = records[i + 1]

        is_abnormal = any(_overlaps(vs, ve, a, b) > 0 for a, b in abnormal_spans)
        pos = left["span"][1]
        if pos < 0:
            continue
        if is_abnormal:
            mark = f"<!{dur_ms:.0f}ms!>"
        else:
            mark = f"<{dur_ms:.0f}ms>"
            # push past trailing punctuation so reader sees "蛋糕。<320ms>虽然"
            while pos < len(text) and is_punct_char(text[pos]) and not text[pos].isspace():
                pos += 1
        inserts.append((pos, mark))

    inserts.sort(key=lambda x: x[0])
    out: list[str] = []
    cursor = 0
    for pos, mark in inserts:
        out.append(text[cursor:pos])
        out.append(mark)
        cursor = pos
    out.append(text[cursor:])
    return "".join(out)


# ---- Core analysis ----------------------------------------------------------
def analyze_pauses(audio_path: str, *, language: str | None = None) -> dict:
    """Run the full ASR + RMS + 5-signal analysis. Returns a JSON-able dict."""
    asr = get_asr(with_aligner=True)
    res = asr.transcribe(audio=audio_path, language=language,
                         return_time_stamps=True)[0]
    text = res.text
    ts = res.time_stamps or []
    records = align_text_to_ts(text, ts)

    rms_db, hop_s, win_s, ref_db = compute_rms_db_p90(audio_path)
    valleys = find_valleys_p90(rms_db, hop_s, win_s, ref_db,
                               drop_db=VALLEY_DROP_DB, min_ms=VALLEY_MIN_MS)
    total_dur = (len(rms_db) * hop_s + win_s) if len(rms_db) else 0.0
    if total_dur == 0.0 and ts:
        total_dur = float(ts[-1].end_time)

    chars = records  # all are spoken units (FA drops punct)
    durs = np.array([w["end_s"] - w["start_s"] for w in chars]) if chars else np.array([])
    mu = float(durs.mean()) if len(durs) else 0.0
    sigma = float(durs.std()) if len(durs) else 0.0

    findings: list[Finding] = []

    # ---------- S5 leading / trailing silence ----------
    if chars and chars[0]["start_s"] > 0.0:
        d = chars[0]["start_s"]
        findings.append(Finding(0.0, d, d, "leading", "<BOS>",
                                reason=f"首部静音 {int(d*1000)}ms"))
    if chars and chars[-1]["end_s"] < total_dur:
        d = total_dur - chars[-1]["end_s"]
        findings.append(Finding(chars[-1]["end_s"], total_dur, d,
                                "trailing", "<EOS>",
                                reason=f"尾部静音 {int(d*1000)}ms"))

    # ---------- S1 inter_char + S3 in_word (per-valley) ----------
    inter_intervals: list[tuple[float, float]] = []

    for vs, ve, depth in valleys:
        d_ms = (ve - vs) * 1000.0

        # S3 in_word — valley fully inside a single char box
        in_word_w = next(
            (w for w in chars
             if vs > w["start_s"] + 0.06 and ve < w["end_s"] - 0.06),
            None,
        )
        if in_word_w is not None and d_ms >= IN_WORD_MIN_MS and depth >= IN_WORD_MIN_DROP_DB:
            findings.append(Finding(
                vs, ve, ve - vs, "in_word",
                in_word_w["text"], depth_db=depth, suspicious=True,
                reason=(f"字 [{in_word_w['text']}] 内部能量谷 "
                        f"{int(d_ms)}ms (drop={depth:.0f}dB)"),
            ))
            continue

        # S1 inter_char — locate prev_w (rightmost char with start ≤ vs+0.05),
        # then next_w = chars[prev_idx + 1].
        prev_w, prev_idx = None, -1
        best_dist = 1e9
        for i, w in enumerate(chars):
            if w["start_s"] < vs + 0.05:
                dist = abs(w["end_s"] - vs)
                if dist < best_dist:
                    best_dist = dist
                    prev_w, prev_idx = w, i
        next_w = chars[prev_idx + 1] if 0 <= prev_idx < len(chars) - 1 else None
        if prev_w is None or next_w is None:
            continue

        # Skip valleys that sit at a punctuation boundary — those become S4.
        if _punct_between(text, prev_w["span"][1], next_w["span"][0]):
            continue

        # Valley must straddle prev_w's midpoint (the silence is at its tail)
        if vs < (prev_w["start_s"] + prev_w["end_s"]) / 2 - 0.03:
            continue

        prev_z = ((prev_w["end_s"] - prev_w["start_s"] - mu) / sigma
                  if sigma > 1e-6 else 0.0)
        cond_long = d_ms >= INTER_CHAR_MIN_MS and depth >= INTER_CHAR_MIN_DROP_DB
        cond_z = prev_z >= CHAR_DUR_Z
        if cond_long or cond_z:
            reasons = [f"字间能量谷 {int(d_ms)}ms (drop={depth:.0f}dB)"]
            if cond_z:
                reasons.append(f"前字 [{prev_w['text']}] z={prev_z:+.1f} 拖长")
            findings.append(Finding(
                vs, ve, ve - vs, "inter_char",
                f"{prev_w['text']}→{next_w['text']}",
                z=prev_z, depth_db=depth, suspicious=True,
                reason="; ".join(reasons),
            ))
            inter_intervals.append((vs, ve))

    def _overlaps_inter(s: float, e: float) -> bool:
        return any(min(e, b) - max(s, a) > 0.05 for a, b in inter_intervals)

    # ---------- S4 long_pause / punct_hard / punct_soft (emit BEFORE S2 so
    # the char-stretch dedup sees them) ----------
    for k in range(len(chars) - 1):
        cur, nxt = chars[k], chars[k + 1]
        between = text[cur["span"][1]:nxt["span"][0]] if cur["span"][1] >= 0 and nxt["span"][0] >= 0 else ""
        punct = "".join(c for c in between if is_punct_char(c) and not c.isspace())
        if not punct:
            continue
        gap = nxt["start_s"] - cur["end_s"]
        gap_ms = gap * 1000.0
        label = f"{cur['text']}→{nxt['text']}"
        if gap_ms >= PUNCT_PAUSE_MIN_MS:
            findings.append(Finding(
                cur["end_s"], nxt["start_s"], gap, "long_pause", label,
                suspicious=True,
                reason=f"标点位停顿 {int(gap_ms)}ms（送 LLM 判定）",
            ))
        else:
            kind = "punct_hard" if any(is_hard_punct(c) for c in punct) else "punct_soft"
            findings.append(Finding(
                cur["end_s"], nxt["start_s"], max(gap, 0.0), kind, punct,
                reason="—",
            ))

    # ---------- S2 char_too_long ----------
    # Suppress char-stretch findings whose anchor (valley or char window)
    # already overlaps an emitted suspicious finding (inter_char or
    # long_pause). Avoids sending the same boundary to LLM twice.
    suspicious_intervals = [(f.start, f.end) for f in findings if f.suspicious]

    def _overlaps_suspicious(s: float, e: float) -> bool:
        return any(min(e, b) - max(s, a) > 0.05 for a, b in suspicious_intervals)

    for j, w in enumerate(chars):
        d = w["end_s"] - w["start_s"]
        z = (d - mu) / sigma if sigma > 1e-6 else 0.0
        if z < CHAR_DUR_Z:
            continue

        best_v = None
        for vs, ve, depth in valleys:
            if (vs >= w["start_s"] - 0.05 and vs <= w["end_s"] + 0.05
                    and (ve - vs) >= 0.15 and depth >= 15):
                if best_v is None or depth > best_v[2]:
                    best_v = (vs, ve, depth)

        if best_v is not None:
            vs, ve, depth = best_v
            if _overlaps_inter(vs, ve) or _overlaps_suspicious(vs, ve):
                continue
            mid_rel = ((vs + ve) / 2 - w["start_s"]) / max(d, 0.001)
            prev_char = chars[j - 1] if j > 0 else None
            next_char = chars[j + 1] if j + 1 < len(chars) else None
            if mid_rel >= 0.5:
                pw_label = w["text"]
                nw_label = next_char["text"] if next_char else "<EOS>"
            else:
                pw_label = prev_char["text"] if prev_char else "<BOS>"
                nw_label = w["text"]
            findings.append(Finding(
                vs, ve, ve - vs, "char_too_long", f"{pw_label}→{nw_label}",
                z=z, depth_db=depth, suspicious=True,
                reason=(f"字 [{w['text']}] 拖长 {int(d*1000)}ms (z={z:.1f})；"
                        f"实际能量谷 {int((ve-vs)*1000)}ms (drop={depth:.0f}dB)"),
            ))
        elif (not _overlaps_inter(w["start_s"], w["end_s"])
              and not _overlaps_suspicious(w["start_s"], w["end_s"])):
            findings.append(Finding(
                w["start_s"], w["end_s"], d, "char_too_long",
                w["text"], z=z, suspicious=True,
                reason=f"字 [{w['text']}] 时长 {int(d*1000)}ms (z={z:.1f}) 异常拖长",
            ))

    # findings_sort comes after S2.

    findings.sort(key=lambda f: f.start)

    # ---- payload ----
    return {
        "audio": os.path.basename(audio_path),
        "duration": float(total_dur),
        "language": res.language,
        "text": text,
        "annotated_text": _annotated_text(text, records, valleys, findings),
        "char_dur_mu_ms": round(mu * 1000, 1),
        "char_dur_sigma_ms": round(sigma * 1000, 1),
        "ref_db": round(float(ref_db), 2),
        "words": [
            {
                "word": w["text"],
                "start": round(w["start_s"], 4),
                "end": round(w["end_s"], 4),
                "score": 0.0,
                "is_punct": False,
            }
            for w in chars
        ],
        "findings": [
            {
                "start": round(f.start, 4),
                "end": round(f.end, 4),
                "duration_ms": round(f.duration * 1000, 1),
                "kind": f.kind,
                "label": f.label,
                "z": round(f.z, 2),
                "depth_db": round(f.depth_db, 2),
                "suspicious": f.suspicious,
                "reason": f.reason,
            }
            for f in findings
        ],
        "rms_db": [round(float(x), 2) for x in rms_db.tolist()],
        "rms_hop_s": round(float(hop_s), 4),
    }


# ---- pretty-printer ---------------------------------------------------------
def _print_payload(payload: dict) -> None:
    print(f"\n===== {payload['audio']}")
    print(f"  language : {payload['language']!r}")
    print(f"  text     : {payload['text']!r}")
    print(f"  annotated: {payload['annotated_text']}")
    print("             (legend: <NNNms>=标点位停顿, <!NNNms!>=句中异常停顿)")
    print(f"  μ={payload['char_dur_mu_ms']}ms σ={payload['char_dur_sigma_ms']}ms"
          f"  ref_db={payload['ref_db']}dB  duration={payload['duration']:.2f}s")

    sus = [f for f in payload["findings"] if f["suspicious"]]
    print(f"  suspicious: {len(sus)} / {len(payload['findings'])}")
    for f in payload["findings"]:
        flag = "★" if f["suspicious"] else " "
        print(f"   {flag} [{f['start']:>6.2f}s-{f['end']:>6.2f}s "
              f"{f['duration_ms']:>6.0f}ms] {f['kind']:>14}  "
              f"{f['label']:<12}  z={f['z']:>5.2f} drop={f['depth_db']:>5.1f}dB"
              f"  {f['reason']}")


# ---- CLI --------------------------------------------------------------------
def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio", nargs="*",
                   help="audio path(s) or globs (default: ./audios/*.wav)")
    p.add_argument("--language", default=None,
                   help="forced language (default: auto-detect)")
    p.add_argument("--json", default=None,
                   help="write payload to this JSON path (omit to print only)")


@tool("detect_pauses",
      "Detect TTS-style abnormal pauses (5 signals: inter_char / char_too_long / "
      "in_word / long_pause / leading-trailing).",
      _add_args)
def detect_pauses_cmd(args: argparse.Namespace) -> int:
    if not args.audio:
        paths = sorted(glob.glob("./audios/*.wav"))
    else:
        paths = []
        for p in args.audio:
            if any(ch in p for ch in "*?["):
                paths.extend(sorted(glob.glob(p)))
            else:
                paths.append(p)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("no audio files", file=sys.stderr)
        return 1

    payloads = []
    for p in paths:
        payload = analyze_pauses(p, language=args.language)
        payloads.append(payload)
        _print_payload(payload)

    if args.json:
        if len(payloads) == 1:
            with open(args.json, "w", encoding="utf-8") as fp:
                json.dump(payloads[0], fp, ensure_ascii=False, indent=2)
        else:
            with open(args.json, "w", encoding="utf-8") as fp:
                json.dump(payloads, fp, ensure_ascii=False, indent=2)
        print(f"\n[json] {args.json}")
    return 0
