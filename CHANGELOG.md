# Changelog

All notable changes to this project are documented here.

## [0.1.0] — 2026-06-21

Initial public release.

### Added
- Full pipeline: Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B → 5-signal pause detection (S1–S5) → OpenAI-compatible LLM semantic judgment → annotated markdown report.
- Five detector signals:
  - `S1 inter_char` — silent gap between adjacent characters (no punctuation).
  - `S2 char_too_long` — z-score-anomalous character duration; silence swallowed into the char window.
  - `S3 in_word` — energy valley fully inside a single character (broken word).
  - `S4 long_pause` — overlong pause at a punctuation boundary.
  - `S5 leading / trailing` — silence at the head/tail of the clip.
- CLI commands (auto-registered via `@tool`):
  - `tts-pause-audit transcribe` — Qwen3-ASR transcription with optional timestamps.
  - `tts-pause-audit detect_pauses` — pure-acoustic 5-signal detection.
  - `tts-pause-audit judge_pauses` — LLM judgment over a detect-output JSON.
  - `tts-pause-audit run_pauses` — composite detect + judge.
  - `tts-pause-audit report_pauses` — render a markdown report with annotated transcript, ASCII timeline, focused issues list, optional per-file LLM summary, and a full findings table.
- OpenAI-compatible LLM client with priority CLI > env > defaults. Defaults to DashScope OpenAI-compatible mode + `qwen-plus`.
- Calibrated Chinese system prompt covering both "natural" and "unnatural" pause heuristics, with five real samples (A–E).
- Four sample Chinese audios under `examples/audios/` and a pre-rendered report at `examples/example_report.md`.

### Notes
- The DashScope-based reference implementation that the open-source pipeline was ported from is preserved at `docs/legacy/pause_detect_dashscope.py` for comparison.
