# tts-pause-audit

> Detect **unnatural pauses** in TTS audio, locate them precisely with **acoustic energy valleys**, then let an **LLM judge** which ones a listener would actually find weird.

Built on top of [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), with the LLM judgment going through any OpenAI-compatible endpoint (DashScope-compatible mode by default).

## What it catches

Five complementary signals, each addressing a different way TTS pauses can go wrong:

| Signal | "我抓的是" | 严重度 |
|---|---|---|
| `S1 inter_char` | 不该断的地方断了 | ★★★（最常见的 TTS 毛病） |
| `S2 char_too_long` | 字被拖长，藏了静音 | ★★（反向定位的关键） |
| `S3 in_word` | 一个字被读破 | ★★★（极少见但绝对刺耳） |
| `S4 long_pause` | 标点处停太久 | ★（多数是自然的，需 LLM 判） |
| `S5 leading/trailing` | 头尾的静音 | —（仅记录，非异常） |

详细原理见 [`docs/signals.md`](docs/signals.md)。

## Example report

Below is a snippet from [`examples/example_report.md`](examples/example_report.md) (rendered from one of the bundled samples). Issues jump out visually.

> ⏮320ms 刚收到快递，发现是男朋友 **❌350ms** 偷偷寄来的 **❌300ms** 生日蛋糕。 ✓320ms 虽然隔着屏幕， ✓320ms 但这份心 ✓320ms **❌210ms** 意让我笑得合不拢嘴。 ⏭490ms

```
时间轴: __·········||·········XXX······XXX·····===··········==·····===XX···········_____
       |                  |                   |                   |                   |
       0s               2.2s                4.5s                6.7s                9.0s
图例: X=不自然 · ==自然 · |=标点 · _=首尾静音 · ·=正常发声
```

🚩 **需要关注**：
- ❌ **友 → 偷** @ 2.50s · `inter_char` · 350ms · *切开固定词"男朋友"*
- ❌ **的 → 生** @ 3.55s · `inter_char` · 300ms · *切开'寄来的'与'生日蛋糕'*
- ❌ **心 → 意** @ 7.04s · `inter_char` · 210ms · *切开固定词"心意"*

## Quickstart

### 1. Install

We recommend a fresh Python 3.12 environment.

```bash
conda create -n tts-pause-audit python=3.12 -y
conda activate tts-pause-audit
pip install tts-pause-audit
# or, from source:
git clone https://github.com/notedit/tts-pause-audit.git
cd tts-pause-audit
pip install -e .
```

CUDA tip: install a `torch` build that matches your driver (e.g. `pip install torch==2.5.1+cu121 -i https://download.pytorch.org/whl/cu121`). If you see `Error 803: system has unsupported display driver / cuda driver combination`, try `unset LD_LIBRARY_PATH` before running.

### 2. Get the models

```bash
# Either ModelScope (recommended in Mainland China)
modelscope download --model Qwen/Qwen3-ASR-0.6B            --local_dir ./models/Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B  --local_dir ./models/Qwen3-ForcedAligner-0.6B

# Or Hugging Face
huggingface-cli download Qwen/Qwen3-ASR-0.6B           --local-dir ./models/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/Qwen3-ForcedAligner-0.6B
```

The tool reads paths from these env vars (defaults shown):

```bash
export QWEN3_ASR_PATH=./models/Qwen3-ASR-0.6B
export QWEN3_FA_PATH=./models/Qwen3-ForcedAligner-0.6B
export QWEN3_DEVICE=cuda:0
```

### 3. Configure the LLM

Any OpenAI-compatible endpoint works. Defaults point to DashScope's OpenAI-compatible mode:

```bash
export OPENAI_API_KEY=sk-...                            # DASHSCOPE_API_KEY also accepted
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export PAUSE_LLM_MODEL=qwen-plus                        # or qwen3-max for tighter judgments
```

### 4. Run

```bash
# detect + LLM judge in one shot
tts-pause-audit run_pauses examples/audios/*.wav --json out.json --model qwen3-max

# or two steps (cheap detect first, judge later)
tts-pause-audit detect_pauses examples/audios/*.wav --json out.json
tts-pause-audit judge_pauses out.json --model qwen3-max

# render a markdown report (with per-file LLM summaries)
tts-pause-audit report_pauses out.json --md report.md --summary --model qwen3-max
```

`tts-pause-audit --list` prints every registered subcommand.

## CLI reference

| Command | Purpose |
|---|---|
| `transcribe` | Plain Qwen3-ASR transcription, with optional char-level timestamps. |
| `detect_pauses` | Pure-acoustic 5-signal detection. No LLM call, no API key needed. |
| `judge_pauses` | Send each `suspicious=true` finding in a detect-JSON to an OpenAI-compatible LLM; writes back `llm_natural` / `llm_reason`. |
| `run_pauses` | `detect_pauses` then `judge_pauses` in one go. Use `--no-llm` to skip the LLM. |
| `report_pauses` | Render a JSON payload as a markdown report (annotated transcript + ASCII timeline + focused issues + full table). With `--summary` it also calls the LLM for a 1-2 sentence per-file verdict. |

LLM-related flags everywhere: `--api-key`, `--base-url`, `--model`. CLI > env > default.

## How it works

```
audio.wav
  ├─ Qwen3-ASR + Qwen3-FA  →  text + char-level timestamps
  ├─ RMS-dB envelope (p90 baseline)  →  energy valleys
  └─ 5-signal joint analysis  →  findings[]
        ↓
   build JSON payload  (compat with pause_detect.py findings_to_payload)
        ↓
   OpenAI-compatible LLM judge  →  llm_natural / llm_reason per finding
        ↓
   markdown report (annotated text + timeline + table + per-file summary)
```

## JSON output shape

```json
{
  "audio": "x.wav",
  "duration": 8.97,
  "language": "Chinese",
  "text": "...",
  "annotated_text": "...<!350ms!>...",
  "char_dur_mu_ms": 178.5,
  "char_dur_sigma_ms": 64.0,
  "ref_db": -10.65,
  "words":    [{"word": "刚", "start": 0.32, "end": 0.48, "score": 0.0, "is_punct": false}, ...],
  "findings": [
    {
      "start": 2.50, "end": 2.85, "duration_ms": 350,
      "kind": "inter_char", "label": "友→偷",
      "z": -0.29, "depth_db": 37.3, "suspicious": true,
      "reason": "字间能量谷 349ms (drop=37dB)",
      "llm_natural": false, "llm_reason": "切开固定词\"男朋友\"",
      "llm_prev": "友", "llm_next": "偷"
    },
    ...
  ],
  "rms_db": [...],
  "rms_hop_s": 0.01
}
```

## Extending — adding a new capability

Capabilities are auto-registered. Drop a new file in `pause_detector/tools/`:

```python
# pause_detector/tools/my_tool.py
import argparse
from ..registry import tool

def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio")

@tool("my_tool", "What it does in one sentence.", _add_args)
def my_tool_cmd(args: argparse.Namespace) -> int:
    print("hello", args.audio)
    return 0
```

Then add `from . import my_tool` in `pause_detector/tools/__init__.py`. `tts-pause-audit --list` will pick it up automatically.

## Sample data

`examples/audios/` ships four ~8s Chinese TTS clips. They map to the calibrated samples A–D embedded in the LLM system prompt (`pause_detector/prompts.py`):

| File | Calibrated sample | Expected issues |
|---|---|---|
| `audio_v-female-T3P8sZ0Q_0006(6).wav` | A | "男朋友" / "寄来的生日" 被切 |
| `audio_v-female-T3P8sZ0Q_0021(5).wav` | B | "违反规则在先" 被切 |
| `audio_v-female-T3P8sZ0Q_0022(5).wav` | C | 全句流畅 |
| `audio_v-female-T3P8sZ0Q_0025(5).wav` | D | "保修" 接代词刺耳 |

A pre-rendered markdown report is at [`examples/example_report.md`](examples/example_report.md).

## Acknowledgements

This project depends on, but does not modify, the upstream open-source releases from the Alibaba Qwen team:

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Apache-2.0)
- [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

The pause-detection algorithm and LLM-judgment system prompt were ported from an internal DashScope-based reference (`docs/legacy/pause_detect_dashscope.py`).

## License

[MIT](LICENSE).
