# tts-pause-audit

[English](README.md) · 中文

> 用 **声学能量谷**精确定位 TTS 音频里的不自然停顿，再让 **LLM 判定**哪些停顿真的会让听众觉得别扭。

基于 [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) + [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)，LLM 判定走任意 OpenAI 兼容端点（默认指向 DashScope 兼容模式）。

## 它能抓到什么

5 类互补的信号，每一类对应一种 TTS 停顿可能"出错"的方式：

| 信号 | 我抓的是 | 严重度 |
|---|---|---|
| `S1 inter_char` | 不该断的地方断了 | ★★★（最常见的 TTS 毛病） |
| `S2 char_too_long` | 字被拖长，藏了静音 | ★★（反向定位的关键） |
| `S3 in_word` | 一个字被读破 | ★★★（极少见但绝对刺耳） |
| `S4 long_pause` | 标点处停太久 | ★（多数是自然的，需 LLM 判） |
| `S5 leading/trailing` | 头尾的静音 | —（仅记录，非异常） |

详细原理见 [`docs/signals.md`](docs/signals.md)。

## 示例报告

下面这段截选自 [`examples/example_report.md`](examples/example_report.md)（从内置样本之一渲染得到），异常位置一眼可见。

> ⏮320ms 刚收到快递，发现是男朋友 **❌350ms** 偷偷寄来的 **❌300ms** 生日蛋糕。 ✓320ms 虽然隔着屏幕， ✓320ms 但这份心 ✓320ms **❌210ms** 意让我笑得合不拢嘴。 ⏭490ms

```
时间轴: __·········||·········XXX······XXX·····===··········==·····===XX···········_____
       |                  |                   |                   |                   |
       0s               2.2s                4.5s                6.7s                9.0s
图例: X=不自然 · ==自然 · |=标点 · _=首尾静音 · ·=正常发声
```

🚩 **需要关注**：
- ❌ **友 → 偷** @ 2.50s · `inter_char` · 350ms · *切开固定词"男朋友"*
- ❌ **的 → 生** @ 3.55s · `inter_char` · 300ms · *切开"寄来的"与"生日蛋糕"*
- ❌ **心 → 意** @ 7.04s · `inter_char` · 210ms · *切开固定词"心意"*

## 快速开始

### 1. 安装

建议用全新的 Python 3.12 环境。

```bash
conda create -n tts-pause-audit python=3.12 -y
conda activate tts-pause-audit
pip install tts-pause-audit
# 或者从源码安装：
git clone https://github.com/notedit/tts-pause-audit.git
cd tts-pause-audit
pip install -e .
```

CUDA 提示：装匹配你显卡驱动的 `torch`（比如 `pip install torch==2.5.1+cu121 -i https://download.pytorch.org/whl/cu121`）。如果遇到 `Error 803: system has unsupported display driver / cuda driver combination`，先 `unset LD_LIBRARY_PATH` 再跑。

### 2. 下载模型

```bash
# ModelScope（推荐国内用户）
modelscope download --model Qwen/Qwen3-ASR-0.6B            --local_dir ./models/Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B  --local_dir ./models/Qwen3-ForcedAligner-0.6B

# 或者 Hugging Face
huggingface-cli download Qwen/Qwen3-ASR-0.6B           --local-dir ./models/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/Qwen3-ForcedAligner-0.6B
```

工具会读以下环境变量定位模型（括号为默认值）：

```bash
export QWEN3_ASR_PATH=./models/Qwen3-ASR-0.6B
export QWEN3_FA_PATH=./models/Qwen3-ForcedAligner-0.6B
export QWEN3_DEVICE=cuda:0
```

### 3. 配置 LLM

任何 OpenAI 兼容端点都能用。默认指向 DashScope 的 OpenAI 兼容模式：

```bash
export OPENAI_API_KEY=sk-...                            # 也接受 DASHSCOPE_API_KEY
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export PAUSE_LLM_MODEL=qwen-plus                        # 想要更严格的判定可用 qwen3-max
```

### 4. 跑

```bash
# 检测 + LLM 判定一气呵成
tts-pause-audit run_pauses examples/audios/*.wav --json out.json --model qwen3-max

# 或拆两步（先便宜地检测，再单独判定）
tts-pause-audit detect_pauses examples/audios/*.wav --json out.json
tts-pause-audit judge_pauses out.json --model qwen3-max

# 渲染 markdown 报告（含整句 LLM 总结）
tts-pause-audit report_pauses out.json --md report.md --summary --model qwen3-max
```

`tts-pause-audit --list` 列出所有已注册的子命令。

## CLI 命令一览

| 命令 | 用途 |
|---|---|
| `transcribe` | 纯 Qwen3-ASR 转录，可选返回字符级时间戳 |
| `detect_pauses` | 纯声学的 5 信号检测，**不需要** API key |
| `judge_pauses` | 把检测 JSON 中所有 `suspicious=true` 的项送到 OpenAI 兼容 LLM；写回 `llm_natural` / `llm_reason` |
| `run_pauses` | 一步完成 `detect_pauses` + `judge_pauses`；可加 `--no-llm` 跳过判定 |
| `report_pauses` | 把 JSON 渲染成 markdown 报告（标注转录 + ASCII 时间轴 + 重点列表 + 全表）。加 `--summary` 会让 LLM 给每个文件写一两句整句总结 |

LLM 相关参数全部通用：`--api-key`、`--base-url`、`--model`。优先级：CLI > 环境变量 > 默认值。

## 工作原理

```
audio.wav
  ├─ Qwen3-ASR + Qwen3-FA  →  文本 + 字级时间戳
  ├─ RMS-dB 包络（p90 基线） →  能量谷
  └─ 5 信号联合分析  →  findings[]
        ↓
   JSON 输出  （字段与原 pause_detect.py findings_to_payload 对齐）
        ↓
   OpenAI 兼容 LLM 判定  →  每个 finding 加 llm_natural / llm_reason
        ↓
   markdown 报告（标注文本 + 时间轴 + 表格 + 整句总结）
```

## JSON 输出契约

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

## 添加能力（扩展）

能力是自动注册的。在 `pause_detector/tools/` 下加一个新文件：

```python
# pause_detector/tools/my_tool.py
import argparse
from ..registry import tool

def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("audio")

@tool("my_tool", "一句话描述这个工具能干嘛。", _add_args)
def my_tool_cmd(args: argparse.Namespace) -> int:
    print("hello", args.audio)
    return 0
```

然后在 `pause_detector/tools/__init__.py` 里加一行 `from . import my_tool`。`tts-pause-audit --list` 就能看到新命令。

## 样例数据

`examples/audios/` 自带 4 条约 8 秒的中文 TTS 片段，对应 LLM system prompt（`pause_detector/prompts.py`）里 5 条已校准样例 A–D：

| 文件 | 校准样例 | 预期问题 |
|---|---|---|
| `audio_v-female-T3P8sZ0Q_0006(6).wav` | A | "男朋友" / "寄来的生日" 被切 |
| `audio_v-female-T3P8sZ0Q_0021(5).wav` | B | "违反规则在先" 被切 |
| `audio_v-female-T3P8sZ0Q_0022(5).wav` | C | 全句流畅 |
| `audio_v-female-T3P8sZ0Q_0025(5).wav` | D | "保修" 接代词刺耳 |

预渲染好的 markdown 报告：[`examples/example_report.md`](examples/example_report.md)。

## 致谢

本项目依赖（但**不修改**）阿里 Qwen 团队的开源发布：

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)（Apache-2.0）
- [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

停顿检测算法和 LLM 判定 system prompt 移植自内部一份基于 DashScope 的参考实现（`docs/legacy/pause_detect_dashscope.py`）。

## 许可证

[MIT](LICENSE)。
