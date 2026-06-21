"""Lazy singletons for Qwen3-ASR and Qwen3-ForcedAligner.

Tools call `get_asr()` / `get_aligner()` and pay the load cost once per process.
"""

import os

import torch
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

DEFAULT_ASR_PATH = os.environ.get("QWEN3_ASR_PATH", "./models/Qwen3-ASR-0.6B")
DEFAULT_FA_PATH = os.environ.get("QWEN3_FA_PATH", "./models/Qwen3-ForcedAligner-0.6B")
DEFAULT_DEVICE = os.environ.get("QWEN3_DEVICE", "cuda:0")

_asr: Qwen3ASRModel | None = None
_aligner: Qwen3ForcedAligner | None = None


def get_asr(asr_path: str = DEFAULT_ASR_PATH, fa_path: str = DEFAULT_FA_PATH,
            device: str = DEFAULT_DEVICE,
            with_aligner: bool = True) -> Qwen3ASRModel:
    """Return the Qwen3ASRModel singleton (loads on first call)."""
    global _asr
    if _asr is None:
        kwargs = dict(
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=8,
            max_new_tokens=256,
        )
        if with_aligner:
            kwargs["forced_aligner"] = fa_path
            kwargs["forced_aligner_kwargs"] = dict(dtype=torch.bfloat16, device_map=device)
        _asr = Qwen3ASRModel.from_pretrained(asr_path, **kwargs)
    return _asr


def get_aligner(fa_path: str = DEFAULT_FA_PATH, device: str = DEFAULT_DEVICE) -> Qwen3ForcedAligner:
    """Return the Qwen3ForcedAligner singleton (loads on first call)."""
    global _aligner
    if _aligner is None:
        _aligner = Qwen3ForcedAligner.from_pretrained(
            fa_path, dtype=torch.bfloat16, device_map=device,
        )
    return _aligner
