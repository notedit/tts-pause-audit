"""RMS dB envelope + energy-valley detector.

A "valley" is a contiguous run of frames whose RMS-dB is below an adaptive
threshold (`speech_p70 - margin_db`, with floor at `noise_p5 + 6 dB`),
that lasts at least `min_ms`. Returned as `(start_s, end_s, mean_db)`.

Used by the pause detector for *reverse verification*: even when the
forced aligner reports gap=0 between two characters, a valley sitting
between their timestamps signals the silence was swallowed inside the
character window.
"""

from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class Valley:
    start_s: float
    end_s: float
    mean_db: float

    @property
    def duration_ms(self) -> float:
        return (self.end_s - self.start_s) * 1000.0


def load_mono(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def rms_db_envelope(wav: np.ndarray, sr: int, frame_ms: float = 20.0,
                    hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_centers_seconds, frame_db) using overlapping frames."""
    n = max(1, int(sr * frame_ms / 1000))
    h = max(1, int(sr * hop_ms / 1000))
    if len(wav) < n:
        return np.array([0.0]), np.array([20.0 * np.log10(np.sqrt((wav ** 2).mean() + 1e-12) + 1e-12)])
    n_frames = 1 + (len(wav) - n) // h
    idx = np.arange(n_frames) * h
    frames = np.lib.stride_tricks.as_strided(
        wav, shape=(n_frames, n),
        strides=(wav.strides[0] * h, wav.strides[0]),
        writeable=False,
    )
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12).astype(np.float32)
    centers = (idx + n / 2) / sr
    return centers, db


def detect_valleys(centers: np.ndarray, db: np.ndarray,
                   min_ms: float = 200.0,
                   margin_db: float = 25.0,
                   merge_gap_ms: float = 60.0) -> list[Valley]:
    """Adaptive-threshold valley detection.

    threshold = max(speech_p70 - margin_db, noise_p5 + 6dB)
    """
    if len(db) == 0:
        return []
    speech_db = float(np.percentile(db, 70))
    noise_db = float(np.percentile(db, 5))
    thresh = max(speech_db - margin_db, noise_db + 6.0)

    below = db < thresh
    valleys: list[Valley] = []
    i = 0
    while i < len(below):
        if not below[i]:
            i += 1
            continue
        j = i
        while j < len(below) and below[j]:
            j += 1
        # frames [i, j)
        s, e = float(centers[i]), float(centers[j - 1])
        valleys.append(Valley(start_s=s, end_s=e, mean_db=float(db[i:j].mean())))
        i = j

    # merge valleys separated by tiny voiced blips
    merged: list[Valley] = []
    for v in valleys:
        if merged and (v.start_s - merged[-1].end_s) * 1000.0 <= merge_gap_ms:
            prev = merged[-1]
            new_db = (prev.mean_db * (prev.end_s - prev.start_s)
                      + v.mean_db * (v.end_s - v.start_s)) / max(
                v.end_s - prev.start_s, 1e-9)
            merged[-1] = Valley(start_s=prev.start_s, end_s=v.end_s, mean_db=new_db)
        else:
            merged.append(v)

    # filter by min duration
    return [v for v in merged if v.duration_ms >= min_ms]


def stats(db: np.ndarray) -> dict:
    if len(db) == 0:
        return {"speech_p70": float("nan"), "noise_p5": float("nan")}
    return {
        "speech_p70": float(np.percentile(db, 70)),
        "noise_p5": float(np.percentile(db, 5)),
        "median": float(np.median(db)),
    }


# ---------------------------------------------------------------------------
# Pause-detector flavour: ref_db = p90, no merge, returns depth_db.
# Mirrors pause_detect.py::compute_rms_db / find_valleys so the joint analysis
# matches the calibrated thresholds (VALLEY_DROP_DB=10, VALLEY_MIN_MS=80, ...).
# ---------------------------------------------------------------------------


def compute_rms_db_p90(wav_path: str, win_ms: int = 30, hop_ms: int = 10
                       ) -> tuple[np.ndarray, float, float, float]:
    """Compute RMS-dB envelope with the p90 reference baseline.

    Returns (rms_db, hop_s, win_s, ref_db) where ref_db = 90th percentile.
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    n = (len(audio) - win) // hop + 1
    if n <= 0:
        return np.array([], dtype=np.float32), hop / sr, win_ms / 1000.0, -60.0
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n, win),
        strides=(hop * audio.strides[0], audio.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = (20.0 * np.log10(rms + 1e-12)).astype(np.float32)
    ref_db = float(np.percentile(rms_db, 90))
    return rms_db, hop / sr, win_ms / 1000.0, ref_db


def find_valleys_p90(rms_db: np.ndarray, hop_s: float, win_s: float,
                     ref_db: float, drop_db: float = 10.0,
                     min_ms: float = 80.0
                     ) -> list[tuple[float, float, float]]:
    """Pause-detector valley finder.

    A valley is a contiguous run of frames with rms_db < (ref_db - drop_db),
    lasting at least `min_ms`. Returns (start_s, end_s, depth_db) tuples
    where depth_db = ref_db - min(rms_db[i:j]).
    """
    if len(rms_db) == 0:
        return []
    is_v = rms_db < (ref_db - drop_db)
    out: list[tuple[float, float, float]] = []
    i = 0
    n = len(is_v)
    while i < n:
        if not is_v[i]:
            i += 1
            continue
        j = i
        while j < n and is_v[j]:
            j += 1
        s_t = i * hop_s
        e_t = j * hop_s + win_s
        if (e_t - s_t) * 1000.0 >= min_ms:
            depth = ref_db - float(rms_db[i:j].min())
            out.append((s_t, e_t, depth))
        i = j
    return out

