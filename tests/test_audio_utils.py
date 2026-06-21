import numpy as np

from pause_detector.audio_utils import find_valleys_p90


def test_find_valleys_p90_detects_silence():
    # Construct a synthetic envelope: 0.5s loud, 0.5s silent, 0.5s loud at 10ms hop.
    hop_s = 0.01
    win_s = 0.03
    loud_db = np.full(50, -10.0, dtype=np.float32)
    silent_db = np.full(50, -60.0, dtype=np.float32)
    rms_db = np.concatenate([loud_db, silent_db, loud_db])
    ref_db = float(np.percentile(rms_db, 90))

    valleys = find_valleys_p90(rms_db, hop_s, win_s, ref_db,
                               drop_db=10.0, min_ms=80.0)
    assert len(valleys) == 1
    s, e, depth = valleys[0]
    # silent block spans frames [50, 100) -> [0.5, 1.03)s with the win tail
    assert 0.45 < s < 0.55
    assert 0.95 < e < 1.10
    assert depth > 40  # ref_db ~ -10, min ~ -60 -> depth ~ 50


def test_find_valleys_p90_respects_min_ms():
    # 30ms blip should be filtered out by min_ms=80.
    hop_s = 0.01
    win_s = 0.03
    rms_db = np.full(100, -10.0, dtype=np.float32)
    rms_db[50:53] = -60.0
    ref_db = float(np.percentile(rms_db, 90))
    assert find_valleys_p90(rms_db, hop_s, win_s, ref_db,
                            drop_db=10.0, min_ms=80.0) == []


def test_find_valleys_p90_empty_input():
    assert find_valleys_p90(np.array([], dtype=np.float32), 0.01, 0.03, -20.0) == []
