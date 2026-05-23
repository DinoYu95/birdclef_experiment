from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def load_audio_mono(path: str | Path, *, sample_rate: int) -> np.ndarray:
    """Load audio as mono float32 waveform in [-1, 1]."""
    path = Path(path)
    try:
        import librosa
    except ImportError as e:  # pragma: no cover
        raise ImportError("audio_io requires librosa (pip install librosa soundfile)") from e

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path, sr=None, mono=True)
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    return y.astype(np.float32, copy=False)


def iter_segments(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    segment_s: float,
    overlap_s: float,
) -> list[np.ndarray]:
    """Fixed-length segments with overlap; pad tail if shorter than window."""
    seg = max(int(round(segment_s * sample_rate)), 1)
    hop = max(int(round((segment_s - overlap_s) * sample_rate)), 1)
    if len(waveform) == 0:
        return []

    segments: list[np.ndarray] = []
    start = 0
    while start + seg <= len(waveform):
        segments.append(waveform[start : start + seg].astype(np.float32, copy=False))
        start += hop

    if start < len(waveform) or not segments:
        tail = waveform[start:] if start < len(waveform) else waveform[-seg:]
        tail = tail.astype(np.float32, copy=False)
        if len(tail) < seg:
            pad = np.zeros(seg - len(tail), dtype=np.float32)
            tail = np.concatenate([tail, pad])
        segments.append(tail[:seg])

    return segments
