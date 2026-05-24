from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from birdclef_a2.audio_io import iter_segments, load_audio_mono
from birdclef_a2.manifest_utils import assert_manifest_columns, resolve_audio_path

logger = logging.getLogger(__name__)

# TensorFlow acoustic (ProtoBuf) BirdNET 2.4 — lazily loaded once.
_ACOUSTIC_MODEL: Any | None = None


def birdnet_inference_device() -> str:
    """e.g. `CPU` or `GPU:0` — set `BIRDCLEF_BIRDNET_DEVICE` to override."""
    return os.environ.get("BIRDCLEF_BIRDNET_DEVICE", "CPU")


def acoustic_birdnet_model() -> Any:
    """Shared BirdNET acoustic 2.4 TF weights (used for embeddings and optional synth verify)."""
    return _acoustic_tf_model()


def _acoustic_tf_model() -> Any:
    global _ACOUSTIC_MODEL
    if _ACOUSTIC_MODEL is None:
        try:
            import birdnet
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "BirdNET embeddings require the PyPI `birdnet` package "
                "(Cornell birdnet-team): pip install 'birdnet>=0.2.5'"
            ) from e
        _ACOUSTIC_MODEL = birdnet.load("acoustic", "2.4", "tf")
    return _ACOUSTIC_MODEL


def birdnet_sample_rate() -> int:
    """Sample rate expected by BirdNET acoustic TF 2.4 (model download on first load)."""
    m = acoustic_birdnet_model()
    return int(m.get_sample_rate())


def _pool_segments_from_encoding_result(res: Any, input_idx: int) -> np.ndarray | None:
    """One row = one embedding vector (mean over valid internal 3 s windows for this input)."""
    emb = np.asarray(res.embeddings[input_idx])
    mask = np.asarray(res.embeddings_masked[input_idx])
    if emb.ndim != 2 or emb.size == 0:
        return None
    if mask.ndim >= 2:
        ok = ~mask.all(axis=-1)
    else:
        ok = ~mask.astype(bool, copy=False)
    rows = emb[ok]
    if rows.size == 0:
        return None
    return np.mean(rows.astype(np.float64), axis=0).astype(np.float32)


def compute_file_embedding(
    audio_path: Path,
    *,
    sample_rate: int,
    segment_s: float,
    overlap_s: float,
    embed_batch_segments: int,
) -> np.ndarray | None:
    model = acoustic_birdnet_model()

    try:
        wav = load_audio_mono(audio_path, sample_rate=sample_rate)
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to load %s: %s", audio_path, exc)
        return None

    segments = iter_segments(
        wav, sample_rate=sample_rate, segment_s=segment_s, overlap_s=overlap_s
    )
    if not segments:
        return None

    pooled: list[np.ndarray] = []
    bsz = embed_batch_segments
    for i in range(0, len(segments), bsz):
        batch = segments[i : i + bsz]
        inputs = [(np.asarray(seg, dtype=np.float32), sample_rate) for seg in batch]
        res = model.encode_arrays(
            inputs,
            batch_size=min(len(inputs), bsz),
            overlap_duration_s=0.0,
            half_precision=False,
            show_stats=None,
            device=birdnet_inference_device(),
        )
        for j in range(len(inputs)):
            vec = _pool_segments_from_encoding_result(res, j)
            if vec is not None:
                pooled.append(vec)

    if not pooled:
        return None
    return np.mean(np.stack(pooled, axis=0), axis=0).astype(np.float32)


def manifest_to_embeddings_npz(
    manifest_csv: Path,
    data_root: Path,
    out_npz: Path,
    *,
    label_col: str = "primary_label",
    segment_s: float = 3.0,
    overlap_s: float = 1.5,
    embed_batch_segments: int = 16,
    limit_rows: int | None = None,
    audio_relative_to: str = "train_audio",
) -> None:
    df = pd.read_csv(manifest_csv)
    assert_manifest_columns(df)
    if label_col not in df.columns:
        raise KeyError(f"missing {label_col}")

    logger.info("BirdNET device: %s", birdnet_inference_device())

    sr = birdnet_sample_rate()
    xs: list[np.ndarray] = []
    ys: list[str] = []
    paths_ok: list[str] = []

    n = len(df) if limit_rows is None else min(len(df), limit_rows)
    for i in range(n):
        row = df.iloc[i]
        rel = str(row["filename"])
        ap = resolve_audio_path(data_root, rel, relative_to=audio_relative_to)
        if not ap.is_file():
            logger.warning("missing audio (skipped): %s", ap)
            continue
        emb = compute_file_embedding(
            ap,
            sample_rate=sr,
            segment_s=segment_s,
            overlap_s=overlap_s,
            embed_batch_segments=embed_batch_segments,
        )
        if emb is None:
            continue
        xs.append(emb)
        ys.append(str(row[label_col]))
        paths_ok.append(rel)
        if (i + 1) % 50 == 0:
            logger.info("embedded %s / %s rows", i + 1, n)

    if not xs:
        raise RuntimeError("no embeddings produced (check paths and audio files)")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=np.stack(xs, axis=0).astype(np.float32),
        y=np.asarray(ys),
        filename=np.asarray(paths_ok),
        birdnet_sr=np.int32(sr),
        segment_s=np.float32(segment_s),
        overlap_s=np.float32(overlap_s),
    )
