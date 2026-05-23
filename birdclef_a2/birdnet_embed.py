from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef_a2.audio_io import iter_segments, load_audio_mono
from birdclef_a2.manifest_utils import assert_manifest_columns, resolve_audio_path

logger = logging.getLogger(__name__)


def birdnet_sample_rate() -> int:
    """BirdNET acoustic TF model sample rate (downloads weights on first call)."""
    try:
        import birdnet
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Missing the `birdnet` package (distinct from birdnet-analyzer). "
            "Install: pip install birdnet-analyzer birdnet"
        ) from e

    model = birdnet.load("acoustic", "2.4", "tf")
    return int(model.get_sample_rate())


def compute_file_embedding(
    audio_path: Path,
    *,
    sample_rate: int,
    segment_s: float,
    overlap_s: float,
    embed_batch_segments: int,
) -> np.ndarray | None:
    from birdnet_analyzer.model_utils import get_embeddings_array

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

    embs: list[np.ndarray] = []
    for i in range(0, len(segments), embed_batch_segments):
        chunk = segments[i : i + embed_batch_segments]
        arr = get_embeddings_array(chunk, batch_size=min(len(chunk), embed_batch_segments))
        embs.append(arr)
    stacked = np.concatenate(embs, axis=0)
    return stacked.mean(axis=0)


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
