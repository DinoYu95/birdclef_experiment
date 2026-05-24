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

# Lazy cache: PB (SavedModel) and TF (Lite path) bundles are different binaries.
_ACOUSTIC_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _maybe_apply_tf_thread_caps() -> None:
    """Keep TF Eigen pools small so cgroup-low ``ulimit`` / ``pids.max`` hosts don't explode.

    Set ``BIRDCLEF_TF_NO_THREAD_CAPS=1`` or override ``TF_*`` yourself to disable defaults.
    """
    if _env_truthy("BIRDCLEF_TF_NO_THREAD_CAPS"):
        return
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRA_OP_THREADS", "2")


def _maybe_pin_single_cuda_device_for_pb() -> None:
    """BirdNET ``set_gpu_device_tf`` asserts exactly one TF-visible GPU."""

    bk, _ = birdnet_backend_and_session_device()
    if bk != "pb":
        return
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None and str(existing).strip() != "":
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def birdnet_inference_device() -> str:
    """Raw selection from ``BIRDCLEF_BIRDNET_DEVICE`` or CLI (e.g. ``CPU``, ``GPU:0``)."""
    return os.environ.get("BIRDCLEF_BIRDNET_DEVICE", "CPU").strip()


def birdnet_backend_and_session_device() -> tuple[str, str]:
    """Return ``(backend, session_device)`` for ``birdnet.load``.

    ``backend=="tf"`` uses the Lite-style stack in `birdnet` and only supports **CPU** inference.

    Strings like ``GPU:0`` (or bare ``gpu`` → ``GPU:0``) use ``backend=="pb"`` (TensorFlow
    SavedModel) so NVIDIA GPUs work when CUDA-enabled TensorFlow is installed.

    Set ``BIRDCLEF_BIRDNET_FORCE_TF_CPU=1`` (or CLI ``--birdnet-force-tf-cpu``) to always use
    the slower ``tf`` CPU path — e.g. when GPU fails with cuDNN errors such as **No DNN support for stream**.
    """
    if _env_truthy("BIRDCLEF_BIRDNET_FORCE_TF_CPU"):
        raw = birdnet_inference_device()
        if raw.lower().startswith(("gpu", "cuda")):
            logger.warning(
                "BIRDCLEF_BIRDNET_FORCE_TF_CPU: ignoring GPU device %r; using backend=tf on CPU.",
                raw,
            )
        return ("tf", "CPU")
    raw = birdnet_inference_device()
    lowered = raw.lower()
    if lowered.startswith(("gpu", "cuda")):
        session = raw if lowered != "gpu" else "GPU:0"
        return ("pb", session)
    return ("tf", "CPU")


def birdnet_session_device() -> str:
    """Device passed to ``encode_arrays`` / ``predict`` — always ``CPU`` when backend is ``tf``."""
    return birdnet_backend_and_session_device()[1]


def _get_acoustic_model() -> Any:
    bk, session_key = birdnet_backend_and_session_device()
    cache_key = (bk, session_key)
    hit = _ACOUSTIC_MODEL_CACHE.get(cache_key)
    if hit is not None:
        return hit
    _maybe_apply_tf_thread_caps()
    _maybe_pin_single_cuda_device_for_pb()
    try:
        import birdnet
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "BirdNET requires PyPI package `birdnet` (Cornell): pip install 'birdnet>=0.2.5'"
        ) from e
    if bk == "tf":
        model = birdnet.load("acoustic", "2.4", "tf")
    elif bk == "pb":
        model = birdnet.load("acoustic", "2.4", "pb")
    else:  # pragma: no cover
        raise AssertionError(bk)
    _ACOUSTIC_MODEL_CACHE[cache_key] = model
    raw_disp = birdnet_inference_device()
    logger.info(
        "Loaded BirdNET acoustic 2.4: backend=%s session_device=%s "
        "(BIRDCLEF_BIRDNET_DEVICE=%r)",
        bk,
        session_key,
        raw_disp,
    )
    return model


def acoustic_birdnet_model() -> Any:
    """Shared BirdNET acoustic 2.4 model (PB+GPU when requested, otherwise TF-lite path @ CPU)."""
    return _get_acoustic_model()


def birdnet_sample_rate() -> int:
    """Sample rate expected by BirdNET acoustic v2.4 (weights download on first load)."""
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

    sess_dev = birdnet_session_device()
    pooled: list[np.ndarray] = []
    bsz = embed_batch_segments
    for i in range(0, len(segments), bsz):
        batch = segments[i : i + bsz]
        inputs = [(np.asarray(seg, dtype=np.float32), sample_rate) for seg in batch]
        res = model.encode_arrays(
            inputs,
            n_producers=1,
            n_workers=1,
            batch_size=min(len(inputs), bsz),
            prefetch_ratio=1,
            overlap_duration_s=0.0,
            half_precision=False,
            show_stats=None,
            device=sess_dev,
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

    bk, sess = birdnet_backend_and_session_device()
    logger.info(
        "BirdNET embedding: backend=%s session_device=%s (BIRDCLEF_BIRDNET_DEVICE=%s)",
        bk,
        sess,
        birdnet_inference_device(),
    )

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
