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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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


def _embed_segment_inputs_chunk_cap() -> int:
    """Each ``encode_arrays`` opens a multiprocessing-heavy BirdNET session.

    Default: encode **all segments of one file** in one call (minimal sessions).
    If one file yields huge segment counts / OOM, set ``BIRDCLEF_EMBED_ENCODE_SEGMENT_CHUNK``
    e.g. ``256``.
    """
    raw = os.environ.get("BIRDCLEF_EMBED_ENCODE_SEGMENT_CHUNK", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 999_999


def _parse_embed_parallel_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default).strip()
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{name} must be a positive integer (got {raw!r})")
    return int(raw)


def embed_encode_n_workers() -> int | None:
    """BirdNET ``encode_arrays`` worker count.

    Default ``1`` (safe on low ``ulimit`` hosts). Set ``auto`` to use birdnet default
    (~physical CPU cores). Any positive integer is passed through.
    """
    raw = os.environ.get("BIRDCLEF_EMBED_N_WORKERS", "1").strip().lower()
    if raw in ("auto", "default"):
        return None
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    raise ValueError(
        "BIRDCLEF_EMBED_N_WORKERS must be a positive integer or 'auto' "
        f"(got {raw!r})"
    )


def embed_encode_n_producers() -> int:
    return _parse_embed_parallel_int("BIRDCLEF_EMBED_N_PRODUCERS", "1")


def embed_encode_prefetch_ratio() -> int:
    return _parse_embed_parallel_int("BIRDCLEF_EMBED_PREFETCH_RATIO", "1")


def log_embed_parallel_settings_once() -> None:
    """Log BirdNET pipeline parallelism once per process."""
    if getattr(log_embed_parallel_settings_once, "_done", False):
        return
    nw = embed_encode_n_workers()
    logger.info(
        "BirdNET encode_arrays parallelism: n_workers=%s n_producers=%s prefetch_ratio=%s "
        "(env BIRDCLEF_EMBED_N_WORKERS=%r; use 'auto' for birdnet physical-core default)",
        nw if nw is not None else "auto",
        embed_encode_n_producers(),
        embed_encode_prefetch_ratio(),
        os.environ.get("BIRDCLEF_EMBED_N_WORKERS", "1"),
    )
    log_embed_parallel_settings_once._done = True  # type: ignore[attr-defined]


def compute_file_embedding(
    audio_path: Path,
    *,
    sample_rate: int,
    segment_s: float,
    overlap_s: float,
    embed_batch_segments: int,
) -> np.ndarray | None:
    model = acoustic_birdnet_model()

    logger.debug("embedding: decode start %s", audio_path)

    try:
        wav = load_audio_mono(audio_path, sample_rate=sample_rate)
    except Exception as exc:  # pragma: no cover
        logger.warning("failed to load %s: %s", audio_path, exc)
        return None

    logger.info(
        "embedding: decoded %s -> %s samples @ %s Hz (BirdNET inference next)",
        audio_path.name,
        len(wav),
        sample_rate,
    )

    segments = iter_segments(
        wav, sample_rate=sample_rate, segment_s=segment_s, overlap_s=overlap_s
    )
    if not segments:
        return None

    sess_dev = birdnet_session_device()
    # One BirdNET encode session per chunk — NOT per segment-batch (would fork workers endlessly).
    inputs_all = [(np.asarray(seg, dtype=np.float32), sample_rate) for seg in segments]
    logger.info(
        "embedding: %s BirdNET encode — %s segment(s), batch≤%s (first call can be slow on CPU)",
        audio_path.name,
        len(inputs_all),
        embed_batch_segments,
    )
    cap = _embed_segment_inputs_chunk_cap()
    pooled: list[np.ndarray] = []
    n_workers = embed_encode_n_workers()
    n_producers = embed_encode_n_producers()
    prefetch_ratio = embed_encode_prefetch_ratio()
    log_embed_parallel_settings_once()

    for c0 in range(0, len(inputs_all), cap):
        chunk = inputs_all[c0 : c0 + cap]
        if c0 == 0 and len(inputs_all) > cap:
            logger.info(
                "embedding: %s encoding segment batch 0-%s/%s …",
                audio_path.name,
                min(len(chunk), len(inputs_all)),
                len(inputs_all),
            )
        res = model.encode_arrays(
            chunk,
            n_producers=n_producers,
            n_workers=n_workers,
            batch_size=min(len(chunk), embed_batch_segments),
            prefetch_ratio=prefetch_ratio,
            overlap_duration_s=0.0,
            half_precision=False,
            show_stats=None,
            device=sess_dev,
        )
        for j in range(len(chunk)):
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
    offset_rows: int = 0,
    limit_rows: int | None = None,
    audio_relative_to: str = "train_audio",
    manifest_df: pd.DataFrame | None = None,
) -> None:
    df = manifest_df if manifest_df is not None else pd.read_csv(manifest_csv)
    assert_manifest_columns(df)
    if label_col not in df.columns:
        raise KeyError(f"missing {label_col}")

    n_manifest = len(df)
    start = int(offset_rows)
    if start < 0:
        raise ValueError("offset_rows must be >= 0")
    if start >= n_manifest:
        raise ValueError(f"offset_rows={start} >= manifest rows ({n_manifest})")
    stop = n_manifest if limit_rows is None else min(n_manifest, start + int(limit_rows))
    sub = df.iloc[start:stop]

    bk, sess = birdnet_backend_and_session_device()
    logger.info(
        "BirdNET embedding: backend=%s session_device=%s (BIRDCLEF_BIRDNET_DEVICE=%s)",
        bk,
        sess,
        birdnet_inference_device(),
    )
    logger.info(
        "manifest slice: rows [%s:%s] (%s rows of %s total)",
        start,
        stop,
        len(sub),
        n_manifest,
    )

    sr = birdnet_sample_rate()
    xs: list[np.ndarray] = []
    ys: list[str] = []
    paths_ok: list[str] = []

    n_chunk = len(sub)
    for i in range(n_chunk):
        row = sub.iloc[i]
        rel = str(row["filename"])
        # Heartbeat before slow BirdNET/audio work: previously we only logged every 50 *successful*
        # embeddings, so CPU-first-file could run for hours with no INFO at all.
        if i == 0 or (i + 1) % 50 == 0:
            logger.info(
                "manifest progress: slice row %s / %s (csv row ~ %s / %s), embeddings_ok=%s, "
                "next_file=%s",
                i + 1,
                n_chunk,
                start + i + 1,
                n_manifest,
                len(xs),
                rel,
            )

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
