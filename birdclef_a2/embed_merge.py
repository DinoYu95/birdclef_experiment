"""Merge multiple embedding NPZs (same keys / compatible hyperparams) into one file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def merge_embedding_npz_files(out_npz: Path, input_npzs: list[Path]) -> None:
    """Concatenate ``X``, ``y``, ``filename``. Copy scalar meta from first NPZ."""
    if not input_npzs:
        raise ValueError("no input_npzs")
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    fns: list[np.ndarray] = []
    meta_arrays: dict[str, Any] = {}
    chunk_dims: list[int] = []

    for idx, p in enumerate(input_npzs):
        if not p.is_file():
            raise FileNotFoundError(str(p))
        d = np.load(p, allow_pickle=True)
        for req in ("X", "y", "filename"):
            if req not in d.files:
                raise KeyError(f"{p}: missing {req} in NPZ")

        Xi = np.asarray(d["X"], dtype=np.float32)
        if Xi.ndim != 2:
            raise ValueError(f"{p}: expected X.ndim==2, got {Xi.ndim}")
        Xs.append(Xi)
        chunk_dims.append(Xi.shape[0])

        yi = np.asarray(d["y"])
        fni = np.asarray(d["filename"])
        if yi.shape[0] != Xi.shape[0] or fni.shape[0] != Xi.shape[0]:
            raise ValueError(f"{p}: X/y/filename row counts mismatch")
        ys.append(yi)
        fns.append(fni)

        if idx == 0:
            for k in ("birdnet_sr", "segment_s", "overlap_s"):
                if k in d.files:
                    meta_arrays[k] = np.asarray(d[k])
        else:
            for k in ("birdnet_sr", "segment_s", "overlap_s"):
                if k in meta_arrays and k in d.files:
                    a0, a1 = np.asarray(meta_arrays[k]), np.asarray(d[k])
                    if not np.allclose(a0.astype(float), a1.astype(float)):
                        logger.warning(
                            "NPZ scalar meta %s differs between %s and first chunk — using first.",
                            k,
                            p.name,
                        )

        d.close()

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    filename = np.concatenate(fns, axis=0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict[str, Any] = {
        "X": X.astype(np.float32),
        "y": y,
        "filename": filename,
    }
    save_kw.update(meta_arrays)
    np.savez_compressed(out_npz, **save_kw)

    logger.info(
        "merged %s chunks -> %s rows into %s (chunk sizes=%s)",
        len(input_npzs),
        X.shape[0],
        out_npz,
        chunk_dims,
    )
