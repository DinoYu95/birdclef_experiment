from __future__ import annotations

from pathlib import Path


def assert_manifest_columns(df, *, need_filename: bool = True) -> None:
    if need_filename and "filename" not in df.columns:
        raise KeyError(f"manifest needs column 'filename'; got {list(df.columns)}")


def resolve_audio_path(
    data_root: Path,
    filename: str,
    *,
    relative_to: str = "train_audio",
) -> Path:
    """
    relative_to='train_audio': .../train_audio/<filename> (BirdCLEF default rows).
    relative_to='data_root': .../<filename> (use when CSV paths already include prefixes).
    """
    data_root = Path(data_root)
    if relative_to == "train_audio":
        return (data_root / "train_audio" / filename).resolve()
    if relative_to == "data_root":
        return (data_root / filename).resolve()
    raise ValueError("relative_to must be 'train_audio' or 'data_root'")


def normalize_manifest_paths_for_data_root(
    df,
    *,
    filename_col: str = "filename",
):
    """Prefix bare ``species/file.ogg`` rows with ``train_audio/`` for ``--audio-relative-to data_root``."""
    import pandas as pd

    out = df.copy()
    if filename_col not in out.columns:
        raise KeyError(f"manifest needs column {filename_col!r}")

    def _fix(fn: object) -> str:
        s = str(fn)
        if s.startswith(("train_audio/", "synthetic_train_audio/")):
            return s
        return f"train_audio/{s}"

    out[filename_col] = out[filename_col].map(_fix)
    return out
