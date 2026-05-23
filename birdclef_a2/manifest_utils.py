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
