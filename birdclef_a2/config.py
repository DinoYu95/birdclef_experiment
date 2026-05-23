from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLayout:
    """Paths under the unzipped Kaggle competition bundle (defaults match typical layout)."""

    root: Path
    train_csv_name: str = "train.csv"
    train_audio_subdir: str = "train_audio"

    @property
    def train_csv(self) -> Path:
        return self.root / self.train_csv_name

    @property
    def train_audio_dir(self) -> Path:
        return self.root / self.train_audio_subdir


def resolve_data_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("BIRDCLEF_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    raise FileNotFoundError(
        "Data root missing: pass --data-root or set BIRDCLEF_DATA_ROOT."
    )


def load_layout(root: str | Path | None = None) -> DataLayout:
    r = resolve_data_root(str(root) if root is not None else None)
    return DataLayout(root=r)
