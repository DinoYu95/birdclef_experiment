"""CLI: train tiny mel-CNN from scratch on competition manifests."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from birdclef_a2.config import load_layout
from birdclef_a2.torch_scratch import train_torch_classifier


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, required=True)
    p.add_argument("--label-col", default="primary_label")
    p.add_argument(
        "--audio-relative-to",
        choices=("train_audio", "data_root"),
        default="train_audio",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/torch_scratch"))
    args = p.parse_args()

    layout = load_layout(args.data_root)
    train_torch_classifier(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        data_root=layout.root,
        label_col=args.label_col,
        relative_to=args.audio_relative_to,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
