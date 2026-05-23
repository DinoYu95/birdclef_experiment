"""CLI: write train/val manifests under assignment split rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from birdclef_a2.config import load_layout
from birdclef_a2.split import stratified_train_val_with_singletons_forced_to_val


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "BirdCLEF+ 2026: stratified train/val split; classes with a single row "
            "(singleton primary_label) are forced into validation only."
        )
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Unzipped competition root (contains train.csv and train_audio/). "
        "Or set env BIRDCLEF_DATA_ROOT.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/splits"),
        help="Write train.manifest.csv, val.manifest.csv, split_summary.json",
    )
    p.add_argument("--label-col", default="primary_label", help="Label column name")
    p.add_argument(
        "--group-col",
        default=None,
        help="Optional column: keep whole group in train or val only (no leakage).",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    layout = load_layout(args.data_root)
    if not layout.train_csv.is_file():
        raise FileNotFoundError(f"train.csv not found: {layout.train_csv}")

    df = pd.read_csv(layout.train_csv)

    spl = stratified_train_val_with_singletons_forced_to_val(
        df,
        label_col=args.label_col,
        val_fraction=args.val_fraction,
        seed=args.seed,
        group_col=args.group_col,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.manifest.csv"
    val_path = args.out_dir / "val.manifest.csv"
    spl.train.to_csv(train_path, index=False)
    spl.val.to_csv(val_path, index=False)

    summary = {
        "data_root": str(layout.root),
        "train_csv": str(layout.train_csv),
        "n_total": int(len(df)),
        "n_train": int(len(spl.train)),
        "n_val": int(len(spl.val)),
        "val_fraction_requested": args.val_fraction,
        "fraction_val_actual": round(len(spl.val) / max(len(df), 1), 6),
        "n_singleton_labels": len(spl.singleton_labels),
        "n_singleton_rows": spl.n_singleton_rows,
        "n_classes_total": int(df[args.label_col].nunique()),
        "seed": args.seed,
    }
    (args.out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote:\n  {train_path}\n  {val_path}")


if __name__ == "__main__":
    main()
