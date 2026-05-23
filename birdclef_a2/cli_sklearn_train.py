"""CLI: train / evaluate sklearn classifier on embedding NPZ files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from birdclef_a2.sklearn_head import evaluate, save_bundle, train_sklearn_classifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", type=Path, required=True)
    p.add_argument("--val-npz", type=Path, required=True)
    p.add_argument("--model", choices=("logreg", "hgb"), default="logreg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/sklearn_head"))
    args = p.parse_args()

    tr = np.load(args.train_npz, allow_pickle=True)
    va = np.load(args.val_npz, allow_pickle=True)
    pipe, le = train_sklearn_classifier(
        tr["X"], tr["y"], model=args.model, seed=args.seed
    )
    report, metrics = evaluate(pipe, le, va["X"], va["y"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out_dir / "model.joblib"
    save_bundle(pipe, le, bundle_path)
    (args.out_dir / "val_classification_report.txt").write_text(report, encoding="utf-8")
    (args.out_dir / "val_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Saved bundle to {bundle_path}")


if __name__ == "__main__":
    main()
