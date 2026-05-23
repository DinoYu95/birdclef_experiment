"""CLI: train / evaluate sklearn classifier on embedding NPZ files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from birdclef_a2.report_exports import (
    classification_report_txt,
    save_confusion_matrix_csv,
    save_val_predictions_csv,
    scalar_metrics,
    write_json,
)
from birdclef_a2.sklearn_head import save_bundle, train_sklearn_classifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", type=Path, required=True)
    p.add_argument("--val-npz", type=Path, required=True)
    p.add_argument("--model", choices=("logreg", "hgb"), default="logreg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/sklearn_head"))
    p.add_argument(
        "--logreg-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression (smaller => stronger regularization).",
    )
    p.add_argument("--logreg-max-iter", type=int, default=2000)
    p.add_argument("--hgb-max-depth", type=int, default=8)
    p.add_argument("--hgb-learning-rate", type=float, default=0.05)
    p.add_argument("--hgb-max-iter", type=int, default=300)
    args = p.parse_args()

    tr = np.load(args.train_npz, allow_pickle=True)
    va = np.load(args.val_npz, allow_pickle=True)
    pipe, le = train_sklearn_classifier(
        tr["X"],
        tr["y"],
        model=args.model,
        seed=args.seed,
        logreg_c=args.logreg_c,
        logreg_max_iter=args.logreg_max_iter,
        hgb_max_depth=args.hgb_max_depth,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_iter=args.hgb_max_iter,
    )

    y_hat_enc = pipe.predict(va["X"])
    y_hat = le.inverse_transform(y_hat_enc.astype(int))

    report = classification_report_txt(va["y"], y_hat)
    metrics = scalar_metrics(va["y"], y_hat)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out_dir / "model.joblib"
    save_bundle(pipe, le, bundle_path)

    (args.out_dir / "val_classification_report.txt").write_text(report, encoding="utf-8")
    write_json(args.out_dir / "val_metrics.json", metrics)

    filenames = va["filename"] if "filename" in va.files else None
    save_val_predictions_csv(
        args.out_dir / "val_predictions.csv",
        y_true=va["y"],
        y_pred=y_hat,
        filenames=filenames,
    )
    save_confusion_matrix_csv(
        args.out_dir / "val_confusion_matrix.csv",
        y_true=va["y"],
        y_pred=y_hat,
    )

    write_json(
        args.out_dir / "experiment_config.json",
        {
            "cli": "cli_sklearn_train",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "train_npz_keys": list(tr.files),
            "val_npz_keys": list(va.files),
        },
    )

    print(report)
    print(json.dumps(metrics, indent=2))
    print(f"Saved bundle to {bundle_path}")
    print(f"Wrote predictions + confusion matrix under {args.out_dir}")


if __name__ == "__main__":
    main()
