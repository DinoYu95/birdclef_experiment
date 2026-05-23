"""Metrics and tables for experiment reports (file exports)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def scalar_metrics(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> dict[str, float]:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    out: dict[str, float] = {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
    }
    try:
        out["cohen_kappa"] = float(cohen_kappa_score(yt, yp))
    except ValueError:
        out["cohen_kappa"] = float("nan")
    return out


def save_val_predictions_csv(
    path: Path,
    *,
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    filenames: np.ndarray | list[str] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {
        "y_true": np.asarray(y_true).astype(str),
        "y_pred": np.asarray(y_pred).astype(str),
    }
    if filenames is not None:
        rows["filename"] = np.asarray(filenames).astype(str)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def save_confusion_matrix_csv(
    path: Path,
    *,
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
) -> None:
    yt = np.asarray(y_true).astype(str)
    yp = np.asarray(y_pred).astype(str)
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm = confusion_matrix(yt, yp, labels=labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=[f"true_{lab}" for lab in labels], columns=labels).to_csv(
        path, encoding="utf-8"
    )


def classification_report_txt(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> str:
    return classification_report(
        np.asarray(y_true).astype(str),
        np.asarray(y_pred).astype(str),
        zero_division=0,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
