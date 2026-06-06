from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from birdclef_a2.report_exports import classification_report_txt, scalar_metrics


def train_sklearn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model: str = "logreg",
    seed: int = 42,
    logreg_c: float = 1.0,
    logreg_max_iter: int = 2000,
    hgb_max_depth: int = 8,
    hgb_learning_rate: float = 0.05,
    hgb_max_iter: int = 300,
) -> tuple[Pipeline, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    if model == "logreg":
        clf = LogisticRegression(
            C=logreg_c,
            max_iter=logreg_max_iter,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model == "hgb":
        # early_stopping splits train internally with stratify; singleton classes in
        # BirdCLEF train trigger ValueError — use fixed max_iter instead.
        clf = HistGradientBoostingClassifier(
            max_depth=hgb_max_depth,
            learning_rate=hgb_learning_rate,
            max_iter=hgb_max_iter,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError("model must be 'logreg' or 'hgb'")

    pipe = Pipeline([("scale", StandardScaler(with_mean=True, with_std=True)), ("clf", clf)])
    pipe.fit(X_train, y_enc)
    return pipe, le


def evaluate(
    pipe: Pipeline,
    le: LabelEncoder,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[str, dict]:
    y_hat_enc = pipe.predict(X_val)
    y_hat = le.inverse_transform(y_hat_enc.astype(int))
    report = classification_report_txt(y_val, y_hat)
    metrics = scalar_metrics(y_val, y_hat)
    return report, metrics


def save_bundle(pipe: Pipeline, le: LabelEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "label_encoder": le}, path)


def load_bundle(path: Path) -> tuple[Pipeline, LabelEncoder]:
    obj = joblib.load(path)
    return obj["pipeline"], obj["label_encoder"]
