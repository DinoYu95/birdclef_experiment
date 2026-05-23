from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_sklearn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model: str = "logreg",
    seed: int = 42,
) -> tuple[Pipeline, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    if model == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model == "hgb":
        clf = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
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
    report = classification_report(y_val, y_hat, zero_division=0)
    macro_f1 = float(f1_score(y_val, y_hat, average="macro", zero_division=0))
    return report, {"macro_f1": macro_f1}


def save_bundle(pipe: Pipeline, le: LabelEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "label_encoder": le}, path)


def load_bundle(path: Path) -> tuple[Pipeline, LabelEncoder]:
    obj = joblib.load(path)
    return obj["pipeline"], obj["label_encoder"]
