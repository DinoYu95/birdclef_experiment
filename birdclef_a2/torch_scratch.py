from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

from birdclef_a2.audio_io import load_audio_mono
from birdclef_a2.manifest_utils import assert_manifest_columns, resolve_audio_path
from birdclef_a2.report_exports import (
    classification_report_txt,
    save_confusion_matrix_csv,
    save_val_predictions_csv,
    scalar_metrics,
    write_json,
)

logger = logging.getLogger(__name__)


class ManifestMelDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Path,
        data_root: Path,
        *,
        label_col: str,
        classes: list[str],
        label_to_idx: dict[str, int],
        segment_s: float = 5.0,
        sample_rate: int = 32000,
        relative_to: str = "train_audio",
        train_mode: bool = True,
        seed: int = 42,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop: int = 256,
    ) -> None:
        self.df = pd.read_csv(manifest_csv)
        assert_manifest_columns(self.df)
        self.data_root = Path(data_root)
        self.label_col = label_col
        self.segment_s = segment_s
        self.sr = sample_rate
        self.relative_to = relative_to
        self.train_mode = train_mode
        self.rng = np.random.default_rng(seed)
        self.label_to_idx = label_to_idx
        self.classes = classes

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        rel = str(row["filename"])
        path = resolve_audio_path(self.data_root, rel, relative_to=self.relative_to)
        wav_np = load_audio_mono(path, sample_rate=self.sr)
        seg_n = int(self.segment_s * self.sr)
        if len(wav_np) <= seg_n:
            pad = np.zeros(seg_n - len(wav_np), dtype=np.float32)
            seg = np.concatenate([wav_np, pad])
        else:
            if self.train_mode:
                start = int(self.rng.integers(0, len(wav_np) - seg_n + 1))
            else:
                start = max(0, (len(wav_np) - seg_n) // 2)
            seg = wav_np[start : start + seg_n]

        wav = torch.from_numpy(seg).unsqueeze(0)
        mel = self.db(self.mel(wav)).clamp(min=-80, max=80)
        mel = (mel + 80) / 80.0

        lab = str(row[self.label_col])
        y = self.label_to_idx[lab]
        return mel, y


class TinyMelCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 16)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_label_mapping(
    train_manifest: Path,
    val_manifest: Path,
    label_col: str,
) -> tuple[list[str], dict[str, int]]:
    tra = pd.read_csv(train_manifest)
    val = pd.read_csv(val_manifest)
    classes = sorted(
        set(tra[label_col].astype(str).tolist())
        | set(val[label_col].astype(str).tolist())
    )
    m = {c: i for i, c in enumerate(classes)}
    return classes, m


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    acc = correct / max(total, 1)
    return acc, correct


def train_torch_classifier(
    *,
    train_manifest: Path,
    val_manifest: Path,
    data_root: Path,
    label_col: str,
    relative_to: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    num_workers: int,
) -> None:
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes, label_to_idx = fit_label_mapping(train_manifest, val_manifest, label_col)
    n_classes = len(classes)
    (out_dir / "classes.json").write_text(
        __import__("json").dumps(classes), encoding="utf-8"
    )

    ds_tr = ManifestMelDataset(
        train_manifest,
        data_root,
        label_col=label_col,
        classes=classes,
        label_to_idx=label_to_idx,
        train_mode=True,
        seed=seed,
        relative_to=relative_to,
    )
    ds_va = ManifestMelDataset(
        val_manifest,
        data_root,
        label_col=label_col,
        classes=classes,
        label_to_idx=label_to_idx,
        train_mode=False,
        seed=seed + 1,
        relative_to=relative_to,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train = len(ds_tr)
    n_val = len(ds_va)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    logger.info(
        "TinyMelCNN: device=%s train=%s val=%s classes=%s batch_size=%s steps/epoch≈%s",
        device,
        n_train,
        n_val,
        n_classes,
        batch_size,
        steps_per_epoch,
    )
    if device.type == "cpu":
        logger.warning(
            "Training on CPU — first epoch can take a long time before any log; "
            "install torch+cu124 and re-run for GPU."
        )

    model = TinyMelCNN(n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = out_dir / "melcnn_best.pt"

    for epoch in range(epochs):
        model.train()
        losses = []
        for step, (xb, yb) in enumerate(dl_tr, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if step == 1 or step % 100 == 0 or step == steps_per_epoch:
                logger.info(
                    "epoch %s train batch %s/%s loss=%.4f",
                    epoch + 1,
                    step,
                    steps_per_epoch,
                    losses[-1],
                )

        logger.info("epoch %s running validation (%s samples)…", epoch + 1, n_val)
        va_acc, _ = evaluate_loader(model, dl_va, device)
        logger.info(
            "epoch %s train_loss=%.4f val_acc=%.4f",
            epoch + 1,
            float(np.mean(losses)) if losses else 0.0,
            va_acc,
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {"model": model.state_dict(), "classes": classes, "label_col": label_col},
                best_path,
            )

    print(f"Best val acc ~ {best_acc:.4f}; checkpoint: {best_path}")

    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    idx_cursor = 0
    fnames_collect: list[str] = []
    ys: list[int] = []
    ps: list[int] = []
    with torch.no_grad():
        for xb, yb in dl_va:
            b = int(yb.shape[0])
            fnames_collect.extend(
                ds_va.df.iloc[idx_cursor : idx_cursor + b]["filename"].astype(str).tolist()
            )
            idx_cursor += b
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            ps.extend(pred)
            ys.extend(yb.numpy().tolist())
    y_names = [classes[i] for i in ys]
    p_names = [classes[i] for i in ps]
    report = classification_report_txt(y_names, p_names)
    metrics = scalar_metrics(y_names, p_names)
    metrics["best_epoch_val_accuracy"] = float(best_acc)

    (out_dir / "val_classification_report.txt").write_text(report, encoding="utf-8")
    write_json(out_dir / "val_metrics.json", metrics)
    save_val_predictions_csv(
        out_dir / "val_predictions.csv",
        y_true=y_names,
        y_pred=p_names,
        filenames=fnames_collect,
    )
    save_confusion_matrix_csv(
        out_dir / "val_confusion_matrix.csv",
        y_true=y_names,
        y_pred=p_names,
    )
    print(f"Wrote val report to {out_dir / 'val_classification_report.txt'}")
