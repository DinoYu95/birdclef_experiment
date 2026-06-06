"""BirdNET verification for synthetic wav files (classification + embedding similarity).

Species mode uses ``AcousticModelV2_4.predict()`` on a 48 kHz / 3 s clip.
Embed mode compares BirdNET embeddings to a centroid built from real train audio.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef_a2.audio_io import load_audio_mono
from birdclef_a2.birdnet_embed import (
    acoustic_birdnet_model,
    birdnet_sample_rate,
    birdnet_session_device,
    compute_file_embedding,
)
from birdclef_a2.manifest_utils import resolve_audio_path

logger = logging.getLogger(__name__)

BIRDNET_VERIFY_SEGMENT_S = 3.0

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, distorted, noisy, music, speech, human voice, "
    "frog, insect, static, clipping"
)


def _prediction_result_to_df(pr) -> pd.DataFrame:
    if hasattr(pr, "to_arrow_table"):
        tbl = pr.to_arrow_table()
        try:
            return tbl.to_pandas()
        except Exception:  # pragma: no cover
            return pd.DataFrame(tbl.to_pylist())
    if hasattr(pr, "to_structured_array"):
        sa = pr.to_structured_array()
        return pd.DataFrame.from_records(sa)
    raise TypeError(f"Unhandled BirdNET prediction type: {type(pr)!r}")


def _df_species_confidence_pairs(df: pd.DataFrame) -> list[tuple[str, float]]:
    """Best-effort column detection across birdnet library versions."""
    lower = {c.lower(): c for c in df.columns}
    name_col = (
        lower.get("species_name")
        or lower.get("species")
        or lower.get("label")
        or lower.get("common_name")
    )
    conf_col = lower.get("confidence") or lower.get("score") or lower.get("probability")
    pairs: list[tuple[str, float]] = []
    if name_col is None:
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                name_col = c
                break
    if name_col is None:
        return pairs

    for _, row in df.iterrows():
        name = str(row[name_col])
        if conf_col and conf_col in row.index and pd.notna(row[conf_col]):
            try:
                conf = float(row[conf_col])
            except (TypeError, ValueError):
                conf = 1.0
        else:
            conf = 1.0
        pairs.append((name, conf))
    return pairs


def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def prediction_text_matches_taxonomy(pred_text: str, scientific_name: str, common_name: str) -> bool:
    """Heuristic: substring match on scientific / common tokens (BirdNET labels vary by locale)."""
    pl = _normalize_for_match(pred_text)
    sci = _normalize_for_match(scientific_name) if scientific_name else ""
    com = _normalize_for_match(common_name) if common_name else ""

    if sci and len(sci) > 2 and sci in pl:
        return True
    if sci:
        for part in sci.split():
            if len(part) > 3 and part in pl:
                return True
    if com:
        for w in com.split():
            if len(w) > 2 and w in pl:
                return True
    return False


def birdnet_center_segment_waveform(
    wav_path: Path,
    *,
    sample_rate: int | None = None,
    segment_s: float = BIRDNET_VERIFY_SEGMENT_S,
) -> tuple[np.ndarray, int]:
    """Resample to BirdNET rate and return a centered ``segment_s`` mono clip."""
    sr = int(sample_rate if sample_rate is not None else birdnet_sample_rate())
    wav = load_audio_mono(wav_path, sample_rate=sr)
    seg_n = max(int(round(segment_s * sr)), 1)
    if len(wav) >= seg_n:
        start = (len(wav) - seg_n) // 2
        return wav[start : start + seg_n].astype(np.float32, copy=False), sr
    pad = np.zeros(seg_n - len(wav), dtype=np.float32)
    return np.concatenate([wav, pad]).astype(np.float32, copy=False), sr


def _write_temp_birdnet_clip(wav_path: Path) -> Path:
    """Materialize 48 kHz / 3 s clip for ``predict()`` (BirdNET-native layout)."""
    import soundfile as sf

    clip, sr = birdnet_center_segment_waveform(wav_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    sf.write(tmp_path, clip, sr)
    return tmp_path


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def synthetic_passes_birdnet_classifier(
    wav_path: Path,
    *,
    scientific_name: str,
    common_name: str,
    top_k: int = 30,
    run_min_confidence: float = 0.04,
    row_min_confidence: float = 0.0,
) -> bool:
    """
    Returns True if any BirdNET prediction row matches taxonomy strings above thresholds.
    Uses a resampled 48 kHz / 3 s center clip before ``predict()``.
    """
    try:
        model = acoustic_birdnet_model()
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install PyPI `birdnet` for --verify-birdnet: pip install 'birdnet>=0.2.5'"
        ) from e

    tmp_path: Path | None = None
    try:
        tmp_path = _write_temp_birdnet_clip(wav_path)
        pr = model.predict(
            str(tmp_path),
            top_k=top_k,
            batch_size=1,
            n_producers=1,
            n_workers=1,
            default_confidence_threshold=run_min_confidence,
            show_stats=None,
            device=birdnet_session_device(),
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    df = _prediction_result_to_df(pr)
    pairs = _df_species_confidence_pairs(df)
    if not pairs:
        logger.warning("BirdNET returned no species rows for %s", wav_path)
        return False

    for text, conf in pairs:
        if conf < row_min_confidence:
            continue
        if prediction_text_matches_taxonomy(text, scientific_name, common_name):
            return True
    return False


def synthetic_passes_birdnet_embedding(
    wav_path: Path,
    *,
    centroid: np.ndarray,
    min_cosine: float,
    segment_s: float = BIRDNET_VERIFY_SEGMENT_S,
) -> bool:
    """True when BirdNET embedding cosine similarity to class centroid >= ``min_cosine``."""
    sr = birdnet_sample_rate()
    emb = compute_file_embedding(
        wav_path,
        sample_rate=sr,
        segment_s=segment_s,
        overlap_s=0.0,
        embed_batch_segments=1,
    )
    if emb is None:
        return False
    return _cosine_similarity(emb, centroid) >= float(min_cosine)


class TrainLabelCentroidCache:
    """Lazy per-label BirdNET embedding centroids from real train manifest rows."""

    def __init__(
        self,
        *,
        train_manifest: Path,
        data_root: Path,
        label_col: str,
        max_samples: int = 8,
        audio_relative_to: str = "train_audio",
        segment_s: float = BIRDNET_VERIFY_SEGMENT_S,
    ) -> None:
        self.train_manifest = Path(train_manifest)
        self.data_root = Path(data_root)
        self.label_col = label_col
        self.max_samples = max(1, int(max_samples))
        self.audio_relative_to = audio_relative_to
        self.segment_s = segment_s
        self._df = pd.read_csv(self.train_manifest)
        self._cache: dict[str, np.ndarray | None] = {}
        self._sr = birdnet_sample_rate()

    def get(self, primary_label: str) -> np.ndarray | None:
        lab = str(primary_label)
        if lab in self._cache:
            return self._cache[lab]

        sub = self._df[self._df[self.label_col].astype(str) == lab]
        if sub.empty:
            logger.warning("No train rows for centroid label %s", lab)
            self._cache[lab] = None
            return None

        paths: list[Path] = []
        for rel in sub["filename"].astype(str).head(self.max_samples):
            p = resolve_audio_path(
                self.data_root, rel, relative_to=self.audio_relative_to
            )
            if p.is_file():
                paths.append(p)

        if not paths:
            logger.warning("No on-disk train audio for centroid label %s", lab)
            self._cache[lab] = None
            return None

        vecs: list[np.ndarray] = []
        for p in paths:
            emb = compute_file_embedding(
                p,
                sample_rate=self._sr,
                segment_s=self.segment_s,
                overlap_s=0.0,
                embed_batch_segments=1,
            )
            if emb is not None:
                vecs.append(emb)

        if not vecs:
            logger.warning("Failed to embed train audio for centroid label %s", lab)
            self._cache[lab] = None
            return None

        centroid = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        self._cache[lab] = centroid
        logger.info(
            "Built BirdNET centroid for %s from %s/%s train clips",
            lab,
            len(vecs),
            len(paths),
        )
        return centroid


def synthetic_passes_birdnet_verify(
    wav_path: Path,
    *,
    verify_mode: str,
    scientific_name: str,
    common_name: str,
    centroid: np.ndarray | None,
    top_k: int,
    run_min_confidence: float,
    row_min_confidence: float,
    embed_min_cosine: float,
) -> bool:
    """
    ``verify_mode``: ``species`` | ``embed`` | ``both`` | ``either``.

    - species: taxonomy substring match in BirdNET top-k (48 kHz / 3 s clip)
    - embed: cosine similarity to train centroid
    - both: both must pass
    - either: at least one passes
    """
    mode = verify_mode.strip().lower()
    if mode not in ("species", "embed", "both", "either"):
        raise ValueError(f"verify_mode must be species|embed|both|either, got {verify_mode!r}")

    species_ok = False
    embed_ok = False

    if mode in ("species", "both", "either"):
        species_ok = synthetic_passes_birdnet_classifier(
            wav_path,
            scientific_name=scientific_name,
            common_name=common_name,
            top_k=top_k,
            run_min_confidence=run_min_confidence,
            row_min_confidence=row_min_confidence,
        )

    if mode in ("embed", "both", "either"):
        if centroid is None:
            embed_ok = False
        else:
            embed_ok = synthetic_passes_birdnet_embedding(
                wav_path,
                centroid=centroid,
                min_cosine=embed_min_cosine,
            )

    if mode == "species":
        return species_ok
    if mode == "embed":
        return embed_ok
    if mode == "both":
        return species_ok and embed_ok
    return species_ok or embed_ok
