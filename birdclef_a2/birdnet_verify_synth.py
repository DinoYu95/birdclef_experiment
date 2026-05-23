"""BirdNET verification for synthetic wav files (classification output, not embeddings).

`birdnet_analyzer.model_utils.run_inference` runs the **species classifier** and returns
structured results (`to_dataframe()`). That is **not** the same as `get_embeddings_array`,
which returns fixed-length feature vectors for sklearn.

For filtering synthetic clips we match **predicted species text** (plus confidence) against
taxonomy `scientific_name` / `common_name`. This aligns with the assignment hint:
if BirdNET thinks the clip sounds like the intended taxon, keep it.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


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
        # Binomial pieces (genus / epithet)
        for part in sci.split():
            if len(part) > 3 and part in pl:
                return True
    if com:
        for w in com.split():
            if len(w) > 2 and w in pl:
                return True
    return False


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

    `run_min_confidence` is passed into `run_inference` (model-side filter).
    `row_min_confidence` filters rows after `to_dataframe()` (post-hoc).
    """
    try:
        from birdnet_analyzer.model_utils import run_inference
    except ImportError as e:  # pragma: no cover
        raise ImportError("Install birdnet-analyzer for --verify-birdnet") from e

    res = run_inference(
        str(wav_path),
        top_k=top_k,
        batch_size=1,
        min_confidence=run_min_confidence,
    )
    df = res.to_dataframe()
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
