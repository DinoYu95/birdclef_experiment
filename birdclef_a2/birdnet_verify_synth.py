"""BirdNET verification for synthetic wav files (classification output, not embeddings).

Uses the PyPI `birdnet` package `AcousticModelV2_4.predict()` (species scores), **not**
the embedding vectors used by `birdnet_embed.py`.

For filtering synthetic clips we match **predicted species text** (plus confidence) against
taxonomy `scientific_name` / `common_name`. This aligns with the assignment hint:
if BirdNET thinks the clip sounds like the intended taxon, keep it.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from birdclef_a2.birdnet_embed import acoustic_birdnet_model, birdnet_session_device

logger = logging.getLogger(__name__)


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

    `run_min_confidence` maps to BirdNET `default_confidence_threshold`.
    `row_min_confidence` filters rows after building a DataFrame (post-hoc).
    """
    try:
        model = acoustic_birdnet_model()
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install PyPI `birdnet` for --verify-birdnet: pip install 'birdnet>=0.2.5'"
        ) from e

    pr = model.predict(
        str(wav_path),
        top_k=top_k,
        batch_size=1,
        default_confidence_threshold=run_min_confidence,
        show_stats=None,
        device=birdnet_session_device(),
    )
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
