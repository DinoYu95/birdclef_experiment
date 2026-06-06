"""
Synthetic audio augmentation using AudioLDM2 (diffusers).

Course briefs sometimes refer to this family as \"AudioLM2\"; here we use the public
AudioLDM2 checkpoints from Hugging Face `cvssp/audioldm2*`.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef_a2.birdnet_verify_synth import synthetic_passes_birdnet_classifier

logger = logging.getLogger(__name__)


def prompt_for_taxon(
    primary_label: str,
    *,
    primary_to_common: dict[str, str],
    primary_to_sci: dict[str, str],
) -> str:
    cn = primary_to_common.get(primary_label, "wildlife sound")
    sn = primary_to_sci.get(primary_label, "")
    bits = [
        "high quality mono field recording",
        f"animal vocalization of {cn}",
    ]
    if sn:
        bits.append(sn)
    bits.append("natural ambient noise, no music")
    return ", ".join(bits)


def load_taxonomy_maps(
    taxonomy_csv: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    df = pd.read_csv(taxonomy_csv)
    pl = df["primary_label"].astype(str)
    common = dict(zip(pl.tolist(), df["common_name"].astype(str).tolist()))
    sci = dict(zip(pl.tolist(), df["scientific_name"].astype(str).tolist()))
    cls_col = "class_name" if "class_name" in df.columns else None
    cls_map: dict[str, str] = (
        dict(zip(pl.tolist(), df["class_name"].astype(str).tolist())) if cls_col else {}
    )
    return common, sci, cls_map


def compute_training_deficits(
    *,
    train_manifest: Path,
    val_manifest: Path | None,
    label_col: str,
    target_per_class: int,
) -> dict[str, int]:
    tr = pd.read_csv(train_manifest)
    vc = tr[label_col].astype(str).value_counts()
    labs = set(vc.index.astype(str))
    if val_manifest is not None:
        va = pd.read_csv(val_manifest)
        labs |= set(va[label_col].astype(str))
    deficits: dict[str, int] = {}
    for lab in sorted(labs):
        cnt = int(vc.get(lab, 0))
        need = int(target_per_class) - cnt
        if need > 0:
            deficits[lab] = need
    return deficits


def _existing_synth_counts(
    existing_manifest: Path | None,
    label_col: str,
) -> dict[str, int]:
    if existing_manifest is None or not existing_manifest.is_file():
        return {}
    df = pd.read_csv(existing_manifest)
    if label_col not in df.columns:
        raise ValueError(f"{existing_manifest} missing column {label_col!r}")
    return df[label_col].astype(str).value_counts().astype(int).to_dict()


def compute_remaining_deficits(
    *,
    train_manifest: Path,
    val_manifest: Path | None,
    label_col: str,
    target_per_class: int,
    existing_manifest: Path | None = None,
    start_label: str | None = None,
    only_aves: bool = False,
    primary_to_class: dict[str, str] | None = None,
) -> dict[str, int]:
    """Deficit slots still to fill, subtracting prior synthetic manifest rows."""
    tr = pd.read_csv(train_manifest)
    vc = tr[label_col].astype(str).value_counts()
    labs = set(vc.index.astype(str))
    if val_manifest is not None:
        va = pd.read_csv(val_manifest)
        labs |= set(va[label_col].astype(str))

    existing = _existing_synth_counts(existing_manifest, label_col)
    deficits: dict[str, int] = {}
    for lab in sorted(labs):
        if start_label is not None and lab < start_label:
            continue
        if only_aves:
            cls_name = (primary_to_class or {}).get(lab, "")
            if not _taxon_is_bird(cls_name):
                continue
        real_cnt = int(vc.get(lab, 0))
        synth_cnt = int(existing.get(lab, 0))
        need = int(target_per_class) - real_cnt - synth_cnt
        if need > 0:
            deficits[lab] = need
    return deficits


def _taxon_is_bird(class_name: str) -> bool:
    return class_name.strip().lower() == "aves"


def _estimate_max_attempts(
    items: list[tuple[str, int]],
    verify_birdnet: bool,
    verify_only_bird_classes: bool,
    primary_to_class: dict[str, str],
    verify_max_retries: int,
) -> int:
    total = 0
    for lab, need in items:
        cls_name = primary_to_class.get(lab, "")
        run_verify = bool(verify_birdnet) and (
            not verify_only_bird_classes or _taxon_is_bird(cls_name)
        )
        tries = verify_max_retries if run_verify else 1
        total += need * tries
    return total


def _load_audioldm2_pipeline(hf_model_id: str, torch_dtype):
    """Load AudioLDM2 with ``GPT2LMHeadModel`` — newer ``transformers`` breaks ``GPT2Model`` in the HF repo."""
    from diffusers import AudioLDM2Pipeline
    from transformers import GPT2LMHeadModel

    logger.info(
        "Loading AudioLDM2 %s (language_model=GPT2LMHeadModel for transformers compat)",
        hf_model_id,
    )
    lm = GPT2LMHeadModel.from_pretrained(
        hf_model_id, subfolder="language_model", torch_dtype=torch_dtype
    )
    return AudioLDM2Pipeline.from_pretrained(
        hf_model_id, torch_dtype=torch_dtype, language_model=lm
    )


def generate_balanced_synthetic_manifest(
    *,
    data_root: Path,
    train_manifest: Path,
    val_manifest: Path | None,
    taxonomy_csv: Path,
    out_manifest: Path,
    out_audio_dir: Path,
    label_col: str,
    hf_model_id: str,
    target_per_class: int | None,
    max_target_cap: int | None,
    audio_seconds: float,
    inference_steps: int,
    seed: int,
    verify_birdnet: bool,
    verify_top_k: int,
    verify_run_min_confidence: float,
    verify_row_min_confidence: float,
    verify_max_retries: int,
    verify_only_bird_classes: bool,
    dtype: str,
    limit_classes: int | None,
    existing_manifest: Path | None,
    start_label: str | None,
    only_aves: bool,
    dry_run: bool,
) -> None:
    primary_to_common, primary_to_sci, primary_to_class = load_taxonomy_maps(taxonomy_csv)
    df_tr = pd.read_csv(train_manifest)
    vc = df_tr[label_col].astype(str).value_counts()
    max_cnt = int(vc.max())

    tgt = max_cnt if target_per_class is None else int(target_per_class)
    if max_target_cap is not None:
        tgt = min(tgt, int(max_target_cap))

    if existing_manifest is not None or start_label is not None or only_aves:
        deficits = compute_remaining_deficits(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            label_col=label_col,
            target_per_class=tgt,
            existing_manifest=existing_manifest,
            start_label=start_label,
            only_aves=only_aves,
            primary_to_class=primary_to_class,
        )
    else:
        deficits = compute_training_deficits(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            label_col=label_col,
            target_per_class=tgt,
        )
    items = sorted(deficits.items(), key=lambda x: x[0])
    if limit_classes is not None:
        items = items[: int(limit_classes)]

    out_audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    total_new = sum(n for _, n in items)
    existing = _existing_synth_counts(existing_manifest, label_col)

    if dry_run:
        print(
            f"target_per_class={tgt} labels_to_run={len(items)} "
            f"slots_to_try={total_new} max_attempts~={_estimate_max_attempts(items, verify_birdnet, verify_only_bird_classes, primary_to_class, verify_max_retries)}"
        )
        if existing_manifest is not None:
            print(f"existing_manifest={existing_manifest} prior_rows={sum(existing.values())}")
        if start_label is not None:
            print(f"start_label={start_label!r} (skip alphabetically earlier)")
        if only_aves:
            print("only_aves=True (non-bird taxa skipped)")
        for lab, need in items[:15]:
            real = int(vc.get(lab, 0))
            prior = int(existing.get(lab, 0))
            print(f"  {lab}: real_train={real} prior_synth={prior} slots={need}")
        if len(items) > 15:
            print(f"  ... and {len(items) - 15} more labels")
        return

    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Synthetic generation requires: pip install diffusers transformers accelerate torch soundfile"
        ) from e

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    pipe = _load_audioldm2_pipeline(hf_model_id, torch_dtype)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        try:
            pipe.enable_attention_slicing()
        except Exception:  # pragma: no cover
            pass
    else:
        pipe.enable_model_cpu_offload()

    try:
        sr = int(pipe.vocoder.config.sampling_rate)
    except Exception:
        sr = 16000

    import soundfile as sf

    logger.info(
        "Resume plan: target=%s labels=%s slots=%s existing_manifest=%s start_label=%s only_aves=%s",
        tgt,
        len(items),
        total_new,
        existing_manifest,
        start_label,
        only_aves,
    )

    done = 0
    n_rejected = 0
    for lab, need in items:
        subdir = out_audio_dir / lab
        subdir.mkdir(parents=True, exist_ok=True)
        prompt = prompt_for_taxon(
            lab, primary_to_common=primary_to_common, primary_to_sci=primary_to_sci
        )

        sci_n = primary_to_sci.get(lab, "")
        com_n = primary_to_common.get(lab, "")
        cls_name = primary_to_class.get(lab, "")
        run_birdnet_verify = bool(verify_birdnet) and (
            not verify_only_bird_classes or _taxon_is_bird(cls_name)
        )
        if verify_birdnet and verify_only_bird_classes and not _taxon_is_bird(cls_name):
            logger.info(
                "BirdNET verify skipped for non-Aves taxon %s (class_name=%r)",
                lab,
                cls_name,
            )

        produced = 0
        attempts = 0
        tries_per_slot = verify_max_retries if run_birdnet_verify else 1

        # One "slot" = one deficit clip. Up to tries_per_slot generations per slot;
        # if all tries fail, move on. Never spend more than need * tries_per_slot
        # attempts on a single label (avoids infinite retry on hard taxa).
        for _slot in range(need):
            for _try in range(tries_per_slot):
                attempts += 1
                fname = f"{uuid.uuid4().hex}.wav"
                out_wav = subdir / fname

                out = pipe(
                    prompt,
                    audio_length_in_s=float(audio_seconds),
                    num_inference_steps=int(inference_steps),
                )
                audio = np.asarray(out.audios[0], dtype=np.float32)

                sf.write(out_wav, audio, sr)

                if run_birdnet_verify:
                    ok = synthetic_passes_birdnet_classifier(
                        out_wav,
                        scientific_name=sci_n,
                        common_name=com_n,
                        top_k=verify_top_k,
                        run_min_confidence=verify_run_min_confidence,
                        row_min_confidence=verify_row_min_confidence,
                    )
                    if not ok:
                        n_rejected += 1
                        try:
                            out_wav.unlink()
                        except OSError:
                            pass
                        continue

                rel_path = Path(out_wav).resolve().relative_to(
                    Path(data_root).resolve()
                )
                rows.append({"filename": rel_path.as_posix(), label_col: lab})
                produced += 1
                done += 1
                if done % 10 == 0:
                    logger.info("synthetic accepted %s / ~%s total", done, total_new)
                break

        if produced < need:
            logger.warning(
                "label %s: filled %s/%s slots (%s slots exhausted all %s tries; "
                "total_attempts=%s verify=%s)",
                lab,
                produced,
                need,
                need - produced,
                tries_per_slot,
                attempts,
                verify_birdnet,
            )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(
        f"Wrote {len(rows)} synthetic rows to {out_manifest} "
        f"(birdnet_rejected_segments={n_rejected})"
    )
