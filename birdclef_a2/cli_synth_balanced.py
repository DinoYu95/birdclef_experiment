"""CLI: balanced synthetic clips via AudioLDM2 + manifest rows."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from birdclef_a2.birdnet_verify_synth import DEFAULT_NEGATIVE_PROMPT
from birdclef_a2.config import load_layout
from birdclef_a2.report_exports import write_json
from birdclef_a2.synth_audioldm2 import generate_balanced_synthetic_manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, default=None)
    p.add_argument("--taxonomy-csv", type=Path, default=None)
    p.add_argument("--label-col", default="primary_label")
    p.add_argument(
        "--out-audio-dir",
        type=Path,
        default=None,
        help="Directory under DATA root for wav files (default: synthetic_train_audio).",
    )
    p.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("outputs/synthetic/synthetic.manifest.csv"),
    )
    p.add_argument("--hf-model-id", default="cvssp/audioldm2")
    p.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Override target count; default = max count in train manifest.",
    )
    p.add_argument(
        "--max-target-cap",
        type=int,
        default=None,
        help="Clamp target per class after computing from max-count (VRAM/time control).",
    )
    p.add_argument("--audio-seconds", type=float, default=5.0)
    p.add_argument("--inference-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="AudioLDM2 negative prompt (discourage low-quality / non-bird artefacts).",
    )
    p.add_argument(
        "--verify-birdnet",
        action="store_true",
        help="After each wav: run BirdNET verification (see --verify-mode).",
    )
    p.add_argument(
        "--verify-mode",
        choices=("embed", "species", "both", "either"),
        default="embed",
        help=(
            "embed=cosine vs real-train centroid (default, higher pass rate); "
            "species=top-k name match; both/either=combine modes."
        ),
    )
    p.add_argument(
        "--verify-embed-min-cosine",
        type=float,
        default=0.55,
        help="Min cosine similarity for embed verify mode.",
    )
    p.add_argument(
        "--verify-centroid-max-samples",
        type=int,
        default=8,
        help="Real train clips per label when building BirdNET embedding centroids.",
    )
    p.add_argument(
        "--birdnet-device",
        default="CPU",
        help=(
            "BirdNET device for verify/embed centroids. Default CPU — AudioLDM2 uses "
            "PyTorch CUDA separately; TF BirdNET on GPU often crashes with CUDA_ERROR_NOT_INITIALIZED."
        ),
    )
    p.add_argument(
        "--birdnet-force-tf-cpu",
        action="store_true",
        help="Force BirdNET TF-lite CPU path even when --birdnet-device requests GPU.",
    )
    p.add_argument(
        "--verify-include-non-aves",
        action="store_true",
        help=(
            "Also run BirdNET on taxa where class_name is not Aves "
            "(BirdNET may rarely match insects/amphibians)."
        ),
    )
    p.add_argument("--verify-top-k", type=int, default=30)
    p.add_argument(
        "--verify-run-min-confidence",
        type=float,
        default=0.04,
        help="Passed to run_inference (model-side minimum confidence).",
    )
    p.add_argument(
        "--verify-row-min-confidence",
        type=float,
        default=0.0,
        help="Minimum per-row confidence in the prediction dataframe (post-hoc).",
    )
    p.add_argument(
        "--verify-max-retries",
        type=int,
        default=10,
        help=(
            "When --verify-birdnet: max generation attempts per deficit slot "
            "(need slots per label; failed slot moves on). Default 10. "
            "Without verify: one generation per slot."
        ),
    )
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--limit-classes", type=int, default=None)
    p.add_argument(
        "--existing-manifest",
        type=Path,
        default=None,
        help=(
            "Prior synthetic.manifest.csv; per-label slot counts are subtracted "
            "so completed classes are skipped."
        ),
    )
    p.add_argument(
        "--start-label",
        default=None,
        help=(
            "Only process labels >= this primary_label (alphabetical). "
            "Use ashgre1 to continue the bird queue without touching earlier taxa."
        ),
    )
    p.add_argument(
        "--only-aves",
        action="store_true",
        help="Only synthesize taxa with class_name=Aves (skip frog/mammal/etc.).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    os.environ["BIRDCLEF_BIRDNET_DEVICE"] = args.birdnet_device
    if args.birdnet_force_tf_cpu:
        os.environ["BIRDCLEF_BIRDNET_FORCE_TF_CPU"] = "1"

    layout = load_layout(args.data_root)
    tax = args.taxonomy_csv or (layout.root / "taxonomy.csv")
    if not tax.is_file():
        raise FileNotFoundError(f"taxonomy.csv not found: {tax}")

    out_audio = args.out_audio_dir or (layout.root / "synthetic_train_audio")

    generate_balanced_synthetic_manifest(
        data_root=layout.root,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        taxonomy_csv=tax,
        out_manifest=args.out_manifest,
        out_audio_dir=out_audio,
        label_col=args.label_col,
        hf_model_id=args.hf_model_id,
        target_per_class=args.target_per_class,
        max_target_cap=args.max_target_cap,
        audio_seconds=args.audio_seconds,
        inference_steps=args.inference_steps,
        seed=args.seed,
        verify_birdnet=args.verify_birdnet,
        verify_top_k=args.verify_top_k,
        verify_run_min_confidence=args.verify_run_min_confidence,
        verify_row_min_confidence=args.verify_row_min_confidence,
        verify_max_retries=args.verify_max_retries,
        verify_mode=args.verify_mode,
        verify_embed_min_cosine=args.verify_embed_min_cosine,
        verify_centroid_max_samples=args.verify_centroid_max_samples,
        verify_birdnet_device=args.birdnet_device,
        verify_only_bird_classes=not args.verify_include_non_aves,
        negative_prompt=args.negative_prompt,
        dtype=args.dtype,
        limit_classes=args.limit_classes,
        existing_manifest=args.existing_manifest,
        start_label=args.start_label,
        only_aves=args.only_aves,
        dry_run=args.dry_run,
    )
    synth_cfg_path = args.out_manifest.with_name(
        f"{args.out_manifest.stem}_experiment_config.json"
    )
    write_json(
        synth_cfg_path,
        {
            "cli": "cli_synth_balanced",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )


if __name__ == "__main__":
    main()
