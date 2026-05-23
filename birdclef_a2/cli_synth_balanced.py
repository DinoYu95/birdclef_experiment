"""CLI: balanced synthetic clips via AudioLDM2 + manifest rows."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--verify-birdnet",
        action="store_true",
        help=(
            "After each wav: run BirdNET classifier (species names, not embeddings) "
            "and keep only clips whose predictions match taxonomy (see verify-only-bird-classes)."
        ),
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
        default=25,
        help="Max generation attempts per label = need * this value when verify is on.",
    )
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--limit-classes", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

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
        verify_only_bird_classes=not args.verify_include_non_aves,
        dtype=args.dtype,
        limit_classes=args.limit_classes,
        dry_run=args.dry_run,
    )
    synth_cfg_path = Path(args.out_manifest).parent / "synthetic_experiment_config.json"
    write_json(
        synth_cfg_path,
        {
            "cli": "cli_synth_balanced",
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )


if __name__ == "__main__":
    main()
