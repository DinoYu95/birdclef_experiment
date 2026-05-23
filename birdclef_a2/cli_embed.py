"""CLI: BirdNET embeddings -> compressed NPZ (X, y, filename)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from birdclef_a2.birdnet_embed import manifest_to_embeddings_npz
from birdclef_a2.config import load_layout
from birdclef_a2.report_exports import write_json


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="BirdNET 2.4 embeddings for a manifest CSV")
    p.add_argument("--data-root", default=None)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--label-col", default="primary_label")
    p.add_argument("--segment-s", type=float, default=3.0)
    p.add_argument("--overlap-s", type=float, default=1.5)
    p.add_argument("--embed-batch-segments", type=int, default=16)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument(
        "--audio-relative-to",
        choices=("train_audio", "data_root"),
        default="train_audio",
        help="train_audio: paths join DATA/train_audio/<filename>; "
        "data_root: paths join DATA/<filename> (for merged synth manifests).",
    )
    args = p.parse_args()

    layout = load_layout(args.data_root)
    manifest_to_embeddings_npz(
        args.manifest,
        layout.root,
        args.out_npz,
        label_col=args.label_col,
        segment_s=args.segment_s,
        overlap_s=args.overlap_s,
        embed_batch_segments=args.embed_batch_segments,
        limit_rows=args.limit_rows,
        audio_relative_to=args.audio_relative_to,
    )
    write_json(
        args.out_npz.with_name(args.out_npz.stem + ".embed_config.json"),
        {
            "cli": "cli_embed",
            "data_root": str(layout.root),
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
    )
    print(f"Wrote {args.out_npz}")


if __name__ == "__main__":
    main()
