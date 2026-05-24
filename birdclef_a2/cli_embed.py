"""CLI: BirdNET embeddings -> compressed NPZ (X, y, filename)."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from birdclef_a2.birdnet_embed import (
    birdnet_backend_and_session_device,
    birdnet_inference_device,
    manifest_to_embeddings_npz,
)
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
    p.add_argument(
        "--birdnet-device",
        default=None,
        metavar="DEVICE",
        help="BirdNET inference device: CPU, GPU:0, etc. GPU maps to protobuf backend (CUDA TF); "
        "CPU uses tf-lite path. If omitted, uses env BIRDCLEF_BIRDNET_DEVICE or CPU.",
    )
    p.add_argument(
        "--birdnet-force-tf-cpu",
        action="store_true",
        help="Use backend=tf (CPU-only) even if --birdnet-device GPU:0 — work around CUDA/cuDNN "
        "errors like 'No DNN support for stream' on misconfigured renters.",
    )
    args = p.parse_args()

    if args.birdnet_device is not None:
        os.environ["BIRDCLEF_BIRDNET_DEVICE"] = args.birdnet_device
    if args.birdnet_force_tf_cpu:
        os.environ["BIRDCLEF_BIRDNET_FORCE_TF_CPU"] = "1"

    bk, sess = birdnet_backend_and_session_device()
    logging.getLogger(__name__).info(
        "BirdNET backend=%s session_device=%s (BIRDCLEF_BIRDNET_DEVICE=%r)",
        bk,
        sess,
        birdnet_inference_device(),
    )

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
