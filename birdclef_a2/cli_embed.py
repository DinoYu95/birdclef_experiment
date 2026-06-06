"""CLI: BirdNET embeddings -> compressed NPZ (X, y, filename)."""
from __future__ import annotations

import os

# Apply before any import that may pull TensorFlow (oneDNN port.cc / absl spam on STDERR).
# Pre-set env wins over setdefault; ``--verbose-tf-logging`` upgrades to noisy in main().
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

import argparse
import logging
from pathlib import Path

import pandas as pd

from birdclef_a2.birdnet_embed import (
    birdnet_backend_and_session_device,
    birdnet_inference_device,
    manifest_to_embeddings_npz,
)
from birdclef_a2.config import load_layout
from birdclef_a2.embed_merge import merge_embedding_npz_files
from birdclef_a2.manifest_utils import assert_manifest_columns
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
    p.add_argument(
        "--embed-n-workers",
        default=None,
        metavar="N|auto",
        help="BirdNET encode_arrays n_workers (default: env BIRDCLEF_EMBED_N_WORKERS=1). "
        "Use 'auto' for birdnet default (~physical CPU cores). Higher => more CPU/RAM.",
    )
    p.add_argument(
        "--embed-n-producers",
        type=int,
        default=None,
        help="BirdNET encode_arrays n_producers (default: env BIRDCLEF_EMBED_N_PRODUCERS=1).",
    )
    p.add_argument(
        "--embed-prefetch-ratio",
        type=int,
        default=None,
        help="BirdNET encode_arrays prefetch_ratio (default: env BIRDCLEF_EMBED_PREFETCH_RATIO=1).",
    )
    p.add_argument(
        "--offset-rows",
        type=int,
        default=0,
        help="Skip this many manifest rows before embedding (chunked CPU runs).",
    )
    p.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Caps how many manifest rows after --offset-rows to embed. "
        "Single run: that many rows only. With --chunk-rows: total span to cover in batches. "
        "Omit = through end of CSV.",
    )
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        metavar="N",
        help="Process the manifest range in passes of <=N manifest rows each, merge into --out-npz (one CLI). "
        "Uses chunk dir for temporary part *.npz; final result equals one full embedding run",
    )
    p.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="Directory for intermediate part NPZs (--chunk-rows). Default: <out-npz parent>/<stem>_parts/",
    )
    p.add_argument(
        "--chunk-keep-parts",
        action="store_true",
        help="After merge (--chunk-rows), do not delete intermediate part *.npz files.",
    )
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
    p.add_argument(
        "--verbose-tf-logging",
        action="store_true",
        help="Verbose TensorFlow C++ STDERR (default: TF_CPP_MIN_LOG_LEVEL=3 via module init).",
    )
    args = p.parse_args()

    if args.birdnet_device is not None:
        os.environ["BIRDCLEF_BIRDNET_DEVICE"] = args.birdnet_device
    if args.birdnet_force_tf_cpu:
        os.environ["BIRDCLEF_BIRDNET_FORCE_TF_CPU"] = "1"
    if args.embed_n_workers is not None:
        os.environ["BIRDCLEF_EMBED_N_WORKERS"] = str(args.embed_n_workers).strip()
    if args.embed_n_producers is not None:
        os.environ["BIRDCLEF_EMBED_N_PRODUCERS"] = str(args.embed_n_producers)
    if args.embed_prefetch_ratio is not None:
        os.environ["BIRDCLEF_EMBED_PREFETCH_RATIO"] = str(args.embed_prefetch_ratio)
    if args.verbose_tf_logging:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        os.environ["GLOG_minloglevel"] = "0"

    bk, sess = birdnet_backend_and_session_device()
    logging.getLogger(__name__).info(
        "BirdNET backend=%s session_device=%s (BIRDCLEF_BIRDNET_DEVICE=%r)",
        bk,
        sess,
        birdnet_inference_device(),
    )

    layout = load_layout(args.data_root)

    if args.chunk_rows is not None:
        if args.chunk_rows < 1:
            raise ValueError("--chunk-rows must be >= 1")

        df = pd.read_csv(args.manifest)
        assert_manifest_columns(df)
        if args.label_col not in df.columns:
            raise KeyError(f"missing {args.label_col}")

        n_manifest = len(df)
        start = int(args.offset_rows)
        if start < 0:
            raise ValueError("--offset-rows must be >= 0")
        if start >= n_manifest:
            raise ValueError(f"--offset-rows={start} >= manifest rows ({n_manifest})")
        stop = n_manifest if args.limit_rows is None else min(n_manifest, start + int(args.limit_rows))
        chunk_dir = args.chunk_dir if args.chunk_dir is not None else args.out_npz.with_name(
            args.out_npz.stem + "_parts"
        )
        chunk_dir.mkdir(parents=True, exist_ok=True)

        part_paths: list[Path] = []
        chunk_specs: list[dict[str, int | str]] = []
        for offs in range(start, stop, args.chunk_rows):
            lim = min(args.chunk_rows, stop - offs)
            part = chunk_dir / f"{args.out_npz.stem}_chunk_{offs:06d}_{offs + lim:06d}.npz"
            manifest_to_embeddings_npz(
                args.manifest,
                layout.root,
                part,
                label_col=args.label_col,
                segment_s=args.segment_s,
                overlap_s=args.overlap_s,
                embed_batch_segments=args.embed_batch_segments,
                offset_rows=offs,
                limit_rows=lim,
                audio_relative_to=args.audio_relative_to,
                manifest_df=df,
            )
            part_paths.append(part)
            chunk_specs.append(
                {"offset_rows": offs, "limit_rows": lim, "part_npz": str(part.resolve())}
            )

        merge_embedding_npz_files(args.out_npz, part_paths)
        log = logging.getLogger(__name__)
        if args.chunk_keep_parts:
            log.info("Keeping %s intermediate part NPZs under %s", len(part_paths), chunk_dir)
        else:
            for p in part_paths:
                try:
                    p.unlink(missing_ok=True)
                except OSError as exc:  # pragma: no cover
                    log.warning("Could not delete part %s: %s", p, exc)
            if args.chunk_dir is None:
                try:
                    chunk_dir.rmdir()
                except OSError:
                    log.info("Parts directory not removed (may not be empty): %s", chunk_dir)

        cfg = {
            "cli": "cli_embed",
            "data_root": str(layout.root),
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "chunk_specs": chunk_specs,
            "merged_out_npz": str(args.out_npz),
            "manifest_row_span": {"start": start, "stop": stop, "n_manifest": n_manifest},
        }
    else:
        manifest_to_embeddings_npz(
            args.manifest,
            layout.root,
            args.out_npz,
            label_col=args.label_col,
            segment_s=args.segment_s,
            overlap_s=args.overlap_s,
            embed_batch_segments=args.embed_batch_segments,
            offset_rows=args.offset_rows,
            limit_rows=args.limit_rows,
            audio_relative_to=args.audio_relative_to,
        )
        cfg = {
            "cli": "cli_embed",
            "data_root": str(layout.root),
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }

    write_json(args.out_npz.with_name(args.out_npz.stem + ".embed_config.json"), cfg)
    print(f"Wrote {args.out_npz}")


if __name__ == "__main__":
    main()
