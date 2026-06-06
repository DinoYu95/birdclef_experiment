"""CLI: merge chunk NPZs into one train/val-compatible embedding file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from birdclef_a2.embed_merge import merge_embedding_npz_files

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Merge multiple embedding .npz (cli_embed chunks) "
        "into one NPZ for cli_sklearn_train."
    )
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Chunk NPZs in order (same segment_s/overlap_s/birdnet backend as embeddings).",
    )
    p.add_argument("--out-npz", type=Path, required=True)
    args = p.parse_args()

    merge_embedding_npz_files(args.out_npz, list(args.inputs))
    print(f"Wrote merged {args.out_npz}")


if __name__ == "__main__":
    main()
