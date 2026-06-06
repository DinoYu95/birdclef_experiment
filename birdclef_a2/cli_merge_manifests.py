"""CLI: concatenate CSV manifests (same columns)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from birdclef_a2.manifest_utils import normalize_manifest_paths_for_data_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument(
        "--for-data-root",
        action="store_true",
        help="Prefix real train paths with train_audio/ so merged CSV works with "
        "cli_embed/cli_torch_train --audio-relative-to data_root alongside "
        "synthetic_train_audio/... rows.",
    )
    args = p.parse_args()

    dfs = [pd.read_csv(x) for x in args.inputs]
    if args.for_data_root:
        dfs = [normalize_manifest_paths_for_data_root(df) for df in dfs]
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(merged)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
