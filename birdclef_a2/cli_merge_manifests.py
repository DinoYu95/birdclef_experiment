"""CLI: concatenate CSV manifests (same columns)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()

    dfs = [pd.read_csv(x) for x in args.inputs]
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(merged)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
