"""Merge multiple val_metrics.json into one CSV for report tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Concatenate val_metrics.json from several runs.")
    p.add_argument(
        "--rows",
        nargs="+",
        metavar="NAME=DIR",
        help='e.g. A_logreg=outputs/A_sklearn B_torch=outputs/B_torch_scratch',
    )
    p.add_argument("--out-csv", type=Path, default=Path("outputs/metrics_comparison.csv"))
    args = p.parse_args()

    records: list[dict] = []
    for item in args.rows:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got: {item}")
        name, d = item.split("=", 1)
        metrics_path = Path(d) / "val_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"run_name": name.strip(), **data}
        exp = Path(d) / "experiment_config.json"
        if exp.is_file():
            row["experiment_config_path"] = str(exp)
        records.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
