from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    singleton_labels: frozenset
    n_singleton_rows: int


def stratified_train_val_with_singletons_forced_to_val(
    df: pd.DataFrame,
    *,
    label_col: str,
    val_fraction: float,
    seed: int,
    group_col: str | None = None,
) -> SplitResult:
    """
    - Any class with exactly one row in the global table must appear only in validation
      (assignment rule).
    - Remaining rows: stratified split with fraction `val_fraction` to validation; overall
      80/20 may differ slightly because singleton rows are forced into val.
    - Optional `group_col`: keeps each whole group in train xor val (no leakage). Omit if
      the column does not exist.
    """
    if label_col not in df.columns:
        raise KeyError(
            f"Missing label column {label_col!r}; available columns: {list(df.columns)}"
        )
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")

    vc = df[label_col].value_counts()
    singleton_labels = set(vc[vc == 1].index.astype(str))

    singleton_mask = df[label_col].astype(str).isin(singleton_labels)
    singleton_rows = df[singleton_mask]
    pool = df[~singleton_mask].copy()

    if pool.empty:
        raise ValueError("No rows left after reserving singleton-only classes for val.")

    if group_col is not None:
        if group_col not in pool.columns:
            raise KeyError(f"group_col={group_col!r} is not in the DataFrame")
        pool = pool.copy()
        pool["_group_split_key"] = pool[group_col].astype(str)
        grp_first = pool.drop_duplicates(subset=["_group_split_key"], keep="first")
        grp_labels = grp_first[label_col].astype(str)

        gid_train_idx, gid_val_idx = train_test_split(
            np.arange(len(grp_first)),
            test_size=val_fraction,
            stratify=grp_labels,
            random_state=seed,
        )
        val_groups = set(grp_first.iloc[gid_val_idx]["_group_split_key"])
        train_groups = set(grp_first.iloc[gid_train_idx]["_group_split_key"])
        if val_groups & train_groups:
            raise RuntimeError("Train/val group split overlap; this should never happen.")

        val_from_pool = pool[pool["_group_split_key"].isin(val_groups)]
        train_from_pool = pool[pool["_group_split_key"].isin(train_groups)]
        train_from_pool = train_from_pool.drop(columns=["_group_split_key"])
        val_from_pool = val_from_pool.drop(columns=["_group_split_key"])
    else:
        train_from_pool, val_from_pool = train_test_split(
            pool,
            test_size=val_fraction,
            stratify=pool[label_col],
            random_state=seed,
        )

    val = pd.concat([singleton_rows, val_from_pool], axis=0)
    train = train_from_pool

    # Sanity check: singleton classes must never appear in train.
    if not singleton_labels.isdisjoint(set(train[label_col].astype(str).unique())):
        bad = singleton_labels & set(train[label_col].astype(str).unique())
        raise AssertionError(f"Singleton class(es) leaked into train: {bad}")

    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)

    return SplitResult(
        train=train,
        val=val,
        singleton_labels=frozenset(singleton_labels),
        n_singleton_rows=int(len(singleton_rows)),
    )
