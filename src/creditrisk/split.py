"""Development / out-of-time split.

Development vintages are split stratified on (vintage, target) into train / validation / test.
Out-of-time (OOT) vintages are held back entirely. Tuning uses validation only; test and OOT
are scored once, by `creditrisk.cli evaluate`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from creditrisk.features.catalog import TARGET, VINTAGE

SPLITS = ["train", "validation", "test", "oot"]


def assign_splits(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    s, d = cfg["split"], cfg["data"]
    if not np.isclose(s["train"] + s["validation"] + s["test"], 1.0):
        raise ValueError("split shares must sum to 1")
    split = pd.Series(pd.NA, index=df.index, dtype="object")
    split[df[VINTAGE].isin(d["vintages_oot"])] = "oot"

    dev = df[df[VINTAGE].isin(d["vintages_dev"])]
    strata = dev[VINTAGE].astype(str) + "_" + dev[TARGET].astype(str)
    train_idx, rest_idx = train_test_split(
        dev.index, train_size=s["train"], stratify=strata, random_state=s["seed"]
    )
    val_share = s["validation"] / (s["validation"] + s["test"])
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_share, stratify=strata.loc[rest_idx], random_state=s["seed"]
    )
    split[train_idx] = "train"
    split[val_idx] = "validation"
    split[test_idx] = "test"
    return split
