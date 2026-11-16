"""Load Telco churn dataset and produce stratified split indices."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT    = Path(__file__).resolve().parents[3]
W4_TELCO_ROOT   = PROJECT_ROOT / "src" / "w4_autoencoders"

for _p in (str(W4_TELCO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import TELCO_CSV, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, SEED


class SplitDict(TypedDict):
    fit:  npt.NDArray[np.intp]
    stop: npt.NDArray[np.intp]
    test: npt.NDArray[np.intp]


def load_telco_data() -> tuple[
    pd.DataFrame,
    npt.NDArray[np.int32],
    list[str],
    list[str],
    list[str],
    SplitDict,
]:
    """Return (df, y_all, features, categorical, numeric, split) from the Telco CSV.

    Splits are stratified on the churn label: 70% fit / 15% stop / 15% test.
    df contains only the model features (ALL_FEATURES), no customerID or Churn column.
    """
    df_raw = pd.read_csv(TELCO_CSV)

    # TotalCharges is stored as string — 11 rows have a single space (zero-tenure customers)
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").fillna(0.0)

    y_all: npt.NDArray[np.int32] = (df_raw["Churn"] == "Yes").astype(np.int32).values

    df = df_raw[list(ALL_FEATURES)].reset_index(drop=True)

    features   = list(ALL_FEATURES)
    categorical = list(CATEGORICAL_FEATURES)
    numeric     = list(NUMERIC_FEATURES)

    # Stratified 70 / 15 / 15 split on churn label
    all_idx = np.arange(len(df))
    fit_idx, rest_idx, _, y_rest = train_test_split(
        all_idx, y_all,
        test_size=0.30,
        stratify=y_all,
        random_state=SEED,
    )
    stop_idx, test_idx = train_test_split(
        rest_idx,
        test_size=0.50,
        stratify=y_rest,
        random_state=SEED,
    )

    split: SplitDict = {
        "fit":  fit_idx,
        "stop": stop_idx,
        "test": test_idx,
    }

    return df, y_all, features, categorical, numeric, split


def filter_normal(
    split: SplitDict,
    y_all: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Return (fit_normal, stop_normal): non-churned indices for training the AE."""
    fit_normal  = split["fit"][y_all[split["fit"]]   == 0]
    stop_normal = split["stop"][y_all[split["stop"]] == 0]
    return fit_normal, stop_normal
