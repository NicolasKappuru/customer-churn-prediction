"""Load KKBox tabular cohorts and split indices from the W1 export."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
W4_SRC  = PROJECT_ROOT / "workshops" / "w4" / "src"
W1_ROOT = PROJECT_ROOT / "workshops" / "w1"

for _p in (str(W4_SRC), str(W1_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import W1_EXPORT_DIR, W1_SPLIT_PATH
from src.models.feature_preparation.data import load_cohorts, load_metadata


class SplitDict(TypedDict):
    fit: npt.NDArray[np.intp]
    stop: npt.NDArray[np.intp]
    tune: npt.NDArray[np.intp]
    test: npt.NDArray[np.intp]
    development: npt.NDArray[np.intp]


def load_w1_data() -> tuple[
    pd.DataFrame,
    npt.NDArray[np.int32],
    list[str],
    list[str],
    list[str],
    SplitDict,
]:
    """Return (df, y_all, features, categorical, numeric, split) from the W1 export."""
    meta, features, categorical, numeric = load_metadata(W1_EXPORT_DIR)
    df, y_all, _ = load_cohorts(W1_EXPORT_DIR, meta, features)

    raw = np.load(W1_SPLIT_PATH, allow_pickle=False)
    split: SplitDict = {key: raw[key] for key in ("fit", "stop", "tune", "test", "development")}

    return df, y_all, features, categorical, numeric, split


def filter_normal(
    split: SplitDict,
    y_all: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Return (fit_normal, stop_normal): non-churned indices for training the AE."""
    fit_normal  = split["fit"][y_all[split["fit"]]   == 0]
    stop_normal = split["stop"][y_all[split["stop"]] == 0]
    return fit_normal, stop_normal
