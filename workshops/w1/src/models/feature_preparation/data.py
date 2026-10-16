"""Tabular v5 data loading.

Reads the export produced by the preprocessing notebook, checks its contract
(version, cutoffs, feature catalogue) and verifies each parquet's ``msno`` order
against its published hash. The two labelled cohorts are stacked into one frame
(February rows first, then March) with ``cohort`` and ``row`` columns.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import DATASET_VERSION, DEV_COHORT, EXPECTED_CUTOFFS, HOLD_COHORT


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def load_metadata(input_dir, expected_cutoffs=EXPECTED_CUTOFFS):
    """Check the v5 contract and return ``(meta, features, categorical, numeric)``."""
    meta = json.loads((Path(input_dir) / "metadata.json").read_text())
    if meta.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"Expected dataset_version {DATASET_VERSION}, "
            f"got {meta.get('dataset_version')!r}."
        )
    if meta["cutoffs"] != expected_cutoffs:
        raise ValueError(f"Unexpected cutoffs: {meta['cutoffs']}")

    features = meta["features"]
    categorical = meta["categorical"]
    # The identifier and the label must never leak into the design matrix.
    if {"msno", "is_churn"}.intersection(features):
        raise ValueError("Identifier or label appears in features")

    numeric = [column for column in features if column not in categorical]
    return meta, features, categorical, numeric


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _order_hash(msno_series):
    """Recompute the export's published order hash for one ``msno`` column."""
    digest = hashlib.sha1()
    for value in msno_series:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_cohorts(
    input_dir,
    meta,
    features,
    dev_cohort=DEV_COHORT,
    hold_cohort=HOLD_COHORT,
):
    """Load and verify cohorts and test; return ``(train_df, test_df, y_all, cohort_labels)``."""
    input_dir = Path(input_dir)
    cohort_frames = {}

    # Read each labelled cohort, keeping the order the export wrote and
    # confirming it with the published hash before adding the cohort tag.
    for cohort in (dev_cohort, hold_cohort):
        frame = pd.read_parquet(
            input_dir / f"{cohort}.parquet",
            columns=["msno", "is_churn"] + features,
        ).reset_index(drop=True)
        assert frame["msno"].is_unique, f"{cohort} has duplicated msno"
        assert _order_hash(frame["msno"]) == meta[f"hash_{cohort}"], (
            f"{cohort} parquet misaligned"
        )
        frame.insert(1, "cohort", cohort)
        cohort_frames[cohort] = frame

    # Stack development first, then holdout; ``row`` is the shared index every
    # split below is expressed in, and ``cohort`` stays recoverable as text.
    train_df = pd.concat(
        [cohort_frames[dev_cohort], cohort_frames[hold_cohort]],
        ignore_index=True,
    )
    train_df["cohort"] = train_df["cohort"].astype("category")
    train_df["row"] = np.arange(len(train_df))
    cohort_labels = train_df["cohort"].to_numpy(dtype=str)

    # Test carries features only (no labels) and its own row order.
    test_df = pd.read_parquet(
        input_dir / "test.parquet",
        columns=["msno"] + features,
    ).reset_index(drop=True)
    assert test_df["msno"].is_unique
    assert _order_hash(test_df["msno"]) == meta["hash_test"], (
        "test parquet misaligned"
    )
    test_df["row"] = np.arange(len(test_df))

    y_all = train_df["is_churn"].to_numpy(dtype=np.int32)
    assert set(np.unique(y_all)) == {0, 1}

    # Per-cohort diagnostics; a user may appear in both months with a different
    # label, which is the production setting (scored every month).
    for cohort in (dev_cohort, hold_cohort):
        mask = cohort_labels == cohort
        print(
            f"{cohort:9s} rows: {mask.sum():>9,}"
            f" | label month: {meta['label_months'][cohort]}"
            f" | churn rate: {y_all[mask].mean():.4f}"
        )
    print(
        "stacked train:", train_df.shape,
        "| test:", test_df.shape,
        "| users in both labelled months:",
        int(train_df["msno"].duplicated(keep=False).sum() // 2),
    )

    return train_df, test_df, y_all, cohort_labels
