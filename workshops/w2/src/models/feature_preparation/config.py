"""Run configuration for the w2 feature-preparation stage.

The dataset contract lives in the export's ``metadata.json`` and is loaded by
:mod:`data`. This module follows the w1 pattern: it only holds the run options
and resolves the input/output paths.
"""

from pathlib import Path

from ..constants import (
    BLOCK_ROWS,
    DATASET_VERSION,
    DEV_COHORT,
    EXPECTED_CUTOFFS,
    N_DAYS,
    PHASES,
    QUANTILE_SAMPLE,
    SEED,
    SPLIT_HASH_SALT,
    SPLIT_PROTOCOL,
    SPLIT_SHARES,
    TEST_COHORT,
)


def run_config():
    """Return the feature-preparation options."""
    return {
        "seed": SEED,
        "dataset_version": DATASET_VERSION,
        "dev_cohort": DEV_COHORT,
        "test_cohort": TEST_COHORT,
        "expected_cutoffs": dict(EXPECTED_CUTOFFS),
        "n_days": N_DAYS,
        "split_protocol": SPLIT_PROTOCOL,
        "split_shares": dict(SPLIT_SHARES),
        "split_hash_salt": SPLIT_HASH_SALT,
        "quantile_sample": QUANTILE_SAMPLE,
        "block_rows": BLOCK_ROWS,
        "phases": list(PHASES),
        "expected_full_sizes": {
            "fit": 744_699,
            "stop": 124_117,
            "tune": 124_115,
            "test": 970_960,
        },
    }


def run_paths(input_dir=None, out_dir=None):
    """Resolve the sequence export and output directory, locally or on Kaggle."""
    on_kaggle = Path("/kaggle/input").exists()

    if input_dir is not None:
        source = Path(input_dir)
    elif on_kaggle:
        source = Path("/kaggle/input/dataset-final")
    else:
        source = Path("export-sequential-v5")

    if out_dir is not None:
        target = Path(out_dir)
    else:
        base = Path("/kaggle/working") if on_kaggle else Path("artifacts")
        target = base / "sherlock_w2" / "feature_preparation"

    target.mkdir(parents=True, exist_ok=True)
    return on_kaggle, source, target
