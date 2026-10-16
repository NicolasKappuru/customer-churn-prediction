"""Run configuration.

Model budgets/bounds (:func:`run_config`) and input/output paths
(:func:`run_paths`), each picking smoke or full values. Smoke runs exercise
every code path quickly and write to a separate folder.
"""

from pathlib import Path

from ..baselines.random_forest import (
    RF_MAX_DEPTH_CAP,
    RF_MIN_SAMPLES_LEAF_MIN,
    RF_MIN_SAMPLES_SPLIT_MIN,
    RF_N_ESTIMATORS_MAX,
)
from ..baselines.xgboost_model import (
    TREE_EARLY_STOP_FULL,
    TREE_EARLY_STOP_SMOKE,
    TREE_LOG_PERIOD,
    TREE_MAX_ROUNDS_FULL,
    TREE_MAX_ROUNDS_SMOKE,
    XGB_BALANCED_WEIGHT,
)
from ..constants import (
    DEV_COHORT,
    EXPECTED_CUTOFFS,
    TEST_COHORT,
    OPTUNA_MAX_TRIALS,
    SEED,
    SMOKE_USERS,
    TRAIN_VERBOSE,
    TRIALS_FULL,
    TRIALS_SMOKE,
)


# ---------------------------------------------------------------------------
# Budgets and bounds
# ---------------------------------------------------------------------------

def run_config(smoke_test=False):
    """Return the model budgets and safety bounds for a run."""
    # The per-family budgets must fit inside the global Optuna budget.
    assert sum(TRIALS_FULL.values()) <= OPTUNA_MAX_TRIALS
    return {
        "seed": SEED,
        "optuna_max_trials": OPTUNA_MAX_TRIALS,
        "trials": dict(TRIALS_SMOKE if smoke_test else TRIALS_FULL),
        "train_verbose": TRAIN_VERBOSE,
        "tree_log_period": TREE_LOG_PERIOD,
        "xgb_balanced_weight": XGB_BALANCED_WEIGHT,
        "tree_max_rounds": (
            TREE_MAX_ROUNDS_SMOKE if smoke_test else TREE_MAX_ROUNDS_FULL
        ),
        "tree_early_stop": (
            TREE_EARLY_STOP_SMOKE if smoke_test else TREE_EARLY_STOP_FULL
        ),
        "smoke_users": SMOKE_USERS if smoke_test else None,
        "rf_n_estimators_max": RF_N_ESTIMATORS_MAX,
        "rf_max_depth_cap": RF_MAX_DEPTH_CAP,
        "rf_min_samples_leaf_min": RF_MIN_SAMPLES_LEAF_MIN,
        "rf_min_samples_split_min": RF_MIN_SAMPLES_SPLIT_MIN,
    }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def run_paths(smoke_test=False, input_dir=None, out_dir=None):
    """Resolve the input/output paths, on Kaggle or locally (with overrides).

    ``smoke_test`` picks the smoke output folder so smoke artifacts never mix
    with full results.
    """
    on_kaggle = Path("/kaggle/input").exists()

    if input_dir is not None:
        source = Path(input_dir)
    elif on_kaggle:
        source = Path("/kaggle/input/dataset-final")
    else:
        source = Path("export-tabular-v5")

    base = Path("/kaggle/working") if on_kaggle else Path("artifacts")
    folder = "sherlock_w1_smoke" if smoke_test else "sherlock_w1"
    target = Path(out_dir) if out_dir is not None else base / folder
    target.mkdir(parents=True, exist_ok=True)

    return on_kaggle, source, target
