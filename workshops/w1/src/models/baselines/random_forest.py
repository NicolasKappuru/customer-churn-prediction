"""M2 Random Forest baseline."""

import time
from functools import partial

from sklearn.ensemble import RandomForestClassifier

from ..constants import EPS, RANDOM_FOREST_MODEL_ID, SEED
from ..tunning.tunning import run_study
from .common import BaselineData, register_baseline, tune_log_loss


# Constants
RF_N_ESTIMATORS_MAX = 150
RF_MAX_DEPTH_CAP = 24
RF_MIN_SAMPLES_LEAF_MIN = 5
RF_MIN_SAMPLES_SPLIT_MIN = 10
RF_DEPTH_CHOICES = (8, 10, 12, 16, 20, RF_MAX_DEPTH_CAP)
RF_MAX_FEATURE_CHOICES = ("sqrt", "log2", 0.5)
RF_INITIAL_PARAMS = {
    "n_estimators": RF_N_ESTIMATORS_MAX,
    "max_depth": 12,
    "min_samples_leaf": RF_MIN_SAMPLES_LEAF_MIN,
    "min_samples_split": RF_MIN_SAMPLES_SPLIT_MIN,
    "max_features": "sqrt",
}


# Data preparation
def prepare_random_forest_data(data, split_data):
    """Encode all splits using only the fit phase vocabulary."""
    return BaselineData(
        fit_frame=data.frame(split_data.rows("fit"), "fit"),
        fit_labels=split_data.y("fit"),
        tune_frame=data.frame(split_data.rows("tune"), "fit"),
        tune_labels=split_data.y("tune"),
        holdout_frame=data.frame(split_data.rows("holdout"), "fit"),
    )


# Model
def make_random_forest_model(params, seed=SEED):
    """Build a forest with exactly the supplied parameters."""
    return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)


def fit_random_forest(params, frame, labels, *, seed=SEED):
    """Fit a forest without modifying its parameters."""
    return make_random_forest_model(params, seed).fit(frame, labels)


# Search space and objective
def suggest_random_forest_params(trial):
    """Sample within the forest's memory and runtime budget."""
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, RF_N_ESTIMATORS_MAX, step=50
        ),
        "max_depth": trial.suggest_categorical("max_depth", RF_DEPTH_CHOICES),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", RF_MIN_SAMPLES_LEAF_MIN, 100
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split", RF_MIN_SAMPLES_SPLIT_MIN, 100
        ),
        "max_features": trial.suggest_categorical(
            "max_features", RF_MAX_FEATURE_CHOICES
        ),
    }


def validate_rf_study(study):
    """Reject legacy trials whose recorded parameters were silently clamped."""
    for trial in study.get_trials(deepcopy=False):
        params = trial.params
        if not params:
            continue
        depth = params.get("max_depth", RF_MAX_DEPTH_CAP)
        trees = params.get("n_estimators", RF_N_ESTIMATORS_MAX)
        leaf = params.get("min_samples_leaf", RF_MIN_SAMPLES_LEAF_MIN)
        split = params.get("min_samples_split", RF_MIN_SAMPLES_SPLIT_MIN)
        if (
            depth is None
            or depth > RF_MAX_DEPTH_CAP
            or not 1 <= trees <= RF_N_ESTIMATORS_MAX
            or leaf < RF_MIN_SAMPLES_LEAF_MIN
            or split < RF_MIN_SAMPLES_SPLIT_MIN
            or params.get("max_features") == 1.0
        ):
            raise ValueError(
                f"RF trial {trial.number} contains legacy parameters. "
                "Create a new study with the rf_bounded_v2 tag; "
                "legacy scores cannot be reused with exact parameters."
            )


def random_forest_objective(trial, *, data, seed, eps):
    """Fit one forest and return its tune log-loss."""
    params = suggest_random_forest_params(trial)
    model = fit_random_forest(params, data.fit_frame, data.fit_labels, seed=seed)
    return tune_log_loss(model, data, eps)


# Registration
def register_random_forest(registry, model, data, study, counts, seconds):
    """Store M2 predictions and the exact trained parameters."""
    return register_baseline(
        registry,
        RANDOM_FOREST_MODEL_ID,
        model,
        data,
        counts,
        seconds,
        family="randomforest",
        best_params=dict(study.best_params),
    )


# Run
def run_random_forest(
    *,
    data,
    split_data,
    registry,
    study,
    total_trials,
    out_dir,
    seed=SEED,
    eps=EPS,
):
    """Coordinate M2 preparation, tuning, final fit, and registration."""
    validate_rf_study(study)
    prepared = prepare_random_forest_data(data, split_data)
    objective = partial(random_forest_objective, data=prepared, seed=seed, eps=eps)

    start = time.perf_counter()
    study, counts = run_study(
        RANDOM_FOREST_MODEL_ID,
        objective,
        study,
        total_trials,
        out_dir,
        enqueue=[dict(RF_INITIAL_PARAMS)],
    )
    model = fit_random_forest(
        study.best_params,
        prepared.fit_frame,
        prepared.fit_labels,
        seed=seed,
    )
    register_random_forest(
        registry, model, prepared, study, counts, time.perf_counter() - start
    )
    print("M2 best params:", study.best_params)
    return model
