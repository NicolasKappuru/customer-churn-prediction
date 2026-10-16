"""M3 XGBoost baseline."""

import time
from functools import partial

import numpy as np

from ..constants import EPS, SEED, XGBOOST_MODEL_ID
from ..tunning.tunning import run_study
from .common import BaselineData, register_baseline, tune_log_loss


# Constants
TREE_LOG_PERIOD = 100
TREE_MAX_ROUNDS_FULL = 3000
TREE_MAX_ROUNDS_SMOKE = 50
TREE_EARLY_STOP_FULL = 100
TREE_EARLY_STOP_SMOKE = 20
XGB_BALANCED_WEIGHT = True
XGB_INITIAL_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 1e-4,
    "reg_lambda": 1.0,
    "gamma": 1e-4,
}


# Data preparation
def prepare_xgboost_data(tree_frame, split_data):
    """Build categorical frames using the fit phase preprocessing state."""
    return BaselineData(
        fit_frame=tree_frame(split_data.rows("fit"), "fit"),
        fit_labels=split_data.y("fit"),
        tune_frame=tree_frame(split_data.rows("tune"), "fit"),
        tune_labels=split_data.y("tune"),
        holdout_frame=tree_frame(split_data.rows("holdout"), "fit"),
        stop_frame=tree_frame(split_data.rows("stop"), "fit"),
        stop_labels=split_data.y("stop"),
    )


def balanced_scale_pos_weight(labels, balanced=XGB_BALANCED_WEIGHT):
    """Compute the class weight from the supplied training labels."""
    labels = np.asarray(labels)
    positives = max(int((labels == 1).sum()), 1)
    negatives = int((labels == 0).sum())
    return negatives / positives if balanced else 1.0


# Model
def make_xgboost_model(
    params,
    labels,
    *,
    n_estimators,
    device,
    early_stop=None,
    seed=SEED,
    balanced=XGB_BALANCED_WEIGHT,
):
    """Build a categorical classifier with optional early stopping."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        device=device,
        enable_categorical=True,
        eval_metric="logloss",
        early_stopping_rounds=early_stop,
        scale_pos_weight=balanced_scale_pos_weight(labels, balanced),
        random_state=seed,
        n_jobs=-1,
        **params,
    )


def fit_xgboost(
    params,
    train_frame,
    train_labels,
    stop_frame,
    stop_labels,
    *,
    device,
    max_rounds=TREE_MAX_ROUNDS_FULL,
    early_stop=TREE_EARLY_STOP_FULL,
    log_period=TREE_LOG_PERIOD,
    seed=SEED,
    balanced=XGB_BALANCED_WEIGHT,
):
    """Fit one candidate with early stopping on the stop split."""
    model = make_xgboost_model(
        params,
        train_labels,
        n_estimators=max_rounds,
        device=device,
        early_stop=early_stop,
        seed=seed,
        balanced=balanced,
    )
    return model.fit(
        train_frame,
        train_labels,
        eval_set=[(stop_frame, stop_labels)],
        verbose=log_period,
    )


def refit_xgboost(
    params,
    frame,
    labels,
    *,
    n_estimators,
    device,
    seed=SEED,
    balanced=XGB_BALANCED_WEIGHT,
):
    """Fit a fixed number of trees on the supplied development data."""
    model = make_xgboost_model(
        params,
        labels,
        n_estimators=max(1, int(n_estimators)),
        device=device,
        seed=seed,
        balanced=balanced,
    )
    return model.fit(frame, labels, verbose=False)


# Search space and objective
def suggest_xgboost_params(trial):
    """Sample tree shape, learning rate, and regularization."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    }


def xgboost_objective(trial, *, data, fit_options, eps):
    """Fit one candidate, record its stopping iteration, and score tune."""
    params = suggest_xgboost_params(trial)
    model = fit_xgboost(
        params,
        data.fit_frame,
        data.fit_labels,
        data.stop_frame,
        data.stop_labels,
        **fit_options,
    )
    trial.set_user_attr(
        "best_iteration",
        int(getattr(model, "best_iteration", fit_options["max_rounds"])),
    )
    return tune_log_loss(model, data, eps)


# Registration
def register_xgboost(
    registry, model, data, study, counts, seconds, *, max_rounds, balanced
):
    """Store M3 predictions, stopping iteration, and class weight."""
    return register_baseline(
        registry,
        XGBOOST_MODEL_ID,
        model,
        data,
        counts,
        seconds,
        family="xgboost",
        best_params=dict(study.best_params),
        best_iteration=int(getattr(model, "best_iteration", max_rounds)),
        scale_pos_weight=balanced_scale_pos_weight(data.fit_labels, balanced),
    )


# Run
def run_xgboost(
    *,
    tree_frame,
    split_data,
    registry,
    study,
    total_trials,
    out_dir,
    device,
    max_rounds,
    early_stop,
    log_period,
    seed=SEED,
    balanced=XGB_BALANCED_WEIGHT,
    eps=EPS,
):
    """Coordinate M3 preparation, tuning, final fit, and registration."""
    data = prepare_xgboost_data(tree_frame, split_data)
    fit_options = {
        "device": device,
        "max_rounds": max_rounds,
        "early_stop": early_stop,
        "log_period": log_period,
        "seed": seed,
        "balanced": balanced,
    }
    objective = partial(
        xgboost_objective, data=data, fit_options=fit_options, eps=eps
    )

    start = time.perf_counter()
    study, counts = run_study(
        XGBOOST_MODEL_ID,
        objective,
        study,
        total_trials,
        out_dir,
        enqueue=[dict(XGB_INITIAL_PARAMS)],
    )
    model = fit_xgboost(
        study.best_params,
        data.fit_frame,
        data.fit_labels,
        data.stop_frame,
        data.stop_labels,
        **fit_options,
    )
    record = register_xgboost(
        registry, model, data, study, counts, time.perf_counter() - start,
        max_rounds=max_rounds,
        balanced=balanced,
    )
    print(
        "M3 best iteration:", record["best_iteration"],
        "| scale_pos_weight:", record["scale_pos_weight"],
    )
    return model
