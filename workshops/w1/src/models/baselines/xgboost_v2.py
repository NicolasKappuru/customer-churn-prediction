"""M3v2 XGBoost: AP objective, expanded search space, isotonic calibration."""

import json
import time
from functools import partial

import numpy as np
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from ..constants import EPS, SEED, XGB_V2_MODEL_ID
from ..tunning.tunning import (
    contract_hash,
    count_states,
    remaining_trials,
    save_study_artifacts,
)
from .xgboost_model import balanced_scale_pos_weight, prepare_xgboost_data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_M3_V5_BEST_BASE = {
    "learning_rate": 0.02869950628708393,
    "grow_policy": "depthwise",
    "max_depth": 10,
    "min_child_weight": 5.718451902005254,
    "gamma": 4.156526807771914,
    "subsample": 0.9002101873610451,
    "colsample_bytree": 0.6097502366481172,
    "reg_alpha": 0.39257213672988855,
    "reg_lambda": 0.04712632932766597,
}


def _initial_enqueue_params(fit_labels):
    """Two warmup trials: M3v5 best params with original SPW and with SPW=1."""
    spw = balanced_scale_pos_weight(fit_labels, balanced=True, factor=2.0)
    return [
        {**_M3_V5_BEST_BASE, "scale_pos_weight": float(spw)},
        {**_M3_V5_BEST_BASE, "scale_pos_weight": 1.0},
    ]


# ---------------------------------------------------------------------------
# Model factory (scale_pos_weight is part of params, not set separately)
# ---------------------------------------------------------------------------

def _make_model_v2(params, *, n_estimators, device, early_stop=None, seed=SEED):
    """XGBClassifier where scale_pos_weight and all new params live in params."""
    return XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        device=device,
        eval_metric="aucpr",
        early_stopping_rounds=early_stop,
        random_state=seed,
        n_jobs=-1,
        **params,
    )


def _fit_v2(params, train_frame, train_labels, stop_frame, stop_labels, *,
            device, max_rounds, early_stop, log_period, seed=SEED):
    """Fit one M3v2 candidate with early stopping on the stop split."""
    model = _make_model_v2(
        params, n_estimators=max_rounds, device=device,
        early_stop=early_stop, seed=seed,
    )
    return model.fit(
        train_frame, train_labels,
        eval_set=[(stop_frame, stop_labels)],
        verbose=log_period,
    )


def _complete_params(params):
    """Add implicitly-set conditional params omitted from study.best_params."""
    p = dict(params)
    if p.get("grow_policy") == "depthwise":
        p.setdefault("max_leaves", 0)
    elif p.get("grow_policy") == "lossguide":
        p.setdefault("max_depth", 0)
    return p


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def suggest_xgboost_v2_params(trial):
    """Expanded search space for M3v2."""
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if grow_policy == "depthwise":
        max_depth = trial.suggest_int("max_depth", 4, 14)
        max_leaves = 0
    else:
        max_depth = 0
        max_leaves = trial.suggest_int("max_leaves", 31, 1024, log=True)

    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "grow_policy": grow_policy,
        "max_depth": max_depth,
        "max_leaves": max_leaves,
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 30, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
    }


# ---------------------------------------------------------------------------
# Objective (maximise average precision on tune)
# ---------------------------------------------------------------------------

def xgboost_v2_objective(trial, *, data, fit_options):
    """Fit one candidate; return average precision on tune (maximise)."""
    params = suggest_xgboost_v2_params(trial)
    model = _fit_v2(
        params,
        data.fit_frame, data.fit_labels,
        data.stop_frame, data.stop_labels,
        **fit_options,
    )
    best_iter = int(getattr(model, "best_iteration", fit_options["max_rounds"]))
    proba = model.predict_proba(data.tune_frame)[:, 1]
    ap = float(average_precision_score(data.tune_labels, proba))

    trial.set_user_attr("best_iteration", best_iter)
    trial.set_user_attr("tune_roc_auc", float(roc_auc_score(data.tune_labels, proba)))
    trial.set_user_attr("tune_log_loss", float(
        log_loss(data.tune_labels, np.clip(proba, EPS, 1 - EPS), labels=[0, 1])
    ))
    return ap


# ---------------------------------------------------------------------------
# Study (direction=maximize)
# ---------------------------------------------------------------------------

def get_study_v2(model_id, contract, storage, seed, n_trials, tag="M3v2"):
    """Create or resume an Optuna study that maximises average precision."""
    name = f"{model_id}_{contract_hash(contract)}"
    if tag:
        name += f"_{tag}"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=max(1, min(5, n_trials // 4)),
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    if study.user_attrs.get("contract") != contract:
        study.set_user_attr("contract", contract)
    return study


def _run_study_v2(model_id, objective, study, total_trials, out_dir, enqueue=None):
    """Resume the budget, save artifacts, and return (study, counts)."""
    if enqueue:
        existing = {json.dumps(t.params, sort_keys=True) for t in study.trials}
        for params in enqueue:
            key = json.dumps(params, sort_keys=True)
            if key not in existing:
                study.enqueue_trial(params, skip_if_exists=True)
                existing.add(key)

    n_remaining = remaining_trials(study, total_trials)
    print(f"{model_id}: budget={total_trials}, remaining={n_remaining}", flush=True)

    def wrapped(trial):
        print(f"{model_id}: starting trial #{trial.number}", flush=True)
        return objective(trial)

    def on_end(_study, trial):
        secs = trial.duration.total_seconds() if trial.duration else 0.0
        print(
            f"{model_id}: trial #{trial.number} {trial.state.name}"
            f" | ap={trial.value:.5f} | {secs:.1f}s",
            flush=True,
        )

    if n_remaining > 0:
        study.optimize(
            wrapped,
            n_trials=n_remaining,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[on_end],
        )

    counts = count_states(study)
    if counts["complete"] == 0:
        raise RuntimeError(f"{model_id}: no completed trials.")

    save_study_artifacts(study, model_id, out_dir)
    print(f"{model_id}: best tune AP={study.best_value:.5f} | trials {counts}")
    return study, counts


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_isotonic_calibrator(proba, y_true):
    """Fit IsotonicRegression on tune split probabilities."""
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(np.asarray(proba), np.asarray(y_true))
    return cal


def apply_calibration(calibrator, proba):
    """Apply a fitted isotonic calibrator."""
    return calibrator.predict(np.asarray(proba))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_xgboost_v2(
    *,
    tree_frame,
    split_data,
    study,
    total_trials,
    out_dir,
    device,
    max_rounds=3000,
    early_stop=100,
    log_period=100,
    seed=SEED,
):
    """Tune M3v2, refit best model on fit, calibrate on tune.

    Returns (model, calibrator, data, study, counts).
    data.tune_frame / data.tune_labels are available for threshold selection.
    """
    data = prepare_xgboost_data(tree_frame, split_data)
    fit_options = {
        "device": device,
        "max_rounds": max_rounds,
        "early_stop": early_stop,
        "log_period": log_period,
        "seed": seed,
    }

    enqueue = _initial_enqueue_params(data.fit_labels)
    objective = partial(xgboost_v2_objective, data=data, fit_options=fit_options)

    start = time.perf_counter()
    study, counts = _run_study_v2(
        XGB_V2_MODEL_ID, objective, study, total_trials, out_dir, enqueue=enqueue,
    )

    best_params = _complete_params(study.best_params)
    model = _fit_v2(
        best_params,
        data.fit_frame, data.fit_labels,
        data.stop_frame, data.stop_labels,
        **fit_options,
    )
    elapsed = time.perf_counter() - start

    proba_tune = model.predict_proba(data.tune_frame)[:, 1]
    calibrator = fit_isotonic_calibrator(proba_tune, data.tune_labels)

    best_iter = int(getattr(model, "best_iteration", max_rounds))
    print(f"M3v2 best_iteration: {best_iter} | elapsed: {elapsed:.0f}s")
    return model, calibrator, data, study, counts
