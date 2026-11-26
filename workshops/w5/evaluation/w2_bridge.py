"""Bridge between the w5 synthetic users and the w2 churn classifiers.

Question: do synthetic users help the w2 LSTM/GRU predict churn in March?

The w2 data object (``PreparedSequenceData``) is extended with the synthetic
rows: they are stacked after the real February and March rows, get their GAN
class as label, and go through the same w2 ``fit`` scaler (fitted on real users
only). Every scenario then follows one protocol, so only the training rows
change:

* train on ``train_rows`` with the w2 best hyper-parameters of the model,
* early stopping on real ``stop``,
* threshold selected on real ``tune`` (max churn F1, as in w2),
* score real March (``test``) once.

Scenarios are built in the notebook, e.g. ``fit`` (baseline), ``fit`` + 50 % or
100 % synthetic churners, or synthetic only (TSTR).
"""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

W2_DIR = Path(__file__).resolve().parents[2] / "w2"
if str(W2_DIR) not in sys.path:
    sys.path.insert(0, str(W2_DIR))

from src.models.constants import ARTIFACTS_NAME, EXPORT_NAME, MODEL_CHANNELS  # noqa: E402
from src.models.evaluation.metrics import evaluate, select_threshold  # noqa: E402
from src.models.evaluation.significance import delong_test, paired_bootstrap  # noqa: E402
from src.models.feature_preparation.data import StackedSequenceView  # noqa: E402
from src.models.feature_preparation.scaling import transform_sequence  # noqa: E402
from src.models.recurrent_models.config import BATCH_SIZE, EPOCHS, OUTPUT_DIM, PATIENCE  # noqa: E402
from src.models.recurrent_models.data import load_prepared_data  # noqa: E402
from src.models.recurrent_models.recurrent_classifier import GRUModel, LSTMModel  # noqa: E402
from src.models.recurrent_models.train import predict_proba, train_model  # noqa: E402

EXPORT_DIR = W2_DIR / EXPORT_NAME
W2_ARTIFACTS = W2_DIR / "artifacts" / ARTIFACTS_NAME
FEATURE_DIR = W2_ARTIFACTS / "feature_preparation"
W2_MODELS = {"lstm": LSTMModel, "gru": GRUModel}


# ---------------------------------------------------------------------------
# w2 data and models
# ---------------------------------------------------------------------------

def load_w2_data():
    """The w2 prepared data (split + scalers of the calendar/manual run)."""
    data = load_prepared_data(feature_dir=FEATURE_DIR, export_dir=EXPORT_DIR)
    names = [spec["name"] for spec in data.channel_specs]
    if [names[k] for k in data.keep] != list(MODEL_CHANNELS):
        raise ValueError("The export channels differ from MODEL_CHANNELS.")
    return data


def real_rows(data, part, label=None):
    rows = data.rows(part)
    return rows if label is None else rows[data.y_all[rows] == label]


def scale(data, X, phase):
    """Raw w2-encoded users -> the scaled model input of ``phase``."""
    out = transform_sequence(np.asarray(X, dtype=np.float32), data.states[phase], data.channel_specs)
    return out[:, :, data.keep].astype(np.float32)


def w2_selection(model_name):
    """Best hyper-parameters, best epoch and frozen threshold of the w2 run."""
    base = W2_ARTIFACTS / model_name
    params = json.loads((base / "optuna" / f"{model_name}_best_params.json").read_text())
    meta = json.loads((base / "optuna" / f"{model_name}_best_meta.json").read_text())
    manifest = json.loads((base / "evaluation" / "selection_manifest.json").read_text())
    return {"params": params, "best_epoch": meta["best_epoch"], "threshold": manifest["threshold"]}


def load_w2_classifier(model_name):
    """The final w2 checkpoint (refit on all of February, ``development`` scaler)."""
    ckpt = torch.load(W2_ARTIFACTS / model_name / "checkpoints" / f"{model_name}.pt",
                      map_location="cpu", weights_only=False)
    params = ckpt["params"]
    model = W2_MODELS[model_name](input_dim=len(MODEL_CHANNELS), hidden_dim=int(params["hidden_dim"]),
                                  layer_dim=int(params["layer_dim"]), output_dim=OUTPUT_DIM)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(data, synthetic_by_label):
    """Return ``(data_aug, synth_rows)``: ``data`` plus synthetic rows after the real ones.

    ``synthetic_by_label`` maps class -> ``(n, 90, 14)`` array in the export
    encoding. ``synth_rows[label]`` are their row indices in ``data_aug``.
    Real rows, split and scalers are unchanged.
    """
    parts = list(data.view.parts)
    y_parts = [data.y_all]
    synth_rows = {}
    offset = data.view.shape[0]
    for label, X in synthetic_by_label.items():
        X = np.asarray(X, dtype=np.float16)
        if X.shape[1:] != data.view.shape[1:]:
            raise ValueError(f"synthetic shape {X.shape} vs real {data.view.shape}")
        parts.append(X)
        y_parts.append(np.full(len(X), label, dtype=data.y_all.dtype))
        synth_rows[label] = np.arange(offset, offset + len(X))
        offset += len(X)
    data_aug = replace(data, view=StackedSequenceView(parts), y_all=np.concatenate(y_parts),
                       split=dict(data.split), _raw=None, _scaled={})
    return data_aug, synth_rows


def take(rows, fraction_or_n, seed=0):
    """A random subset of ``rows``: a fraction of ``len(rows)`` (float) or a count (int)."""
    n = int(round(fraction_or_n * len(rows))) if isinstance(fraction_or_n, float) else int(fraction_or_n)
    if n > len(rows):
        raise ValueError(f"asked for {n} synthetic rows, only {len(rows)} generated")
    return np.sort(np.random.default_rng(seed).choice(rows, n, replace=False))


# ---------------------------------------------------------------------------
# One scenario = one classifier trained on a set of rows
# ---------------------------------------------------------------------------

def run_scenario(data, model_name, train_rows, *, device, seed, verbose=False):
    """Train the w2 ``model_name`` on ``train_rows``; return ``(run, tune)`` in w2 format."""
    selection = w2_selection(model_name)
    params = selection["params"]
    model = W2_MODELS[model_name](input_dim=data.input_dim, hidden_dim=int(params["hidden_dim"]),
                                  layer_dim=int(params["layer_dim"]), output_dim=OUTPUT_DIM)
    start = time.perf_counter()
    fit = train_model(model, data, train_rows, data.rows("stop"), epochs=EPOCHS,
                      lr=float(params["lr"]), weight_decay=float(params["weight_decay"]),
                      batch_size=int(params.get("batch_size", BATCH_SIZE)), patience=PATIENCE,
                      device=device, phase="fit", seed=seed, verbose=verbose)
    seconds = time.perf_counter() - start

    y_tune = data.labels("tune")
    proba_tune = predict_proba(model, data, data.rows("tune"), phase="fit", device=device)
    threshold = select_threshold(y_tune, proba_tune)
    proba_test = predict_proba(model, data, data.rows("test"), phase="fit", device=device)
    run = {
        "model": model, "history": fit["history"], "y_test": data.labels("test"),
        "proba_test": proba_test, "pred_test": (proba_test >= threshold).astype(int),
        "threshold": threshold, "epochs": fit["best_epoch"],
        "seconds_per_epoch": seconds / max(fit["stopped_epoch"], 1),
        "n_train": len(train_rows),
        "train_churn_rate": float(data.y_all[train_rows].mean()),
    }
    tune = {"threshold": threshold, "best_epoch": fit["best_epoch"],
            "tune_log_loss": float(evaluate(y_tune, proba_tune, model_name, threshold)["log_loss"])}
    return run, tune


def scenario_table(runs, tunes):
    """Test metrics of every scenario at its own tune threshold."""
    rows = []
    for name, run in runs.items():
        m = evaluate(run["y_test"], run["proba_test"], name, tunes[name]["threshold"])
        rows.append({
            "scenario": name, "train users": run["n_train"],
            "train churn rate": run["train_churn_rate"], "best epoch": run["epochs"],
            "AUC-ROC": m["roc_auc"], "AUC-PR": m["average_precision"], "F1": m["f1_churn"],
            "precision": m["precision_churn"], "recall": m["recall_churn"],
            "log-loss": m["log_loss"], "brier": m["brier"], "threshold": m["threshold"],
        })
    return pd.DataFrame(rows).set_index("scenario")


def versus_baseline(runs, tunes, baseline, n_boot=1000):
    """Paired bootstrap (scenario - baseline) and DeLong for every scenario."""
    rows = []
    for name in runs:
        if name == baseline:
            continue
        pair_runs = {name: runs[name], baseline: runs[baseline]}
        pair_tunes = {name: tunes[name], baseline: tunes[baseline]}
        boot = paired_bootstrap(pair_runs, pair_tunes, n_boot=n_boot)
        delong = delong_test(runs[name]["y_test"], runs[name]["proba_test"], runs[baseline]["proba_test"])
        diff_col = [c for c in boot.columns if c.startswith("Δ ") and "IC95" not in c][0]
        for metric, row in boot.iterrows():
            rows.append({"scenario": name, "metric": metric, "Δ vs baseline": row[diff_col],
                         "Δ IC95": row["Δ IC95"], "significant": row["Significativo"],
                         "DeLong p (AUC-ROC)": delong["p_value"] if metric == "AUC-ROC" else np.nan})
    return pd.DataFrame(rows).set_index(["scenario", "metric"])


# ---------------------------------------------------------------------------
# Trajectory models (w2 approach 3): days 0..82 -> the 11 events of days 83..89
# ---------------------------------------------------------------------------
# No churn label is involved: synthetic users bring their own (x, y) pairs
# through ``TrajectoryData``. Same protocol as the classifier scenarios: w2 best
# hyper-parameters and loss weights, early stopping on real ``stop``, ``fit``
# scaler, real March scored once with ``evaluate_trajectories``.

from src.models.evaluation.significance import _ModelScores  # noqa: E402
from src.models.trajectory_models import config as traj_config  # noqa: E402
from src.models.trajectory_models.data import TrajectoryData  # noqa: E402
from src.models.trajectory_models.evaluation import (  # noqa: E402
    evaluate_trajectories,
    paired_bootstrap_mean,
)
from src.models.trajectory_models.model import build_model as build_trajectory_model  # noqa: E402
from src.models.trajectory_models.train import (  # noqa: E402
    predict_trajectories,
    train_model as train_trajectory_model,
)

TRAJ_ARTIFACTS = W2_DIR / "artifacts" / f"{ARTIFACTS_NAME}_traj"


def trajectory_selection(model_name):
    """w2 best hyper-parameters of the trajectory model and the fixed loss weights."""
    params = json.loads((TRAJ_ARTIFACTS / model_name / "optuna" / f"{model_name}_best_params.json").read_text())
    weights = json.loads((TRAJ_ARTIFACTS / "loss_weights.json").read_text())["weights"]
    return {"params": params, "loss_weights": weights}


def mixed_synthetic_rows(synth_rows, n, churn_rate, seed=0):
    """``n`` synthetic rows from both classes with the given churn rate."""
    if 0 not in synth_rows or 1 not in synth_rows:
        raise ValueError("Trajectory scenarios need synthetic users of both classes (GAN_LABELS = (1, 0)).")
    n_churn = int(round(n * churn_rate))
    return np.sort(np.concatenate([take(synth_rows[1], n_churn, seed=seed),
                                   take(synth_rows[0], n - n_churn, seed=seed)]))


def run_trajectory_scenario(data, model_name, train_rows, *, device, seed, verbose=False):
    """Train the w2 trajectory ``model_name`` on ``train_rows``; return the March forecast."""
    selection = trajectory_selection(model_name)
    params = selection["params"]
    tdata = TrajectoryData(data)
    model = build_trajectory_model(model_name, params, tdata.input_dim)
    start = time.perf_counter()
    fit = train_trajectory_model(
        model, tdata, train_rows, tdata.rows("stop"), loss_weights=selection["loss_weights"],
        epochs=traj_config.EPOCHS, lr=float(params["lr"]), weight_decay=float(params["weight_decay"]),
        batch_size=int(params.get("batch_size", traj_config.BATCH_SIZE)), patience=traj_config.PATIENCE,
        device=device, phase="fit", seed=seed, verbose=verbose)
    seconds = time.perf_counter() - start
    pred = predict_trajectories(model, tdata, tdata.rows("test"), phase="fit", device=device)
    return {"pred": pred, "history": fit["history"], "best_epoch": fit["best_epoch"],
            "best_stop_loss": fit["best_stop_loss"], "n_train": len(train_rows),
            "seconds_per_epoch": seconds / max(fit["stopped_epoch"], 1)}


def trajectory_report(data, runs):
    """``evaluate_trajectories`` on real March for every scenario (``fit`` scaler)."""
    preds = {name: run["pred"] for name, run in runs.items()}
    return evaluate_trajectories(preds, TrajectoryData(data), data.rows("test"),
                                 scaler_state=data.states["fit"])


def trajectory_summary(report, runs):
    """One row per scenario with the headline numbers of the w2 trajectory notebook."""
    rows = []
    for name, run in runs.items():
        label = name.upper()
        events = report["events"]
        rows.append({
            "scenario": name, "train users": run["n_train"], "best epoch": run["best_epoch"],
            "stop loss": run["best_stop_loss"],
            "listening MAE": report["listening"].loc[label, "MAE"],
            "skill vs persistence": report["listening"].loc[label, "skill_vs_persistence"],
            "active AUC": events.loc[("active", label), "AUC"],
            "pay AUC": events.loc[("pay", label), "AUC"],
            "pay AP": events.loc[("pay", label), "AP"],
            "weekly pay AP": events.loc[("pay_week", label), "AP"],
            "MAE paid (NTD)": report["amounts"].loc[label, "MAE_paid_NTD"],
            "MAE plan days": report["amounts"].loc[label, "MAE_plan_days"],
        })
    return pd.DataFrame(rows).set_index("scenario")


def _cluster_bootstrap_events(y, score_a, score_b, n_users, n_boot, seed, alpha=0.05):
    """AUC/AP difference ``a - b`` resampling users (all days of a user move together)."""
    per_user = len(y) // n_users
    a, b = _ModelScores(y, score_a, 0.5), _ModelScores(y, score_b, 0.5)
    rng = np.random.default_rng(seed)
    ones = np.ones(len(y))
    point = a.metrics(ones)[:2] - b.metrics(ones)[:2]
    draws = np.empty((n_boot, 2))
    for k in range(n_boot):
        w = np.repeat(np.bincount(rng.integers(0, n_users, n_users), minlength=n_users), per_user)
        draws[k] = a.metrics(w.astype(np.float64))[:2] - b.metrics(w.astype(np.float64))[:2]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return point, lo, hi


def trajectory_versus_baseline(report, baseline, n_boot=500, seed=0):
    """Scenario - baseline with 95 % paired bootstrap over March users.

    Listening: per-user MAE (lower is better). Events: AUC and AP of ``active``
    and ``pay`` (user-day) and weekly ``pay_week`` (user), resampling users.
    """
    names = list(report["model_names"])
    n_users = len(report["per_user_listen_mae"][baseline])
    rows = []
    for name in names:
        if name == baseline:
            continue
        boot = paired_bootstrap_mean(report["per_user_listen_mae"][name],
                                     report["per_user_listen_mae"][baseline], n_boot=n_boot, seed=seed)
        rows.append({"scenario": name, "metric": "listening MAE (per user)", "Δ vs baseline": boot["diff"],
                     "Δ IC95": f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]",
                     "significant": boot["significant"]})
        for target in ("active", "pay", "pay_week"):
            y = report["event_labels"][target].astype(np.float64)
            scores = report["event_scores"][target]
            point, lo, hi = _cluster_bootstrap_events(
                y, scores[name.upper()], scores[baseline.upper()], n_users, n_boot, seed)
            for k, metric in enumerate(("AUC", "AP")):
                rows.append({"scenario": name, "metric": f"{target} {metric}", "Δ vs baseline": point[k],
                             "Δ IC95": f"[{lo[k]:.4f}, {hi[k]:.4f}]",
                             "significant": bool(lo[k] > 0 or hi[k] < 0)})
    return pd.DataFrame(rows).set_index(["scenario", "metric"])
