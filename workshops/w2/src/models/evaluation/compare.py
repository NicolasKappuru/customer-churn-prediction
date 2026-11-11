"""Side-by-side comparison of the w2 recurrent models (LSTM vs GRU).

Only the comparison outputs live here: ROC/PR and gain/lift with one line per
model, the metrics and hyper-parameter tables, and the bootstrap/DeLong
uncertainty. Per-model diagnostics (confusion matrix, calibration, Optuna
history) stay in ``evaluate_model``.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..recurrent_models.config import ARTIFACTS_ROOT
from .figure_utils import show_or_close
from .metrics import evaluate
from .significance import delong_test, paired_bootstrap

COMPARISON_DIR = ARTIFACTS_ROOT / "comparison"

# Same color for each model across the whole work.
MODEL_COLORS = {"lstm": "tab:blue", "gru": "tab:orange"}

# Table column -> True if higher is better (used to bold the best value).
BEST_IS_MAX = {
    "AUC-ROC": True,
    "AUC-PR": True,
    "F1": True,
    "Log-loss": False,
    "Parámetros": False,
    "Tiempo/época (s)": False,
}


def plot_roc_pr_comparison(runs, out_dir=COMPARISON_DIR, show_plots=True):
    """ROC and precision-recall curves, one line per model on the same axes.

    ``runs`` maps model_name to its ``test_model`` result,
    e.g. ``{"lstm": results_lstm, "gru": results_gru}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for name, result in runs.items():
        y, proba = result["y_test"], result["proba_test"]
        color = MODEL_COLORS.get(name)
        fpr, tpr, _ = roc_curve(y, proba)
        ax_roc.plot(
            fpr, tpr, color=color,
            label=f"{name.upper()} (AUC={roc_auc_score(y, proba):.3f})",
        )
        precision, recall, _ = precision_recall_curve(y, proba)
        ax_pr.plot(
            recall, precision, color=color,
            label=f"{name.upper()} (AP={average_precision_score(y, proba):.3f})",
        )

    # All models are scored on the same test split, so the prevalence is shared.
    y_test = next(iter(runs.values()))["y_test"]
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    ax_roc.set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    ax_roc.legend(fontsize=8)
    ax_pr.axhline(
        float(np.mean(y_test)), color="k", ls="--", alpha=0.5,
        label="prevalence",
    )
    ax_pr.set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax_pr.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "roc_pr_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    show_or_close(fig, show_plots)
    return path


def comparison_table(runs, tunes, out_dir=COMPARISON_DIR):
    """One row per model, test metrics at each model's frozen tune threshold.

    ``tunes`` maps model_name to its ``tune_model`` result (for the threshold).
    """
    rows = []
    for name, result in runs.items():
        threshold = tunes[name]["threshold"]
        metrics = evaluate(
            result["y_test"], result["proba_test"], name.upper(), threshold
        )
        model = result.get("model")
        rows.append({
            "Modelo": name.upper(),
            "AUC-ROC": metrics["roc_auc"],
            "AUC-PR": metrics["average_precision"],
            "F1": metrics["f1_churn"],
            "Log-loss": metrics["log_loss"],
            "Parámetros": (
                int(sum(p.numel() for p in model.parameters()))
                if model is not None else np.nan
            ),
            "Tiempo/época (s)": result.get("seconds_per_epoch", np.nan),
            "Umbral": threshold,
        })

    table = pd.DataFrame(rows).set_index("Modelo")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison_metrics.csv")
    return table


def highlight_best(table):
    """Notebook view of the table with the best value of each metric in bold."""
    def bold_best(column):
        best = column.max() if BEST_IS_MAX[column.name] else column.min()
        return ["font-weight: bold" if value == best else "" for value in column]

    columns = [c for c in BEST_IS_MAX if c in table.columns]
    return (
        table.style
        .apply(bold_best, subset=columns)
        .format(precision=4)
        .format("{:,.0f}", subset=["Parámetros"])
        .format("{:.2f}", subset=["Tiempo/época (s)", "Umbral"])
    )


def hyperparams_table(tunes, out_dir=COMPARISON_DIR):
    """Best Optuna configuration of each model (annex table)."""
    rows = []
    for name, tune in tunes.items():
        params = tune["best_params"]
        rows.append({
            "Modelo": name.upper(),
            "layer_dim": int(params["layer_dim"]),
            "hidden_dim": int(params["hidden_dim"]),
            "lr": float(params["lr"]),
            "weight_decay": float(params["weight_decay"]),
            "batch_size": int(params.get("batch_size", 0)),
            "best_epoch": int(tune.get("best_epoch", 0)),
            "trial": int(tune.get("trial_number", -1)),
            "tune log-loss": float(tune["best_value"]),
        })
    table = pd.DataFrame(rows).set_index("Modelo")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "best_hyperparams.csv")
    return table


def _gain_curve(y, proba, grid):
    """Share of churners captured when contacting the top ``grid`` share of users."""
    y = np.asarray(y)
    ranked = y[np.argsort(-np.asarray(proba), kind="mergesort")]
    captured = np.r_[0, np.cumsum(ranked)] / ranked.sum()
    positions = np.round(grid * len(ranked)).astype(int)
    return captured[positions]


def plot_gain_lift(
    runs,
    out_dir=COMPARISON_DIR,
    show_plots=True,
    cutoffs=(0.05, 0.10, 0.20, 0.30),
):
    """Cumulative gain and lift curves, one line per model, plus a cutoff table.

    Gain at 10% = share of all churners found by contacting the 10% of users with
    the highest predicted risk. Lift = gain / contacted share (1 = random).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, 201)
    fig, (ax_gain, ax_lift) = plt.subplots(1, 2, figsize=(14, 6))

    rows = []
    for name, result in runs.items():
        y, proba = result["y_test"], result["proba_test"]
        gain = _gain_curve(y, proba, grid)
        color = MODEL_COLORS.get(name)
        ax_gain.plot(grid, gain, color=color, label=name.upper())
        ax_lift.plot(grid[1:], gain[1:] / grid[1:], color=color, label=name.upper())

        cut_gain = _gain_curve(y, proba, np.asarray(cutoffs))
        for share, g in zip(cutoffs, cut_gain):
            rows.append({
                "Modelo": name.upper(),
                "Top contactado": f"{share:.0%}",
                "Churners capturados": g,
                "Lift": g / share,
            })

    ax_gain.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    ax_gain.set(
        title="Cumulative gain (test)",
        xlabel="Share of users contacted (highest risk first)",
        ylabel="Share of churners captured",
    )
    ax_gain.legend(fontsize=8)
    ax_lift.axhline(1, color="k", ls="--", alpha=0.5, label="random")
    ax_lift.set(
        title="Lift (test)",
        xlabel="Share of users contacted (highest risk first)",
        ylabel="Lift",
    )
    ax_lift.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "gain_lift_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    show_or_close(fig, show_plots)

    table = pd.DataFrame(rows).pivot(
        index="Top contactado", columns="Modelo",
        values=["Churners capturados", "Lift"],
    )
    table = table.loc[[f"{share:.0%}" for share in cutoffs]]
    table = table.reindex(columns=[name.upper() for name in runs], level=1)
    table.to_csv(out_dir / "gain_lift_cutoffs.csv")
    return path, table


def compare_models(
    runs,
    tunes,
    out_dir=COMPARISON_DIR,
    show_plots=True,
    n_boot=1000,
):
    """Write every LSTM vs GRU comparison output.

    - ROC/PR and gain/lift figures (one line per model),
    - metrics table (best value in bold) and best hyper-parameters table,
    - paired-bootstrap 95% CIs and the DeLong test for AUC-ROC.
    """
    out_dir = Path(out_dir)
    roc_pr_path = plot_roc_pr_comparison(runs, out_dir, show_plots)
    gain_path, lift_table = plot_gain_lift(runs, out_dir, show_plots)
    table = comparison_table(runs, tunes, out_dir)
    hyperparams = hyperparams_table(tunes, out_dir)

    bootstrap = paired_bootstrap(runs, tunes, n_boot=n_boot)
    bootstrap.to_csv(out_dir / "bootstrap_ci.csv")

    delong = None
    names = list(runs)
    if len(names) == 2:
        a, b = names
        delong = delong_test(
            runs[a]["y_test"], runs[a]["proba_test"], runs[b]["proba_test"]
        )
        delong = {"model_a": a.upper(), "model_b": b.upper(), **delong}
        (out_dir / "delong_auc.json").write_text(json.dumps(delong, indent=2))

    return {
        "table": table,
        "display": highlight_best(table),
        "hyperparams": hyperparams,
        "lift": lift_table,
        "bootstrap": bootstrap,
        "delong": delong,
        "paths": {
            "roc_pr": roc_pr_path,
            "gain_lift": gain_path,
            "table": out_dir / "comparison_metrics.csv",
            "hyperparams": out_dir / "best_hyperparams.csv",
            "lift": out_dir / "gain_lift_cutoffs.csv",
            "bootstrap": out_dir / "bootstrap_ci.csv",
            "delong": out_dir / "delong_auc.json" if delong else None,
        },
    }
