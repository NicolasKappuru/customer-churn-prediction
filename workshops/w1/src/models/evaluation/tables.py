"""Build and persist the March evaluation tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import MODEL_IDS
from .metrics import evaluate, metrics_at_half


DISPLAY_COLUMNS = [
    "model_id", "family", "tune_log_loss", "log_loss", "roc_auc",
    "average_precision", "brier", "accuracy", "balanced_accuracy",
    "precision_churn", "recall_churn", "specificity", "f1_churn",
    "f1_macro", "mcc", "threshold",
]


def _model_metadata(record):
    """Flatten registry metadata into one metrics row."""
    return {
        "family": record["family"],
        "input": record["input"],
        "n_channels": record.get("n_channels", "N/A"),
        "n_parameters": record.get("n_parameters", "N/A"),
        "trials_complete": record["trials"]["complete"],
        "trials_pruned": record["trials"]["pruned"],
        "trials_failed": record["trials"]["failed"],
        "fit_seconds": round(float(record.get("fit_seconds", 0.0)), 2),
        "best_iteration": record.get("best_iteration", "N/A"),
        "best_params": json.dumps(record.get("best_params", "N/A")),
    }


def build_holdout_tables(y_tune, y_hold, registry, thresholds, selection):
    """Evaluate all models at their tune threshold and at 0.5."""
    rows_main = []
    rows_half = []
    for model_id in MODEL_IDS:
        proba = registry.proba[model_id]["holdout"]
        row = evaluate(y_hold, proba, model_id, thresholds[model_id])
        row.update(_model_metadata(registry.records[model_id]))
        rows_main.append(row)
        rows_half.append(metrics_at_half(y_hold, proba, model_id))

    main = pd.DataFrame(rows_main).sort_values("log_loss").reset_index(drop=True)
    at_half = pd.DataFrame(rows_half).sort_values("log_loss").reset_index(drop=True)
    main["tune_log_loss"] = main["model_id"].map(selection["tune_log_loss"])
    main["tune_churn_rate"] = float(np.mean(y_tune))
    return main, at_half


def save_holdout_tables(main, at_half, out_dir):
    """Write both fixed-threshold metrics tables."""
    metrics_dir = Path(out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    main.to_csv(metrics_dir / "holdout_metrics.csv", index=False)
    at_half.to_csv(metrics_dir / "holdout_metrics_at_05.csv", index=False)
