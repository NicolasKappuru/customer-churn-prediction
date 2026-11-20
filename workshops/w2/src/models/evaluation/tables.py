"""Metrics tables for the single w2 model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import evaluate, metrics_at_half


DISPLAY_COLUMNS = [
    "model_id",
    "family",
    "input",
    "n_channels",
    "n_parameters",
    "best_epoch",
    "tune_log_loss",
    "log_loss",
    "roc_auc",
    "average_precision",
    "brier",
    "accuracy",
    "balanced_accuracy",
    "precision_churn",
    "recall_churn",
    "specificity",
    "f1_churn",
    "f1_macro",
    "mcc",
    "threshold",
]


def _model_metadata(model_info):
    """Flatten optional model metadata into the metrics row."""
    if not model_info:
        return {}
    return {
        "family": model_info.get("family", "sequence"),
        "input": model_info.get("input", "sequence"),
        "n_channels": model_info.get("n_channels", "N/A"),
        "n_parameters": model_info.get("n_parameters", "N/A"),
        "best_epoch": model_info.get("best_epoch", "N/A"),
        "best_params": json.dumps(model_info.get("best_params", "N/A")),
        "trial_number": model_info.get("trial_number", "N/A"),
    }


def build_metrics_table(
    y_tune,
    y_test,
    proba_test,
    threshold,
    tune_log_loss,
    model_id,
    model_info=None,
):
    """Return the test metrics at the tune threshold and at 0.5."""
    main_row = evaluate(y_test, proba_test, model_id, threshold)
    main_row.update(_model_metadata(model_info))
    main_row["tune_log_loss"] = float(tune_log_loss)
    main_row["tune_churn_rate"] = float(np.mean(y_tune))
    half_row = metrics_at_half(y_test, proba_test, model_id)
    return pd.DataFrame([main_row]), pd.DataFrame([half_row])


def save_metrics_tables(main, at_half, out_dir):
    """Write both fixed-threshold metrics tables."""
    metrics_dir = Path(out_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    main_path = metrics_dir / "test_metrics.csv"
    half_path = metrics_dir / "test_metrics_at_05.csv"
    main.to_csv(main_path, index=False)
    at_half.to_csv(half_path, index=False)
    return main_path, half_path
