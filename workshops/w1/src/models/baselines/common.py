"""Prepared splits, scoring, and registry fields shared by the baselines."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ..constants import EPS


@dataclass
class BaselineData:
    fit_frame: pd.DataFrame
    fit_labels: np.ndarray
    tune_frame: pd.DataFrame
    tune_labels: np.ndarray
    holdout_frame: pd.DataFrame
    stop_frame: pd.DataFrame | None = None
    stop_labels: np.ndarray | None = None


def tune_log_loss(model, data, eps=EPS):
    """Score a fitted model on the tune split."""
    proba = model.predict_proba(data.tune_frame)[:, 1]
    return log_loss(data.tune_labels, np.clip(proba, eps, 1 - eps), labels=[0, 1])


def register_baseline(registry, model_id, model, data, counts, seconds, **record):
    """Store a baseline record and its provisional predictions."""
    registry.set_proba(model_id, "tune", model.predict_proba(data.tune_frame)[:, 1])
    registry.set_proba(
        model_id, "holdout", model.predict_proba(data.holdout_frame)[:, 1]
    )
    return registry.register(
        model_id,
        input="table",
        n_channels="N/A",
        n_parameters="N/A",
        trials=counts,
        fit_seconds=seconds,
        estimator=model,
        **record,
    )
