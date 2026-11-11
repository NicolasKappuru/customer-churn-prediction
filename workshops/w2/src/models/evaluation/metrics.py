"""Threshold selection and point metrics for the w2 model."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..constants import EPS


def select_threshold(y_true, proba):
    """Threshold maximising churn F1; ties prefer the higher threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    grid = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0

    for threshold in grid:
        f1 = f1_score(
            y_true, (proba >= threshold).astype(int), zero_division=0
        )
        if f1 >= best_f1:
            best_f1, best_t = f1, float(threshold)
    return best_t


def evaluate(y_true, proba, model_id, threshold):
    """Full metric row at a fixed threshold. No training or threshold search."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=np.float64), EPS, 1 - EPS)
    pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "model_id": model_id,
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_true, proba) if two_classes else np.nan,
        "average_precision": (
            average_precision_score(y_true, proba) if two_classes else np.nan
        ),
        "brier": brier_score_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_churn": precision_score(y_true, pred, zero_division=0),
        "recall_churn": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "f1_churn": f1_score(y_true, pred, zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred) if two_classes else 0.0,
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "support_0": int((y_true == 0).sum()),
        "support_1": int((y_true == 1).sum()),
        "actual_churn_rate": float(y_true.mean()),
        "predicted_churn_rate": float(pred.mean()),
    }


def metrics_at_half(y_true, proba, model_id):
    return evaluate(y_true, proba, model_id, 0.5)
