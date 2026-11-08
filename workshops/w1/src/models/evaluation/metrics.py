import numpy as np
from ..constants import EPS
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true, proba, criterion="f1"):
    """Return the threshold maximising the given criterion using precision_recall_curve.

    criterion: "f1" | "f2" | "recall>=0.70"
    Default "f1" keeps backward compatibility with existing pipeline calls.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    p, r = precisions[:-1], recalls[:-1]

    if criterion == "f1":
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    elif criterion == "f2":
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where((4 * p + r) > 0, 5 * p * r / (4 * p + r), 0.0)
    elif criterion == "recall>=0.70":
        mask = r >= 0.70
        scores = np.where(mask, p, -1.0) if mask.any() else r
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")

    return float(thresholds[np.argmax(scores)])


def evaluate(y_true, proba, model_id, threshold):
    """Full metric row at a fixed threshold. No training, no threshold search here."""
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
