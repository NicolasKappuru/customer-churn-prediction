"""ROC, precision-recall and calibration diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..constants import CLASS_LABELS
from .figure_utils import show_or_close


def plot_roc_pr(y_true, proba, model_id, out_dir, show_plots=True):
    """Save the ROC and precision-recall curves for one model."""
    curves_dir = Path(out_dir) / "figures" / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    fpr, tpr, _ = roc_curve(y_true, proba)
    ax_roc.plot(
        fpr,
        tpr,
        label=f"{model_id} (AUC={roc_auc_score(y_true, proba):.3f})",
    )
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ax_pr.plot(
        recall,
        precision,
        label=f"{model_id} (AP={average_precision_score(y_true, proba):.3f})",
    )

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    ax_roc.set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    ax_roc.legend(fontsize=7)
    ax_pr.axhline(
        float(np.mean(y_true)),
        color="k",
        ls="--",
        alpha=0.5,
        label="prevalence",
    )
    ax_pr.set(
        title="Precision-Recall (test)",
        xlabel="Recall",
        ylabel="Precision",
    )
    ax_pr.legend(fontsize=7)

    fig.tight_layout()
    path = curves_dir / "roc_pr.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    show_or_close(fig, show_plots)
    return path


def plot_reliability_histogram(
    y_true, proba, model_id, out_dir, show_plots=True
):
    """Save the reliability curve and probability histogram for one model."""
    curves_dir = Path(out_dir) / "figures" / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fraction_positive, mean_predicted = calibration_curve(
        y_true, proba, n_bins=10, strategy="quantile"
    )
    axes[0].plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label=model_id,
        alpha=0.7,
    )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(
        title="Reliability (test, diagnostic only)",
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
    )
    axes[0].legend(fontsize=7)

    labels = np.asarray(y_true)
    for cls in (0, 1):
        axes[1].hist(
            np.asarray(proba)[labels == cls],
            bins=40,
            alpha=0.6,
            label=CLASS_LABELS[cls],
            density=True,
        )
    axes[1].set(
        title=f"Probability histogram - {model_id}",
        xlabel="probability",
    )
    axes[1].legend()

    fig.tight_layout()
    path = curves_dir / "calibration_hist.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    show_or_close(fig, show_plots)
    return path
