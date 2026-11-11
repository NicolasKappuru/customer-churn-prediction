"""Confusion-matrix figure for the single w2 model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from ..constants import CLASS_LABELS


def _draw_confusion(ax_counts, ax_norm, y_true, proba, threshold, title):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)

    norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    im0 = ax_counts.imshow(cm, cmap="Blues")
    ax_counts.figure.colorbar(im0, ax=ax_counts, fraction=0.046, pad=0.04)
    names = ["TN", "FP", "FN", "TP"]

    for (row, column), value in np.ndenumerate(cm):
        ax_counts.text(
            column,
            row,
            f"{names[row * 2 + column]}\n{value:d}",
            ha="center",
            va="center",
            color="white" if value > cm.max() / 2 else "black",
            fontsize=10,
        )

    ax_counts.set_title(
        f"{title}\ncounts @ thr={threshold:.2f}", fontsize=10
    )

    im1 = ax_norm.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    ax_norm.figure.colorbar(im1, ax=ax_norm, fraction=0.046, pad=0.04)

    for (row, column), value in np.ndenumerate(norm):
        ax_norm.text(
            column,
            row,
            f"{value:.1f}%",
            ha="center",
            va="center",
            color="white" if value > 50 else "black",
            fontsize=10,
        )

    ax_norm.set_title(f"{title}\nrow-normalised %", fontsize=10)

    for ax in (ax_counts, ax_norm):
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_LABELS, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_LABELS, fontsize=8)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual class")


def save_confusion(y_true, proba, threshold, model_id, out_dir, tag="test"):
    """Save the counts and row-normalised confusion panels as PNG and SVG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _draw_confusion(
        axes[0], axes[1], y_true, proba, threshold, f"{model_id} ({tag})"
    )
    fig.tight_layout()
    png = out_dir / f"confusion_{model_id}_{tag}.png"
    svg = out_dir / f"confusion_{model_id}_{tag}.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return fig, png, svg
