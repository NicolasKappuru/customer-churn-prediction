"""Confusion-matrix figures: counts and row-normalised percentages."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from ..constants import CLASS_LABELS, MODEL_IDS
from .figure_utils import show_or_close


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

    # Counts panel (own colour scale).
    im0 = ax_counts.imshow(cm, cmap="Blues")
    ax_counts.figure.colorbar(im0, ax=ax_counts, fraction=0.046, pad=0.04)
    names = ["TN", "FP", "FN", "TP"]

    for (r, c), v in np.ndenumerate(cm):
        ax_counts.text(
            c,
            r,
            f"{names[r * 2 + c]}\n{v:d}",
            ha="center",
            va="center",
            color="white" if v > cm.max() / 2 else "black",
            fontsize=10,
        )

    ax_counts.set_title(f"{title}\ncounts @ thr={threshold:.2f}", fontsize=10)

    # Row-normalised panel (shared 0-100 scale).
    im1 = ax_norm.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    ax_norm.figure.colorbar(im1, ax=ax_norm, fraction=0.046, pad=0.04)

    for (r, c), v in np.ndenumerate(norm):
        ax_norm.text(
            c,
            r,
            f"{v:.1f}%",
            ha="center",
            va="center",
            color="white" if v > 50 else "black",
            fontsize=10,
        )

    ax_norm.set_title(f"{title}\nrow-normalised %", fontsize=10)

    for ax in (ax_counts, ax_norm):
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_LABELS, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_LABELS, fontsize=8)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")


def save_confusion(y_true, proba, threshold, model_id, out_dir, tag="test"):
    """Save a single high-resolution confusion figure (counts + normalised) as PNG+SVG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _draw_confusion(
        axes[0], axes[1], y_true, proba, threshold, f"{model_id} ({tag})"
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / f"confusion_{model_id}_{tag}.png", dpi=200, bbox_inches="tight"
    )
    fig.savefig(
        out_dir / f"confusion_{model_id}_{tag}.svg", bbox_inches="tight"
    )
    return fig


def gallery_confusion(entries, thresholds, out_dir, tag="test", per_page=3):
    """Paginated gallery, up to `per_page` models per figure (counts + normalised cols)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = list(entries.items())
    figs = []

    for page in range(0, len(items), per_page):
        chunk = items[page:page + per_page]
        fig, axes = plt.subplots(
            len(chunk), 2, figsize=(9, 4 * len(chunk)), squeeze=False
        )

        for r, (model_id, (y_true, proba)) in enumerate(chunk):
            _draw_confusion(
                axes[r, 0],
                axes[r, 1],
                y_true,
                proba,
                thresholds[model_id],
                f"{model_id} ({tag})",
            )

        fig.tight_layout()
        fig.savefig(
            out_dir / f"gallery_{tag}_{page // per_page}.png",
            dpi=160,
            bbox_inches="tight",
        )
        figs.append(fig)

    return figs


def save_confusion_figures(
    y_test, registry, thresholds, out_dir, show_plots=True
):
    """Write individual matrices and a paginated gallery."""
    confusion_dir = Path(out_dir) / "figures" / "confusion"
    entries = {
        model_id: (y_test, registry.proba[model_id]["test"])
        for model_id in MODEL_IDS
    }
    for model_id in MODEL_IDS:
        fig = save_confusion(
            y_test, entries[model_id][1], thresholds[model_id],
            model_id, confusion_dir, tag="test",
        )
        plt.close(fig)
    gallery = gallery_confusion(
        entries, thresholds, confusion_dir, tag="test", per_page=3
    )
    for fig in gallery:
        show_or_close(fig, show_plots)
    print("saved per-model confusion images (PNG+SVG) and gallery pages")
    return confusion_dir
