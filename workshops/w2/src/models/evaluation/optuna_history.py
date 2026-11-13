import json

import matplotlib.pyplot as plt
import pandas as pd

from .figure_utils import show_or_close
from ..recurrent_models.config import model_dirs

MODEL_COLORS = {"lstm": "tab:blue", "gru": "tab:orange"}


def _load_history(model_name):
    optuna_dir = model_dirs(model_name)["optuna"]
    history_path = optuna_dir / f"{model_name}_best_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(
            f"No history for '{model_name}' at {history_path}. "
            "Re-run tune_model after adding the history attribute."
        )
    history = pd.read_csv(history_path)

    best_epoch = None
    meta_path = optuna_dir / f"{model_name}_best_meta.json"
    if meta_path.exists():
        best_epoch = json.loads(meta_path.read_text()).get("best_epoch")
    return history, best_epoch


def _draw_history(ax, model_name):
    history, best_epoch = _load_history(model_name)
    color = MODEL_COLORS.get(model_name)
    ax.plot(history["epoch"], history["train_loss"],
            color=color, linestyle="--", label="train")
    ax.plot(history["epoch"], history["stop_loss"],
            color=color, linestyle="-", label="stop (val)")
    if best_epoch:
        ax.axvline(best_epoch, color="gray", linestyle=":",
                   label=f"best epoch ({best_epoch})")
    ax.set_title(model_name.upper())
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()


def plot_optuna_history(model_name, out_dir, show_plots=True):
    """Single-model figure (used by evaluate_model). Returns the saved path."""
    model_name = model_name.lower()
    fig, ax = plt.subplots(figsize=(5, 4))
    _draw_history(ax, model_name)
    ax.set_ylabel("BCE loss")
    fig.tight_layout()

    path = out_dir / "figures" / f"{model_name}_training_history.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    show_or_close(fig, show_plots)
    return path


def plot_training_history(model_names=("lstm", "gru"), *, save_path=None):
    """Side-by-side comparison with a shared y-axis. Call it on its own."""
    names = [n.lower() for n in model_names]
    fig, axes = plt.subplots(
        1, len(names), figsize=(5 * len(names), 4), sharey=True, squeeze=False
    )
    for ax, name in zip(axes[0], names):
        _draw_history(ax, name)
    axes[0][0].set_ylabel("BCE loss")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig