"""Optuna optimization history for the single w2 study."""

from pathlib import Path

import optuna

from .figure_utils import show_or_close


def plot_optuna_history(study, model_id, out_dir, show_plots=True):
    """Save the optimization history when the study has at least three trials."""
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) < 3:
        print(
            f"{model_id}: {len(completed)} completed trials - "
            "history plot skipped (need >= 3)."
        )
        return None

    optuna_dir = Path(out_dir) / "figures" / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.set_title(f"Optuna history - {model_id}")
    path = optuna_dir / "optuna_history.png"
    ax.figure.savefig(path, dpi=160, bbox_inches="tight")
    show_or_close(ax.figure, show_plots)
    return path
