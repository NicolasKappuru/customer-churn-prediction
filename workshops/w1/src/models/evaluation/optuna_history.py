"""Optuna history diagnostics."""

import optuna

from ..constants import MODEL_IDS
from .figure_utils import show_or_close


def plot_optuna_histories(get_study, show_plots=True):
    """Show histories only for studies with at least three completed trials."""
    for model_id in MODEL_IDS:
        try:
            study = get_study(model_id)
        except Exception:
            continue
        completed = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < 3:
            print(
                f"{model_id}: {len(completed)} completed trials — "
                "history plot skipped (need >= 3)."
            )
            continue
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        ax.set_title(f"Optuna history — {model_id}")
        show_or_close(ax.figure, show_plots)
