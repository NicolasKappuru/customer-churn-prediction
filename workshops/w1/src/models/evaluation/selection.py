"""Choose thresholds and a winner using February tune predictions."""

import json
from pathlib import Path

from sklearn.metrics import log_loss

from ..constants import MODEL_IDS
from .metrics import select_threshold


def select_on_tune(y_tune, registry, contract_hash, out_dir):
    """Freeze tune decisions and save their manifest before test scoring."""
    thresholds = {
        model_id: select_threshold(y_tune, registry.proba[model_id]["tune"])
        for model_id in MODEL_IDS
    }
    tune_losses = {
        model_id: log_loss(
            y_tune, registry.proba[model_id]["tune"], labels=[0, 1]
        )
        for model_id in MODEL_IDS
    }
    winner = min(MODEL_IDS, key=tune_losses.__getitem__)
    selection = {
        "contract_hash": contract_hash,
        "tune_log_loss": {
            model_id: round(float(loss), 6)
            for model_id, loss in tune_losses.items()
        },
        "thresholds": {
            model_id: round(float(threshold), 4)
            for model_id, threshold in thresholds.items()
        },
        "primary_winner": winner,
    }
    path = Path(out_dir) / "selection_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2))
    print("primary winner (tune):", winner)
    print(
        "thresholds:",
        {model_id: round(value, 3) for model_id, value in thresholds.items()},
    )
    return thresholds, selection, winner
