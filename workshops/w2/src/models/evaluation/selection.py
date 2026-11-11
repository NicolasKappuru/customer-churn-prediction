"""Freeze the tune threshold before test evaluation."""

import json
from pathlib import Path


def save_selection_manifest(
    model_id,
    threshold,
    tune_log_loss,
    contract_hash,
    out_dir,
):
    """Write the single-model selection manifest."""
    path = Path(out_dir) / "selection_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "contract_hash": contract_hash,
        "threshold": round(float(threshold), 4),
        "tune_log_loss": round(float(tune_log_loss), 6),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
