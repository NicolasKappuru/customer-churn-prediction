"""Classification reports per model (JSON + text)."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from ..constants import CLASS_LABELS, MODEL_IDS


def save_classification_report(y_true, proba, threshold, model_id, out_dir):
    """Full classification_report (per class + macro/weighted) as JSON and text."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = (np.asarray(proba) >= threshold).astype(int)

    rep_dict = classification_report(
        y_true,
        pred,
        labels=[0, 1],
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    rep_text = classification_report(
        y_true,
        pred,
        labels=[0, 1],
        target_names=CLASS_LABELS,
        zero_division=0,
    )

    with (out_dir / f"report_{model_id}.json").open("w") as f:
        json.dump(rep_dict, f, indent=2)

    (out_dir / f"report_{model_id}.txt").write_text(rep_text)
    return rep_dict


def save_classification_reports(y_test, registry, thresholds, out_dir):
    """Write a JSON and text classification report per model."""
    report_dir = Path(out_dir) / "metrics" / "classification_reports"
    for model_id in MODEL_IDS:
        save_classification_report(
            y_test,
            registry.proba[model_id]["test"],
            thresholds[model_id],
            model_id,
            report_dir,
        )
    return report_dir
