"""Classification report for the single w2 model."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from ..constants import CLASS_LABELS


def save_classification_report(y_true, proba, threshold, model_id, out_dir):
    """Write the per-class report as JSON and text."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = (np.asarray(proba) >= threshold).astype(int)

    report_dict = classification_report(
        y_true,
        pred,
        labels=[0, 1],
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        pred,
        labels=[0, 1],
        target_names=CLASS_LABELS,
        zero_division=0,
    )

    json_path = out_dir / f"report_{model_id}.json"
    text_path = out_dir / f"report_{model_id}.txt"
    json_path.write_text(json.dumps(report_dict, indent=2))
    text_path.write_text(report_text)
    return report_dict, json_path, text_path
