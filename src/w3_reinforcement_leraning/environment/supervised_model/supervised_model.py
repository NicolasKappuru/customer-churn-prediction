"""Load the trained Telco XGBoost model for the reinforcement-learning environment."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "w1_supervized_learning"
    / "checkpoints"
    / "xgboost_telco.pkl"
)


def load_model_artifact(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load and validate a model artifact saved by the supervised-learning project."""
    artifact_path = Path(model_path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = PROJECT_ROOT / artifact_path
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Supervised model artifact not found: {artifact_path}")

    with artifact_path.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)

    if not isinstance(artifact, dict):
        raise ValueError(f"Invalid model artifact format: {artifact_path}")

    required_keys = {"model", "threshold", "feature_names"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Model artifact is missing required keys: {missing}")

    if not callable(getattr(artifact["model"], "predict_proba", None)):
        raise ValueError("The saved model does not implement predict_proba().")
    if not isinstance(artifact["feature_names"], (list, tuple)):
        raise ValueError("Model artifact feature_names must be a list or tuple.")

    return artifact


MODEL_ARTIFACT = load_model_artifact()
model = MODEL_ARTIFACT["model"]
threshold = float(MODEL_ARTIFACT["threshold"])
feature_names = tuple(MODEL_ARTIFACT["feature_names"])