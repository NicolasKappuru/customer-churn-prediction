"""Load and use the trained Telco churn model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "w1_supervized_learning"
    / "checkpoints"
    / "xgboost_telco.pkl"
)


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and validate the model artifact created by supervised learning."""
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

    model = artifact["model"]
    if not callable(getattr(model, "predict", None)):
        raise ValueError("The saved model does not implement predict().")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("The saved model does not implement predict_proba().")

    feature_names = artifact["feature_names"]
    if (
        not isinstance(feature_names, (list, tuple))
        or not feature_names
        or not all(isinstance(name, str) and name for name in feature_names)
        or len(set(feature_names)) != len(feature_names)
    ):
        raise ValueError("Model artifact feature_names must contain unique names.")

    try:
        threshold = float(artifact["threshold"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Model artifact threshold must be a number between 0 and 1."
        ) from error
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Model artifact threshold must be a number between 0 and 1.")

    return artifact


class SupervisedModel:
    """Predict churn for a single customer already processed into model features."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path

        self.artifact = load_model_artifact(self.model_path)
        self.model = self.artifact["model"]
        self.threshold = float(self.artifact["threshold"])
        self.feature_names = tuple(self.artifact["feature_names"])

    def predict_probability(self, customer_data: Any) -> float:
        """Return the model's probability of churn for one processed customer."""
        features = self._prepare_input(customer_data)
        probabilities = np.asarray(self.model.predict_proba(features))
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError("The model must return probabilities for one customer.")

        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            churn_columns = np.flatnonzero(np.asarray(classes) == 1)
            if len(churn_columns) != 1:
                raise ValueError("The saved model must contain class 1 for Churn.")
            churn_column = int(churn_columns[0])
        else:
            churn_column = 1

        if churn_column >= probabilities.shape[1]:
            raise ValueError("The model returned an unexpected probability shape.")
        return float(probabilities[0, churn_column])

    def predict(self, customer_data: Any) -> str:
        """Return ``'Churn'`` or ``'No churn'`` using the saved threshold."""
        probability = self.predict_probability(customer_data)
        return "Churn" if probability >= self.threshold else "No churn"

    def _prepare_input(self, customer_data: Any) -> np.ndarray:
        if hasattr(customer_data, "columns") and hasattr(customer_data, "to_numpy"):
            input_columns = list(customer_data.columns)
            if (
                len(input_columns) != len(self.feature_names)
                or set(input_columns) != set(self.feature_names)
            ):
                raise ValueError(
                    "Processed customer columns do not match the model feature names."
                )
            try:
                    features = customer_data.loc[:, list(self.feature_names)].to_numpy(
                    dtype=np.float32
                )
            except (TypeError, ValueError) as error:
                raise ValueError("Processed customer features must be numeric.") from error
        else:
            try:
                features = np.asarray(customer_data, dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError("Processed customer features must be numeric.") from error

        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape != (1, len(self.feature_names)):
            raise ValueError(
                "Expected one processed customer with "
                f"{len(self.feature_names)} features; received shape {features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError("Processed customer features must all be finite.")
        return features


# Preserve the module-level names used by existing environment imports.
MODEL_ARTIFACT = load_model_artifact()
model = MODEL_ARTIFACT["model"]
threshold = float(MODEL_ARTIFACT["threshold"])
feature_names = tuple(MODEL_ARTIFACT["feature_names"])
