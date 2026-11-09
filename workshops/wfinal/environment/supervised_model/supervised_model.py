"""Load and use the trained KKBox churn model (M3 XGBoost)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "workshops"
    / "w1"
    / "artifacts"
    / "sherlock_w1"
    / "models"
    / "development"
    / "M3.joblib"
)
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "workshops"
    / "w1"
    / "artifacts"
    / "sherlock_w1"
    / "run_config.json"
)
_FALLBACK_THRESHOLD = 0.88


class SupervisedModel:
    """Predict churn for a single KKBox user already processed into model features."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        self.model = joblib.load(model_path)
        self.feature_names: tuple[str, ...] = tuple(
            str(f) for f in self.model.feature_names_in_
        )

        config_path = Path(config_path).expanduser()
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            self.threshold = float(
                config.get("thresholds", {}).get("M3", _FALLBACK_THRESHOLD)
            )
        else:
            self.threshold = _FALLBACK_THRESHOLD

    def predict_probability(self, customer_data: Any) -> float:
        """Return the model's churn probability for one processed customer."""
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
            if set(input_columns) != set(self.feature_names):
                raise ValueError(
                    "Processed customer columns do not match the model feature names."
                )
            try:
                features = customer_data.loc[:, list(self.feature_names)].to_numpy(
                    dtype=np.float32
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Processed customer features must be numeric."
                ) from error
        else:
            try:
                features = np.asarray(customer_data, dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Processed customer features must be numeric."
                ) from error

        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape != (1, len(self.feature_names)):
            raise ValueError(
                f"Expected one processed customer with {len(self.feature_names)} features; "
                f"received shape {features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError("Processed customer features must all be finite.")
        return features
