"""Load and use the trained KKBox churn model (XGB_reweight, 48 features)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "workshops"
    / "w1"
    / "artifacts"
    / "sherlock_w1_manual_reweighting"
    / "models"
    / "XGB_reweight.joblib"
)
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "workshops"
    / "w1"
    / "artifacts"
    / "sherlock_w1_manual_reweighting"
    / "run_config.json"
)
_FALLBACK_THRESHOLD = 0.5

# 48 input features: 3 categorical + 45 numerical (11 low-signal features dropped from the 59)
_CAT_FEATURES: tuple[str, ...] = ("city", "registered_via", "last_payment_method_id")
_NUM_59: tuple[str, ...] = (
    "bd", "bd_missing", "tenure_days", "registration_year", "registration_month",
    "has_member_record", "is_cancel_change_flag", "is_auto_renew_change_flag",
    "transaction_count", "average_amount_paid", "average_amount_charged",
    "average_plan", "n_payment_methods", "n_plans", "n_list_prices", "n_amounts_paid",
    "duration", "plan_coverage_ratio", "sum_plan_days", "sum_list_price",
    "sum_amount_paid", "more_than_30", "autorenew_not_cancel", "not_autorenew_cancel",
    "max_plan_days", "last_plan_days", "last_is_auto_renew", "last_is_cancel",
    "last_discount", "days_to_expire", "expire_dayofweek", "days_since_last_transaction",
    "difference", "has_transactions",
    "sum_last_month_unq", "sum_last_month_total_secs", "mean_last_month_total_secs",
    "last_login", "last_sum_total_songs", "last_ratio_100", "last_ratio_25",
    "sum_last_second_month_total_secs", "last_second_login",
    "last_second_sum_total_songs", "last_second_ratio_100", "last_second_ratio_25",
    "last_third_sum_total_songs", "last_third_ratio_100", "last_third_ratio_25",
    "login_difference", "trend_songs", "trend_secs", "trend_ratio100", "trend_ratio25",
    "days_since_last_log", "has_listening_logs",
)
_DROP_FEATURES: frozenset[str] = frozenset({
    "has_transactions", "last_is_cancel", "autorenew_not_cancel", "not_autorenew_cancel",
    "last_is_auto_renew", "is_auto_renew_change_flag", "is_cancel_change_flag",
    "last_discount", "bd_missing", "expire_dayofweek", "n_amounts_paid",
})
_NUM_48: tuple[str, ...] = tuple(f for f in _NUM_59 if f not in _DROP_FEATURES)
FEATURE_NAMES_48: tuple[str, ...] = _CAT_FEATURES + _NUM_48


class SupervisedModel:
    """Predict churn for a single KKBox user using XGB_reweight (48 features).

    The artifact is a dict with keys 'preprocessor' (ColumnTransformer) and
    'model' (XGBClassifier).  Input must be a one-row DataFrame with the 48
    expected feature columns.
    """

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        artifact_path = Path(artifact_path).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = PROJECT_ROOT / artifact_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, dict) or "preprocessor" not in artifact or "model" not in artifact:
            raise ValueError(
                f"Unexpected artifact format at {artifact_path}. "
                "Expected dict with keys 'preprocessor' and 'model'."
            )

        self.preprocessor = artifact["preprocessor"]
        self.model = artifact["model"]
        self.feature_names: tuple[str, ...] = FEATURE_NAMES_48

        config_path = Path(config_path).expanduser()
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            self.threshold = float(
                config.get("thresholds", {}).get("XGB_reweight", _FALLBACK_THRESHOLD)
            )
        else:
            self.threshold = _FALLBACK_THRESHOLD

    def predict_probability(self, customer_data: Any) -> float:
        """Return the model's churn probability for one processed customer."""
        df = self._prepare_input(customer_data)
        transformed = self.preprocessor.transform(df)
        probabilities = np.asarray(self.model.predict_proba(transformed))
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

        return float(probabilities[0, churn_column])

    def predict(self, customer_data: Any) -> str:
        """Return 'Churn' or 'No churn' using the saved threshold."""
        probability = self.predict_probability(customer_data)
        return "Churn" if probability >= self.threshold else "No churn"

    def _prepare_input(self, customer_data: Any) -> pd.DataFrame:
        if hasattr(customer_data, "columns"):
            missing = set(self.feature_names) - set(customer_data.columns)
            if missing:
                raise ValueError(
                    f"Input DataFrame is missing expected features: {sorted(missing)}"
                )
            return customer_data[list(self.feature_names)]

        try:
            arr = np.asarray(customer_data, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("customer_data must be a DataFrame or numeric array.") from error

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (1, len(self.feature_names)):
            raise ValueError(
                f"Expected shape (1, {len(self.feature_names)}); received {arr.shape}."
            )
        return pd.DataFrame(arr, columns=list(self.feature_names))
