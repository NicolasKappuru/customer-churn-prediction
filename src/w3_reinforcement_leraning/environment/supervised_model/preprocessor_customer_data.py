"""Prepare raw Telco customer data for the trained churn model."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "src"
    / "w1_supervized_learning"
    / "checkpoints"
    / "telco_preprocessor.pkl"
)


class PreprocessorCustomerData:
    """Load fitted preprocessing objects and transform one raw Telco customer."""

    def __init__(
        self,
        preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        self.preprocessor_path = Path(preprocessor_path).expanduser()
        if not self.preprocessor_path.is_absolute():
            self.preprocessor_path = PROJECT_ROOT / self.preprocessor_path
        if not self.preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Customer preprocessor artifact not found: {self.preprocessor_path}"
            )

        with self.preprocessor_path.open("rb") as artifact_file:
            artifact = pickle.load(artifact_file)
        if not isinstance(artifact, dict):
            raise ValueError(
                f"Invalid customer preprocessor artifact: {self.preprocessor_path}"
            )

        required_keys = {
            "encoder",
            "scaler",
            "categorical_features",
            "continuous_features",
            "feature_names",
        }
        missing_keys = required_keys.difference(artifact)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"Preprocessor artifact is missing required keys: {missing}"
            )

        self.encoder = artifact["encoder"]
        self.scaler = artifact["scaler"]
        self.categorical_features = self._validate_names(
            artifact["categorical_features"], "categorical_features"
        )
        self.continuous_features = self._validate_names(
            artifact["continuous_features"], "continuous_features"
        )
        self.feature_names = self._validate_names(
            artifact["feature_names"], "feature_names"
        )

        if set(self.categorical_features).intersection(self.continuous_features):
            raise ValueError(
                "Preprocessor categorical_features and continuous_features overlap."
            )
        if "AvgMonthlyCharges" not in self.continuous_features:
            raise ValueError(
                "Preprocessor artifact must include AvgMonthlyCharges as a "
                "continuous feature."
            )
        if not callable(getattr(self.encoder, "transform", None)) or not callable(
            getattr(self.encoder, "get_feature_names_out", None)
        ):
            raise ValueError("The saved encoder does not implement the required methods.")
        if not callable(getattr(self.scaler, "transform", None)):
            raise ValueError("The saved scaler does not implement transform().")

        self.raw_numeric_features = (
            "SeniorCitizen",
            *(name for name in self.continuous_features if name != "AvgMonthlyCharges"),
        )
        if len(set(self.raw_numeric_features)) != len(self.raw_numeric_features):
            raise ValueError("Preprocessor artifact contains duplicate numeric features.")
        self.required_customer_features = (
            *self.categorical_features,
            *self.raw_numeric_features,
        )

        encoder_feature_names = list(
            self.encoder.get_feature_names_out(self.categorical_features)
        )
        expected_features = {
            *self.raw_numeric_features,
            "NewCustomer",
            "AvgMonthlyCharges",
            *encoder_feature_names,
        }
        if set(self.feature_names) != expected_features:
            raise ValueError(
                "Preprocessor feature_names do not match its encoder and "
                "numeric feature configuration."
            )

    def preprocess_customer(self, customer: Mapping[str, Any]) -> pd.DataFrame:
        """Return one raw customer's encoded and scaled features in model order."""

        print(f"Features de entrada: {len(customer)}")

        if not isinstance(customer, Mapping):
            raise TypeError("customer must be a mapping of feature names to values.")

        missing_features = [
            name for name in self.required_customer_features if name not in customer
        ]
        if missing_features:
            raise ValueError(
                "Customer data is missing required features: "
                + ", ".join(missing_features)
            )

        customer_data = pd.DataFrame(
            [{name: customer[name] for name in self.required_customer_features}]
        )
        if customer_data.loc[:, list(self.categorical_features)].isna().any().any():
            raise ValueError("Customer categorical features cannot be missing.")

        for feature in self.raw_numeric_features:
            customer_data[feature] = pd.to_numeric(
                customer_data[feature], errors="coerce"
            )

        if pd.isna(customer_data.loc[0, "TotalCharges"]) and customer_data.loc[
            0, "tenure"
        ] == 0:
            customer_data.loc[0, "TotalCharges"] = 0

        numeric_values = customer_data.loc[:, list(self.raw_numeric_features)].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(numeric_values).all():
            raise ValueError(
                "Customer numeric features must be finite; TotalCharges may be "
                "missing only when tenure is zero."
            )

        customer_data["NewCustomer"] = customer_data["tenure"].eq(0).astype(int)
        customer_data["AvgMonthlyCharges"] = np.where(
            customer_data["tenure"] > 0,
            customer_data["TotalCharges"] / customer_data["tenure"],
            0,
        )

        encoded_values = self.encoder.transform(
            customer_data.loc[:, list(self.categorical_features)]
        )
        encoded_feature_names = self.encoder.get_feature_names_out(
            self.categorical_features
        )

        numeric_features = customer_data.drop(
            columns=list(self.categorical_features)
        ).copy()
        continuous_features = numeric_features.loc[
            :, list(self.continuous_features)
        ].astype(float)
        scaled_continuous = pd.DataFrame(
            self.scaler.transform(continuous_features),
            columns=self.continuous_features,
            index=numeric_features.index,
        )
        numeric_features = numeric_features.drop(
            columns=list(self.continuous_features)
        ).join(scaled_continuous)
        encoded_features = pd.DataFrame(
            encoded_values,
            columns=encoded_feature_names,
            index=customer_data.index,
        )
        transformed_customer = pd.concat(
            [numeric_features, encoded_features], axis=1
        )
        transformed_columns = list(transformed_customer.columns)
        if (
            len(transformed_columns) != len(self.feature_names)
            or set(transformed_columns) != set(self.feature_names)
        ):
            raise ValueError(
                "Processed customer features do not match the saved feature names."
            )

        processed_customer = transformed_customer.loc[:, self.feature_names].copy()
        if not np.isfinite(processed_customer.to_numpy(dtype=np.float32)).all():
            raise ValueError("Processed customer features must all be finite.")

        print(f"Features procesados: {processed_customer.shape[1]}")
        return processed_customer

    @staticmethod
    def _validate_names(value: Any, name: str) -> tuple[str, ...]:
        if (
            not isinstance(value, (list, tuple))
            or not value
            or not all(isinstance(item, str) and item for item in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError(f"Preprocessor {name} must contain unique names.")
        return tuple(value)