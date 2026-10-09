"""Prepara las 19 características Telco usadas por el modelo supervisado.

CustomerID es un identificador y no forma parte de RAW_FEATURES ni de las
columnas transformadas. Si aparece como columna adicional en la entrada, se
descarta antes de construir el vector que se entrega al modelo.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIRECTORY = PROJECT_ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from w1_supervized_learning.data_preprocessing.data_preprocessing_telco import (  # noqa: E402
    PreprocessingTelco,
)
from w3_reinforcement_leraning.environment.supervised_model.supervised_model import (  # noqa: E402
    feature_names as model_feature_names,
)

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data_preprocessed" / "telco_preprocessed.csv"
RAW_FEATURES = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)


class TelcoPreprocessor:
    """Fit once on the Telco training split and transform records consistently."""

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        with tempfile.TemporaryDirectory(prefix="telco-preprocessing-") as temp_dir:
            self.preprocessing = PreprocessingTelco(
                dataset_path=dataset_path,
                split_output_path=Path(temp_dir) / "splits.npz",
            )
            self.preprocessing.preprocess()

        self.feature_names = tuple(self.preprocessing.feature_names)
        if self.feature_names != model_feature_names:
            raise ValueError(
                "Preprocessing features do not match the trained model's feature order."
            )

    def transform(
        self, records: Mapping[str, object] | pd.Series | pd.DataFrame
    ) -> pd.DataFrame:
        """Transforma clientes usando solo las 19 características del experimento.

        Las columnas adicionales, incluido CustomerID, no se copian al resultado
        ni se envían al modelo. La salida conserva únicamente el orden de
        características guardado en el checkpoint supervisado.
        """
        if isinstance(records, pd.Series):
            raw = records.to_frame().T
        elif isinstance(records, pd.DataFrame):
            raw = records.copy()
        elif isinstance(records, Mapping):
            raw = pd.DataFrame([records])
        else:
            raise TypeError(
                "records must be a mapping, pandas Series, or pandas DataFrame."
            )

        missing_features = [name for name in RAW_FEATURES if name not in raw.columns]
        if missing_features:
            raise ValueError(
                "Customer data is missing required features: "
                + ", ".join(missing_features)
            )

        raw = raw.loc[:, RAW_FEATURES].copy()
        for column in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

        missing_total_charges = raw["TotalCharges"].isna() & raw["tenure"].eq(0)
        raw.loc[missing_total_charges, "TotalCharges"] = 0
        if raw.loc[:, RAW_FEATURES].isna().any().any():
            raise ValueError("Customer data contains missing or invalid feature values.")

        raw["NewCustomer"] = raw["tenure"].eq(0).astype(int)
        raw["AvgMonthlyCharges"] = np.where(
            raw["tenure"] > 0,
            raw["TotalCharges"] / raw["tenure"],
            0,
        )

        categorical_features = self.preprocessing.categorical_features
        continuous_features = self.preprocessing.continuous_features
        encoded = self.preprocessing.encoder.transform(raw[categorical_features])

        numeric = raw.drop(columns=categorical_features).copy()
        numeric[continuous_features] = numeric[continuous_features].astype(float)
        numeric.loc[:, continuous_features] = self.preprocessing.scaler.transform(
            numeric[continuous_features]
        )

        encoded_frame = pd.DataFrame(
            encoded,
            columns=self.preprocessing.encoder.get_feature_names_out(
                categorical_features
            ),
            index=raw.index,
        )
        transformed = pd.concat([numeric, encoded_frame], axis=1)

        if tuple(transformed.columns) != self.feature_names:
            raise ValueError(
                "Preprocessed features do not match the trained model's feature order."
            )
        return transformed.astype(np.float32)


_DEFAULT_PREPROCESSOR: TelcoPreprocessor | None = None
_CUSTOM_PREPROCESSORS: dict[str, TelcoPreprocessor] = {}


def _get_preprocessor(
    dataset_path: str | Path | None = None,
) -> TelcoPreprocessor:
    global _DEFAULT_PREPROCESSOR

    if dataset_path is None:
        if _DEFAULT_PREPROCESSOR is None:
            _DEFAULT_PREPROCESSOR = TelcoPreprocessor()
        return _DEFAULT_PREPROCESSOR

    cache_key = str(Path(dataset_path).expanduser().resolve())
    if cache_key not in _CUSTOM_PREPROCESSORS:
        _CUSTOM_PREPROCESSORS[cache_key] = TelcoPreprocessor(dataset_path=dataset_path)
    return _CUSTOM_PREPROCESSORS[cache_key]


def preprocess_customer(
    customer: Mapping[str, object] | pd.Series | pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
) -> pd.DataFrame:
    """Transform raw Telco customer features into the model's expected columns."""
    return _get_preprocessor(dataset_path).transform(customer)


def preprocess_telco_dataset(
    dataset_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Preprocess the full Telco dataset, save it, and return the transformed data."""
    preprocessor = _get_preprocessor(dataset_path)
    raw_data = preprocessor.preprocessing.telco_churn_df
    transformed = preprocessor.transform(raw_data.drop(columns="Churn"))
    transformed["Churn"] = raw_data["Churn"].astype(int).to_numpy()

    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(destination, index=False)
    return transformed


if __name__ == "__main__":
    saved_data = preprocess_telco_dataset()
    print(f"Preprocessed {len(saved_data)} records into {DEFAULT_OUTPUT_PATH}")