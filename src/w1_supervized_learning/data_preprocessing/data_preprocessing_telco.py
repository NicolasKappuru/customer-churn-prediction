# This preprocessing is for the Telco dataset.
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()


class PreprocessingTelco:
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
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
    ]
    continuous_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AvgMonthlyCharges",
    ]
    random_state = 42

    def __init__(self, dataset_path=None, split_output_path=None):
        project_root = Path(__file__).resolve().parents[3]
        self.dataset_path = self.resolve_dataset_path(dataset_path)
        self.split_output_path = (
            Path(split_output_path)
            if split_output_path is not None
            else project_root / "data" / "telco_preprocessed_splits.npz"
        )
        self.telco_churn_df = pd.read_csv(self.dataset_path)

    def resolve_dataset_path(self, dataset_path=None):
        project_root = Path(__file__).resolve().parents[3]
        default_path = (
            project_root / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

        if dataset_path is not None:
            resolved_path = Path(dataset_path).expanduser()
            if not resolved_path.is_absolute():
                resolved_path = Path.cwd() / resolved_path
        else:
            configured_path = os.getenv("PATH_DATASET_TELCO", "").strip()
            resolved_path = Path(configured_path).expanduser() if configured_path else default_path
            if not resolved_path.is_absolute():
                resolved_path = Path.cwd() / resolved_path
            if not resolved_path.is_file() and dataset_path is None:
                resolved_path = default_path

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Telco dataset not found: {resolved_path}")
        return resolved_path

    def preprocess(self):
        """Clean, engineer, split, encode, scale, and save the Telco data.

        No augmentation is applied: synthetic or image-style transformations are
        not appropriate for these customer-level tabular records.
        """
        self.select_features()
        self.drop_nan()
        self.make_feature_engineering()
        return self.split_dataset()

    def select_features(self):
        columns = [
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
            "Churn",
        ]
        self.telco_churn_df = self.telco_churn_df[columns].copy()

    def encode(self, X_train, X_validation, X_test):
        """Fit categorical encoding on training data and transform all splits."""
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        encoded_train = self.encoder.fit_transform(
            X_train[self.categorical_features]
        )
        encoded_validation = self.encoder.transform(
            X_validation[self.categorical_features]
        )
        encoded_test = self.encoder.transform(X_test[self.categorical_features])
        encoded_feature_names = self.encoder.get_feature_names_out(
            self.categorical_features
        )

        self.scaler = StandardScaler()
        self.scaler.fit(X_train[self.continuous_features])

        def transform_split(features, encoded_features):
            numeric_features = features.drop(
                columns=self.categorical_features
            ).copy()
            numeric_features[self.continuous_features] = numeric_features[
                self.continuous_features
            ].astype(float)
            numeric_features.loc[:, self.continuous_features] = (
                self.scaler.transform(features[self.continuous_features])
            )
            categorical_features = pd.DataFrame(
                encoded_features,
                columns=encoded_feature_names,
                index=features.index,
            )
            return pd.concat([numeric_features, categorical_features], axis=1)

        encoded_splits = (
            transform_split(X_train, encoded_train),
            transform_split(X_validation, encoded_validation),
            transform_split(X_test, encoded_test),
        )
        self.feature_names = encoded_splits[0].columns.tolist()
        return tuple(split.to_numpy() for split in encoded_splits)

    def drop_nan(self):
        """Impute blank TotalCharges for zero-tenure customers; drop other invalid rows."""
        self.telco_churn_df["TotalCharges"] = pd.to_numeric(
            self.telco_churn_df["TotalCharges"], errors="coerce"
        )
        zero_tenure_missing_charges = (
            self.telco_churn_df["TotalCharges"].isna()
            & self.telco_churn_df["tenure"].eq(0)
        )
        self.total_charges_imputed = int(zero_tenure_missing_charges.sum())
        self.telco_churn_df.loc[zero_tenure_missing_charges, "TotalCharges"] = 0

        self.telco_churn_df["Churn"] = self.telco_churn_df["Churn"].map(
            {"Yes": 1, "No": 0}
        )
        rows_before_cleaning = len(self.telco_churn_df)
        self.telco_churn_df = self.telco_churn_df.dropna().copy()
        self.rows_dropped = rows_before_cleaning - len(self.telco_churn_df)

    def make_feature_engineering(self):
        self.telco_churn_df["NewCustomer"] = (
            self.telco_churn_df["tenure"].eq(0).astype(int)
        )
        self.telco_churn_df["AvgMonthlyCharges"] = np.where(
            self.telco_churn_df["tenure"] > 0,
            self.telco_churn_df["TotalCharges"] / self.telco_churn_df["tenure"],
            0,
        )

    def split_dataset(self):
        """Create stratified 70/15/15 splits, then fit transforms on train only.

        Returns the original four train/test values for compatibility. Validation
        data is exposed as ``X_validation`` and ``y_validation`` and all three
        splits are saved to ``data/telco_preprocessed_splits.npz`` by default.
        """
        X = self.telco_churn_df.drop(columns="Churn")
        y = self.telco_churn_df["Churn"].astype(int)

        X_train_raw, X_remaining, y_train, y_remaining = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=self.random_state,
            stratify=y,
        )
        X_validation_raw, X_test_raw, y_validation, y_test = train_test_split(
            X_remaining,
            y_remaining,
            test_size=0.50,
            random_state=self.random_state,
            stratify=y_remaining,
        )

        X_train, X_validation, X_test = self.encode(
            X_train_raw,
            X_validation_raw,
            X_test_raw,
        )
        self.X_validation = X_validation
        self.y_validation = y_validation.to_numpy()

        self.split_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.split_output_path,
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train.to_numpy(),
            y_validation=self.y_validation,
            y_test=y_test.to_numpy(),
            feature_names=np.asarray(self.feature_names, dtype=str),
        )

        self.split_sizes = {
            "train": len(y_train),
            "validation": len(y_validation),
            "test": len(y_test),
        }
        return X_train, X_test, y_train, y_test
