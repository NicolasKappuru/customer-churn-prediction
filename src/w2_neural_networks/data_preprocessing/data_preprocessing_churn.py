# This preprocessing is for the customer churn dataset.
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

load_dotenv()


class PreprocessingChurn:
    categorical_features = [
        "Subscription Type",
        "Contract Length",
    ]
    continuous_features = [
        "Age",
        "Gender",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction",
        "AvgMonthlySpend",
        "SpendPerUsage",
        "SupportCallRate",
        "PaymentDelayRate",
        "EngagementRate",
        "IsNewCustomer",
        "HasPaymentDelay",
    ]
    raw_columns = [
        "CustomerID",
        "Age",
        "Gender",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Subscription Type",
        "Contract Length",
        "Total Spend",
        "Last Interaction",
        "Churn",
    ]
    random_state = 42

    def __init__(self, dataset_path=None, split_output_path=None):
        project_root = Path(__file__).resolve().parents[3]
        self.dataset_path = self.resolve_dataset_path(dataset_path)
        self.split_output_path = (
            Path(split_output_path)
            if split_output_path is not None
            else project_root / "data" / "churn_preprocessed_splits.npz"
        )
        self.churn_df = pd.read_csv(self.dataset_path)

    def resolve_dataset_path(self, dataset_path=None):
        project_root = Path(__file__).resolve().parents[3]
        default_path = project_root / "data" / "customer_churn_dataset.csv"

        if dataset_path is not None:
            resolved_path = Path(dataset_path).expanduser()
            if not resolved_path.is_absolute():
                resolved_path = Path.cwd() / resolved_path
        else:
            configured_path = os.getenv("PATH_DATASET_CHURN", "").strip()
            resolved_path = Path(configured_path).expanduser() if configured_path else default_path
            if not resolved_path.is_absolute():
                resolved_path = Path.cwd() / resolved_path
            if not resolved_path.is_file() and dataset_path is None:
                resolved_path = default_path

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Churn dataset not found: {resolved_path}")
        return resolved_path

    def preprocess(self):
        """Clean, engineer, split, encode, scale, and save the churn data.

        No augmentation is applied: synthetic or image-style transformations are
        not appropriate for these customer-level tabular records.
        """
        self.select_features()
        self.drop_nan()
        self.make_feature_engineering()
        return self.split_dataset()

    def select_features(self):
        """Keep the modelling columns and drop the CustomerID identifier.

        ``CustomerID`` is a sequential row id that correlates with the target
        (0.53), so keeping it would leak information instead of describing the
        customer behaviour.
        """
        missing_columns = [
            column for column in self.raw_columns if column not in self.churn_df.columns
        ]
        if missing_columns:
            raise KeyError(f"Missing columns in the churn dataset: {missing_columns}")

        self.churn_df = self.churn_df[self.raw_columns].copy()

    def drop_nan(self):
        """Coerce numeric columns, validate the target, and drop invalid rows."""
        numeric_columns = [
            column
            for column in self.raw_columns
            if column not in ("Gender", "Subscription Type", "Contract Length", "Churn")
        ]
        for column in numeric_columns:
            self.churn_df[column] = pd.to_numeric(self.churn_df[column], errors="coerce")

        self.churn_df["Gender"] = self.churn_df["Gender"].astype("string").str.strip()
        self.churn_df["Subscription Type"] = (
            self.churn_df["Subscription Type"].astype("string").str.strip()
        )
        self.churn_df["Contract Length"] = (
            self.churn_df["Contract Length"].astype("string").str.strip()
        )

        invalid_target = ~self.churn_df["Churn"].isin([0, 1])
        self.invalid_target_rows = int(invalid_target.sum())
        rows_before_cleaning = len(self.churn_df)
        self.churn_df = (
            self.churn_df.loc[~invalid_target]
            .drop_duplicates(subset="CustomerID", keep="first")
            .dropna()
            .copy()
        )
        self.rows_dropped = rows_before_cleaning - len(self.churn_df)

        self.churn_df = self.churn_df.drop(columns="CustomerID")

    def make_feature_engineering(self):
        """Encode Gender as binary and add tenure-normalised behaviour features."""
        self.churn_df["Gender"] = self.churn_df["Gender"].map(
            {"Male": 1, "Female": 0}
        )

        tenure = self.churn_df["Tenure"].replace(0, np.nan)
        self.churn_df["AvgMonthlySpend"] = (
            self.churn_df["Total Spend"] / tenure
        ).fillna(0.0)
        self.churn_df["SpendPerUsage"] = (
            self.churn_df["Total Spend"] / self.churn_df["Usage Frequency"]
        ).fillna(0.0)
        self.churn_df["SupportCallRate"] = (
            self.churn_df["Support Calls"] / tenure
        ).fillna(0.0)
        self.churn_df["PaymentDelayRate"] = (
            self.churn_df["Payment Delay"] / tenure
        ).fillna(0.0)
        self.churn_df["EngagementRate"] = (
            self.churn_df["Last Interaction"] / tenure
        ).fillna(0.0)
        self.churn_df["IsNewCustomer"] = self.churn_df["Tenure"].le(12).astype(int)
        self.churn_df["HasPaymentDelay"] = self.churn_df["Payment Delay"].gt(0).astype(int)

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

    def split_dataset(self):
        """Create stratified 70/15/15 splits, then fit transforms on train only.

        Returns the original four train/test values for compatibility. Validation
        data is exposed as ``X_validation`` and ``y_validation`` and all three
        splits are saved to ``data/churn_preprocessed_splits.npz`` by default.
        """
        X = self.churn_df.drop(columns="Churn")
        y = self.churn_df["Churn"].astype(int)

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
