# This preprocessing reproduces churn_prediction_MLP.ipynb, kept as a class.
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()


class PreprocessingTelcoV2:
    """Telco preprocessing, following ``churn_prediction_MLP.ipynb`` step by step.

    Unlike :class:`PreprocessingTelco` (w1), which one-hot encodes with scikit-learn
    and z-scores on training rows only, this version mirrors the notebook exactly:

    1. ``Churn`` mapped ``Yes -> 1`` / ``No -> 0``.
    2. ``TotalCharges`` blanks turned into NaN and cast to float, then the 11 rows
       they belonged to are dropped rather than imputed.
    3. ``MonthlyCharges`` and ``TotalCharges`` log-transformed, then all three
       continuous columns z-scored. ``tenure`` is **not** log-transformed.
    4. ``SeniorCitizen`` turned into the strings ``'Yes'`` / ``'No'`` so it is
       one-hot encoded alongside the other categoricals instead of being used raw.
    5. One-hot encoding with ``pd.get_dummies`` over every remaining column with
       fewer than five unique values, producing the notebook's 46 features.

    The output column set and its order match the notebook's ``df_trans`` exactly,
    which is what makes the two notebooks comparable.

    **Data leakage.** The notebook standardises before splitting, so its z-scores
    use statistics from the full dataset. That is reproduced here by default, via
    ``fit_scalers_on_train_only = False``. Set it to ``True`` to fit the mean and
    standard deviation on training rows only, which is the correct thing to do but
    changes the reported numbers slightly.
    """

    continuous_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    log_transform_features = [
        "MonthlyCharges",
        "TotalCharges",
    ]
    excluded_from_dummies = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    raw_columns = [
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
    random_state = 23
    test_size = 0.20
    validation_size = 0.15
    max_dummy_cardinality = 5
    fit_scalers_on_train_only = False

    def __init__(self, dataset_path=None, split_output_path=None):
        project_root = Path(__file__).resolve().parents[3]
        self.dataset_path = self.resolve_dataset_path(dataset_path)
        self.split_output_path = (
            Path(split_output_path)
            if split_output_path is not None
            else project_root / "data" / "telco_v2_preprocessed_splits.npz"
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
        """Clean, engineer, encode, scale and split the Telco data.

        No augmentation is applied: synthetic or image-style transformations are not
        appropriate for these customer-level tabular records.
        """
        self.select_features()
        self.drop_nan()
        self.make_feature_engineering()
        return self.split_dataset()

    def select_features(self):
        """Keep the modelling columns and drop the ``customerID`` identifier.

        The notebook builds its feature matrix from a hard-coded list, so the same
        columns are declared here for the missing-column check.
        """
        missing_columns = [
            column for column in self.raw_columns if column not in self.telco_churn_df.columns
        ]
        if missing_columns:
            raise KeyError(f"Missing columns in the Telco dataset: {missing_columns}")

        self.telco_churn_df = self.telco_churn_df[self.raw_columns].copy()

    def drop_nan(self):
        """Encode the target, coerce TotalCharges, and drop the rows it blanks."""
        self.telco_churn_df["Churn"] = self.telco_churn_df["Churn"].map(
            {"Yes": 1, "No": 0}
        )

        self.telco_churn_df["TotalCharges"] = (
            self.telco_churn_df["TotalCharges"].replace(" ", np.nan).astype(float)
        )
        self.total_charges_missing = int(self.telco_churn_df["TotalCharges"].isna().sum())

        rows_before_cleaning = len(self.telco_churn_df)
        self.telco_churn_df = self.telco_churn_df.dropna().reset_index(drop=True)
        self.rows_dropped = rows_before_cleaning - len(self.telco_churn_df)

    def make_feature_engineering(self):
        """Log-transform the skewed charge columns and recode ``SeniorCitizen``.

        The log is a per-row transform, so applying it before the split leaks
        nothing. The z-scoring that follows is handled in :meth:`encode`, because
        that is where the split boundary is known.
        """
        for column in self.log_transform_features:
            self.telco_churn_df[column] = np.log(self.telco_churn_df[column])

        self.telco_churn_df["SeniorCitizen"] = self.telco_churn_df[
            "SeniorCitizen"
        ].map({1: "Yes", 0: "No"})

    def encode(self, X_train, X_validation, X_test):
        """Z-score the continuous columns and one-hot encode the categoricals.

        Each split is transformed with the same statistics and the same dummy
        columns, so the three matrices always agree on width and order.
        """
        self.fit_scalers(X_train)
        dummy_template = self.one_hot_encode(self.telco_churn_df.drop(columns="Churn"))

        def transform_split(features):
            numeric_features = features[self.continuous_features].astype(float).copy()
            numeric_features.loc[:, self.continuous_features] = (
                numeric_features[self.continuous_features] - self.scaler_means
            ) / self.scaler_stds

            dummy_features = features.drop(
                columns=[col for col in self.excluded_from_dummies if col in features.columns]
            )
            dummies = pd.get_dummies(dummy_features, dtype=float).reindex(
                columns=dummy_template.columns, fill_value=0.0
            )

            return pd.concat([numeric_features, dummies], axis=1)

        encoded_splits = (
            transform_split(X_train),
            transform_split(X_validation),
            transform_split(X_test),
        )

        self.feature_names = encoded_splits[0].columns.tolist()
        return tuple(split.to_numpy(dtype=float) for split in encoded_splits)

    def fit_scalers(self, X_train):
        """Store the mean and standard deviation used to z-score the split.

        When ``fit_scalers_on_train_only`` is False the statistics come from the
        whole cleaned dataset, reproducing the notebook's pre-split standardisation.
        """
        source = X_train[self.continuous_features] if self.fit_scalers_on_train_only else (
            self.telco_churn_df[self.continuous_features]
        )
        self.scaler_means = source.mean()
        self.scaler_stds = source.std()

    def one_hot_encode(self, features):
        """One-hot encode every column with fewer than five unique values.

        Returns the encoded frame with its column names in the notebook's
        ``column_value`` format. Used as the template that keeps every split's
        dummy columns aligned.
        """
        dummy_features = features.drop(
            columns=[col for col in self.excluded_from_dummies if col in features.columns]
        )

        encoded = None
        for column in dummy_features.columns:
            if dummy_features[column].nunique() < self.max_dummy_cardinality:
                dummy_vars = pd.get_dummies(dummy_features[column], dtype=float)
                dummy_vars.columns = [
                    column + "_" + str(value) for value in dummy_vars.columns
                ]
                encoded = (
                    dummy_vars if encoded is None else pd.concat([encoded, dummy_vars], axis=1)
                )

        return encoded

    def split_dataset(self):
        """Create the 65/15/20 train/validation/test split, then encode each split.

        The notebook holds out a flat 20% for testing. The three-way split is kept
        because the notebook that consumes this class tunes its decision threshold
        on validation data. Returns the train and test values for compatibility;
        validation is exposed as ``X_validation`` / ``y_validation`` and all three
        splits are saved to ``data/telco_v2_preprocessed_splits.npz`` by default.
        """
        X = self.telco_churn_df.drop(columns="Churn")
        y = self.telco_churn_df["Churn"].astype(int)

        X_train_pool, X_test_raw, y_train_pool, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        validation_share = self.validation_size / (1 - self.test_size)
        X_train_raw, X_validation_raw, y_train, y_validation = train_test_split(
            X_train_pool,
            y_train_pool,
            test_size=validation_share,
            random_state=self.random_state,
            stratify=y_train_pool,
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