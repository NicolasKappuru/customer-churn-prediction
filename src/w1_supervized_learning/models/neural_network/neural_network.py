# This model is a Neural Network for both Churn datasets (Telco and Telecom).
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.w1_supervized_learning.data_preprocessing.data_preprocessing_telco import (
    PreprocessingTelco
)
from src.w1_supervized_learning.data_preprocessing.data_preprocessing_telecom import (
    PreprocessingTelecom
)

print("Here come the neural network")


def prepare_for_neural_network(X_train, X_test):
    # TensorFlow/other deep models are not needed for these datasets.
    # Telecom already returns a scaled numpy array.
    # Telco returns a DataFrame with category dtype, which the MLP can't
    # consume, so we one-hot encode its categorical columns and scale
    # the numeric ones (scaler fitted only on the training set).

    if not isinstance(X_train, pd.DataFrame):
        return X_train, X_test

    categorical_columns = X_train.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()

    if categorical_columns:
        X_train = pd.get_dummies(
            X_train, columns=categorical_columns, dtype=float
        )
        X_test = pd.get_dummies(
            X_test, columns=categorical_columns, dtype=float
        )
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


class NeuralNetworkModel:

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        # A single hidden layer already works well on tabular data;
        # two layers help capture interactions between the categorical
        # and the usage features.
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, threshold=0.5):

        probabilities = self.model.predict_proba(X_test)[:, 1]

        preds = (probabilities >= threshold).astype(int)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        balanced_acc = balanced_accuracy_score(y_test, preds)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "balanced_accuracy": balanced_acc
        }

    def train_k_fold(
        self,
        X_train,
        y_train,
        threshold=0.5,
        n_splits=5
    ):

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        accuracy_scores = []
        f1_scores = []
        balanced_accuracy_scores = []

        for fold, (train_index, val_index) in enumerate(
            skf.split(X_train, y_train),
            start=1
        ):

            print(f"Fold {fold}:")

            X_fold_train = X_train[train_index]
            X_fold_val = X_train[val_index]

            y_fold_train = y_train.iloc[train_index]
            y_fold_val = y_train.iloc[val_index]

            self.train(
                X_fold_train,
                y_fold_train
            )

            probabilities = self.model.predict_proba(
                X_fold_val
            )[:, 1]

            preds = (
                probabilities >= threshold
            ).astype(int)

            accuracy = accuracy_score(
                y_fold_val,
                preds
            )

            f1 = f1_score(
                y_fold_val,
                preds
            )

            balanced_acc = balanced_accuracy_score(
                y_fold_val,
                preds
            )

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)
            balanced_accuracy_scores.append(balanced_acc)

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")

        print("\nAverage Results:")

        print(
            "Accuracy:",
            sum(accuracy_scores) / len(accuracy_scores)
        )

        print(
            "F1:",
            sum(f1_scores) / len(f1_scores)
        )

        print(
            "Balanced Accuracy:",
            sum(balanced_accuracy_scores)
            / len(balanced_accuracy_scores)
        )


# ============================================================
# TELCO
# ============================================================

telco = PreprocessingTelco()

X_train_telco, X_test_telco, y_train_telco, y_test_telco = (
    telco.preprocess()
)

X_train_telco, X_test_telco = prepare_for_neural_network(
    X_train_telco, X_test_telco
)

print("\n\n================================")
print("NEURAL NETWORK TELCO")
print("================================")

nn_model_telco = NeuralNetworkModel()

nn_model_telco.train(
    X_train_telco,
    y_train_telco
)

results_telco = nn_model_telco.evaluate(
    X_test_telco,
    y_test_telco,
    threshold=0.5
)

print(results_telco)


# ============================================================
# TELECOM
# ============================================================

telecom = PreprocessingTelecom()

X_train_telecom, X_test_telecom, y_train_telecom, y_test_telecom = (
    telecom.preprocess()
)

print("\n\n================================")
print("NEURAL NETWORK TELECOM")
print("================================")

nn_model_telecom = NeuralNetworkModel()

nn_model_telecom.train(
    X_train_telecom,
    y_train_telecom
)

results_telecom = nn_model_telecom.evaluate(
    X_test_telecom,
    y_test_telecom,
    threshold=0.5
)

print(results_telecom)