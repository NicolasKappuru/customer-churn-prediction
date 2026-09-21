from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)
from xgboost import XGBClassifier

from w1_supervized_learning.data_preprocessing.data_preprocessing_telco import (
    PreprocessingTelco
)
from w1_supervized_learning.data_preprocessing.data_preprocessing_telecom import (
    PreprocessingTelecom
)


print("Here come xgboost")


class XGBoostModel:

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, params):
        self.model = XGBClassifier(**params)
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
        params,
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

            X_fold_train = X_train.iloc[train_index]
            X_fold_val = X_train.iloc[val_index]

            y_fold_train = y_train.iloc[train_index]
            y_fold_val = y_train.iloc[val_index]

            self.train(
                X_fold_train,
                y_fold_train,
                params
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
# HYPERPARAMETER SEARCH
# ============================================================

def optimize_xgboost(X_train, y_train):

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    param_grid = {

        "n_estimators": [
            100,
            200,
            300,
            400,
            500
        ],

        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1
        ],

        "max_depth": [
            3,
            4,
            5,
            6,
            7
        ],

        "min_child_weight": [
            1,
            3,
            5,
            7
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "gamma": [
            0,
            0.1,
            0.3,
            0.5
        ],

        "reg_alpha": [
            0,
            0.01,
            0.1,
            0.5
        ],

        "reg_lambda": [
            1,
            2,
            5,
            10
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring="balanced_accuracy",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("\n==============================")
    print("BEST PARAMETERS")
    print("==============================")

    print(search.best_params_)

    print("\nBest Balanced Accuracy:")
    print(search.best_score_)

    return search.best_params_


# ============================================================
# TELCO
# ============================================================

telco = PreprocessingTelco()

X_train_telco, X_test_telco, y_train_telco, y_test_telco = (
    telco.preprocess()
)

print("\n\n================================")
print("OPTIMIZING TELCO")
print("================================")

best_params_telco = optimize_xgboost(
    X_train_telco,
    y_train_telco
)

print("\nBest parameters Telco:")
print(best_params_telco)


# ============================================================
# TELECOM
# ============================================================

telecom = PreprocessingTelecom()

X_train_telecom, X_test_telecom, y_train_telecom, y_test_telecom = (
    telecom.preprocess()
)

print("\n\n================================")
print("OPTIMIZING TELECOM")
print("================================")

best_params_telecom = optimize_xgboost(
    X_train_telecom,
    y_train_telecom
)

print("\nBest parameters Telecom:")
print(best_params_telecom)


# ============================================================
# FINAL MODELS
# ============================================================

print("\n\n================================")
print("FINAL TELCO MODEL")
print("================================")

xgb_model_telco = XGBoostModel()

xgb_model_telco.train(
    X_train_telco,
    y_train_telco,
    best_params_telco
)

results_telco = xgb_model_telco.evaluate(
    X_test_telco,
    y_test_telco,
    threshold=0.5
)

print(results_telco)


print("\n\n================================")
print("FINAL TELECOM MODEL")
print("================================")

xgb_model_telecom = XGBoostModel()

xgb_model_telecom.train(
    X_train_telecom,
    y_train_telecom,
    best_params_telecom
)

results_telecom = xgb_model_telecom.evaluate(
    X_test_telecom,
    y_test_telecom,
    threshold=0.5
)

print(results_telecom)