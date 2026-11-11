"""M1 logistic regression baseline."""

import time
from functools import partial

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..constants import EPS, LOGISTIC_MODEL_ID, SEED, TRAIN_VERBOSE
from ..tunning.tunning import run_study
from .common import BaselineData, register_baseline, tune_log_loss


# Constants
LOGISTIC_INITIAL_C = 1.0
LOGISTIC_MAX_ITER = 200
LOGISTIC_SOLVER = "newton-cholesky"


# Data preparation
def prepare_logistic_data(cohort_df, features, split_data):
    """Select raw model columns and labels for each split."""
    return BaselineData(
        fit_frame=cohort_df.iloc[split_data.rows("fit")][features],
        fit_labels=split_data.y("fit"),
        tune_frame=cohort_df.iloc[split_data.rows("tune")][features],
        tune_labels=split_data.y("tune"),
    )


def make_logistic_preprocessor(numeric, categorical):
    """Impute and scale numerics; learn one-hot levels during fitting."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ]
    )


# Model
def make_logistic_model(c, numeric, categorical, seed=SEED, verbose=TRAIN_VERBOSE):
    """Build the preprocessing and classifier pipeline."""
    return Pipeline(
        [
            ("pre", make_logistic_preprocessor(numeric, categorical)),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    solver=LOGISTIC_SOLVER,
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=seed,
                    verbose=verbose,
                ),
            ),
        ]
    )


def fit_logistic(
    c, frame, labels, *, numeric, categorical, seed=SEED, verbose=TRAIN_VERBOSE
):
    """Fit the pipeline on the supplied frame and labels."""
    model = make_logistic_model(c, numeric, categorical, seed, verbose)
    return model.fit(frame, labels)


# Search space and objective
def suggest_logistic_params(trial):
    """Sample the regularization strength."""
    return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}


def logistic_objective(trial, *, data, numeric, categorical, seed, verbose, eps):
    """Fit one candidate and return its tune log-loss."""
    params = suggest_logistic_params(trial)
    model = fit_logistic(
        params["C"],
        data.fit_frame,
        data.fit_labels,
        numeric=numeric,
        categorical=categorical,
        seed=seed,
        verbose=verbose,
    )
    return tune_log_loss(model, data, eps)


# Registration
def register_logistic(registry, model, data, study, counts, seconds):
    """Store M1 predictions and regularization metadata."""
    return register_baseline(
        registry,
        LOGISTIC_MODEL_ID,
        model,
        data,
        counts,
        seconds,
        family="logistic",
        best_params=dict(study.best_params),
        best_C=study.best_params["C"],
    )


# Run
def run_logistic(
    *,
    cohort_df,
    features,
    numeric,
    categorical,
    split_data,
    registry,
    study,
    total_trials,
    out_dir,
    seed=SEED,
    verbose=TRAIN_VERBOSE,
    eps=EPS,
):
    """Coordinate M1 preparation, tuning, final fit, and registration."""
    data = prepare_logistic_data(cohort_df, features, split_data)
    objective = partial(
        logistic_objective,
        data=data,
        numeric=numeric,
        categorical=categorical,
        seed=seed,
        verbose=verbose,
        eps=eps,
    )

    start = time.perf_counter()
    study, counts = run_study(
        LOGISTIC_MODEL_ID,
        objective,
        study,
        total_trials,
        out_dir,
        enqueue=[{"C": LOGISTIC_INITIAL_C}],
    )
    model = fit_logistic(
        study.best_params["C"],
        data.fit_frame,
        data.fit_labels,
        numeric=numeric,
        categorical=categorical,
        seed=seed,
        verbose=verbose,
    )
    register_logistic(
        registry, model, data, study, counts, time.perf_counter() - start
    )
    print("M1 best C:", study.best_params["C"])
    return model
