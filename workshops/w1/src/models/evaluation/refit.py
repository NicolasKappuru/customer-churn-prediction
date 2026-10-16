"""Refit the selected model families on February, then score March."""

from pathlib import Path

from ..baselines.logistic_regression import fit_logistic
from ..baselines.random_forest import fit_random_forest
from ..baselines.xgboost_model import refit_xgboost
from ..constants import MODEL_IDS


def fit_development_models(
    *,
    train_df,
    features,
    numeric,
    categorical,
    dev_rows,
    split_data,
    registry,
    tree_frame,
    rf_data,
    seed,
    train_verbose,
    xgb_device,
    xgb_balanced_weight,
):
    """Fit each family once on all development rows using frozen parameters."""
    y_dev = split_data.y_all[dev_rows]
    records = registry.records

    m1 = fit_logistic(
        records["M1"]["best_C"],
        train_df.iloc[dev_rows][features],
        y_dev,
        numeric=numeric,
        categorical=categorical,
        seed=seed,
        verbose=train_verbose,
    )

    m2 = fit_random_forest(
        records["M2"]["best_params"],
        rf_data.frame(dev_rows, "development"),
        y_dev,
        seed=seed,
    )

    m3 = refit_xgboost(
        records["M3"]["best_params"],
        tree_frame(dev_rows, "development"),
        y_dev,
        n_estimators=records["M3"]["best_iteration"],
        device=xgb_device,
        seed=seed,
        balanced=xgb_balanced_weight,
    )

    models = {"M1": m1, "M2": m2, "M3": m3}
    for model_id, model in models.items():
        records[model_id]["dev_estimator"] = model
    return models


def score_holdout(
    *,
    models,
    train_df,
    features,
    split_data,
    registry,
    tree_frame,
    rf_data,
    selection_manifest,
):
    """Score March once after the tune selection manifest has been saved."""
    if not Path(selection_manifest).is_file():
        raise FileNotFoundError(selection_manifest)

    rows = split_data.rows("holdout")
    y_hold = split_data.y("holdout")
    tree_holdout = tree_frame(rows, "development")
    holdout_proba = {
        "M1": models["M1"].predict_proba(
            train_df.iloc[rows][features]
        )[:, 1],
        "M2": models["M2"].predict_proba(
            rf_data.frame(rows, "development")
        )[:, 1],
        "M3": models["M3"].predict_proba(tree_holdout)[:, 1],
    }
    for model_id in MODEL_IDS:
        registry.set_proba(model_id, "holdout", holdout_proba[model_id])
    print("holdout probabilities ready for all tabular models")
    return y_hold, holdout_proba
