"""Persist the February models used in the final March evaluation."""

import json
from pathlib import Path

import joblib


def save_models(models, out_dir):
    """Save fitted estimators; M1 includes its preprocessing pipeline."""
    out_dir = Path(out_dir)
    model_dir = out_dir / "models" / "development"
    model_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for model_id, model in models.items():
        path = model_dir / f"{model_id}.joblib"
        joblib.dump(model, path, compress=3)
        files[model_id] = path.relative_to(out_dir).as_posix()
    return files


def save_preprocessing(
    features, categorical, rf_vocabulary, xgb_vocabulary, source_params, out_dir
):
    """Save feature order and the development vocabularies needed by M2/M3."""
    out_dir = Path(out_dir)
    path = out_dir / "preprocessing" / "development.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "features": list(features),
        "categorical": list(categorical),
        "random_forest_vocabulary": rf_vocabulary,
        "xgboost_vocabulary": xgb_vocabulary,
        "source_preprocess_params": source_params,
    }
    path.write_text(json.dumps(state, indent=2))
    return path.relative_to(out_dir).as_posix()


def save_run_metadata(
    *,
    config,
    env,
    contract,
    selection,
    thresholds,
    model_files,
    preprocessing_file,
    out_dir,
):
    """Describe the saved models, exact thresholds, and evaluation protocol."""
    path = Path(out_dir) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config,
        "env": env,
        "contract": contract,
        "selection": selection,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "trained_on": "February 2017",
        "evaluated_on": "March 2017 (test)",
        "models": model_files,
        "preprocessing": preprocessing_file,
    }
    path.write_text(json.dumps(metadata, indent=2))
    return path
