"""Telco autoencoder — project-wide constants and paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
W4_TELCO_ROOT   = PROJECT_ROOT / "src" / "w4_autoencoders"
DATA_DIR        = PROJECT_ROOT / "data"

CHECKPOINTS_DIR = W4_TELCO_ROOT / "checkpoints"
RUNS_DIR        = W4_TELCO_ROOT / "runs"
FIGURES_DIR     = RUNS_DIR / "figures"

SEED = 110

Z_DIMS: tuple[int, ...] = (8, 16, 32)

TELCO_CSV = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Raw feature lists — matches column order used throughout the pipeline
NUMERIC_FEATURES: tuple[str, ...] = (
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
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
)

ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
