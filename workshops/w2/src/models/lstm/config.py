"""Defaults for the w2 LSTM model and Optuna search."""

from pathlib import Path

MODEL_ID = "LSTM"
SEED = 110
N_TRIALS = 5
EPOCHS = 40
PATIENCE = 5
BATCH_SIZE = 512
INFER_BATCH_SIZE = 2048

INPUT_DIM = 20
OUTPUT_DIM = 1

HIDDEN_DIMS = (32, 64, 128)
LAYER_DIM_RANGE = (2, 4)
LR_RANGE = (1e-4, 1e-2)
WEIGHT_DECAY_RANGE = (1e-6, 1e-2)
GRAD_CLIP_NORM = 1.0

FEATURE_DIR = Path("artifacts/sherlock_w2/feature_preparation")
EXPORT_DIR = Path("export-sequential-v5")
ARTIFACTS_DIR = Path("artifacts/sherlock_w2/lstm")
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
