"""Defaults for the w2 recurrent models (LSTM, GRU) and Optuna search."""

from pathlib import Path

from ..constants import ARTIFACTS_NAME, EXPORT_NAME, MODEL_CHANNELS

MODEL_ID = "LSTM"
SEED = 110
N_TRIALS = 10   # same budget for every run (manual_c14 used 10)
EPOCHS = 40
PATIENCE = 5
BATCH_SIZE = 512
INFER_BATCH_SIZE = 2048

INPUT_DIM = len(MODEL_CHANNELS)   # the 14 channels of the v6 export
OUTPUT_DIM = 1

HIDDEN_DIMS = (32, 64, 128)
LAYER_DIM_RANGE = (2, 4)
LR_RANGE = (1e-4, 1e-2)
WEIGHT_DECAY_RANGE = (1e-6, 1e-2)
GRAD_CLIP_NORM = 1.0

ARTIFACTS_ROOT = Path("artifacts") / ARTIFACTS_NAME   # run selection in constants.py
FEATURE_DIR = ARTIFACTS_ROOT / "feature_preparation"
EXPORT_DIR = Path(EXPORT_NAME)


def model_dirs(model_name):
    """Artifact folders for one family: ``model_dirs("lstm")`` or ``("gru")``."""
    base = ARTIFACTS_ROOT / model_name.lower()
    return {
        "artifacts": base,
        "optuna": base / "optuna",
        "checkpoints": base / "checkpoints",
        "evaluation": base / "evaluation",
    }


LSTM_DIRS = model_dirs("lstm")
GRU_DIRS = model_dirs("gru")

# Default (LSTM) paths, kept so existing imports keep working.
ARTIFACTS_DIR = LSTM_DIRS["artifacts"]
OPTUNA_DIR = LSTM_DIRS["optuna"]
CHECKPOINT_DIR = LSTM_DIRS["checkpoints"]
EVALUATION_DIR = LSTM_DIRS["evaluation"]
