"""Defaults for the w2 trajectory models (LSTM, GRU).

The 90-day window is split in two: the model reads days ``0..82`` and predicts
the 11 event targets of days ``83..89``. The 7 state channels are not
predicted (option B); this package does not simulate them, it only evaluates
the predicted events.

The export, the split/scalers and the artifact folder follow the same run
selection as the classifiers (``src/models/constants.py``: ``SEGMENT`` and
``ALIGNMENT``, overridable with ``W2_SEGMENT`` / ``W2_ALIGNMENT`` /
``W2_CYCLE_K``). The classifier's feature preparation of that run must exist.
"""

from pathlib import Path

from ..constants import ARTIFACTS_NAME, EXPORT_NAME, MODEL_CHANNELS

# Same values as ``recurrent_models/config.py``. Copied, not imported: importing
# that package runs its ``__init__``, which loads the classifier data module.
SEED = 110
BATCH_SIZE = 512
INFER_BATCH_SIZE = 2048
GRAD_CLIP_NORM = 1.0
LR_RANGE = (1e-4, 1e-2)
WEIGHT_DECAY_RANGE = (1e-6, 1e-2)
EXPORT_DIR = Path(EXPORT_NAME)
FEATURE_DIR = Path("artifacts") / ARTIFACTS_NAME / "feature_preparation"   # split + scalers of the classifier run

N_DAYS = 90
HORIZON = 7
INPUT_DAYS = N_DAYS - HORIZON  # 83
INPUT_DIM = len(MODEL_CHANNELS)   # the 14 channels of the v6 export

# Channels of the input tensor (same order as the export).
LISTEN_AMOUNT_CHANNELS = (
    "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs",
)
EVENT_AMOUNT_CHANNELS = ("tx_paid_sum", "tx_plan_days_sum")
STATE_CHANNELS = ("tx_state_known", "tx_days_since_last", "tx_days_to_expire")   # not predicted

# Target layout: (batch, HORIZON, N_TARGETS). Positions inside the last axis.
# Every target comes from one of the 14 channels the model also reads
# (MODEL_CHANNELS): the listening events and the payment events. The 3 state
# channels are not predicted; they follow from the payments.
TARGET_NAMES = (
    *LISTEN_AMOUNT_CHANNELS,   # 0..6  scaled amounts, defined on active days
    "active",                  # 7     0/1
    "pay",                     # 8     0/1  (raw tx_count > 0)
    *EVENT_AMOUNT_CHANNELS,    # 9..10 scaled amounts, defined on payment days
)
N_TARGETS = len(TARGET_NAMES)  # 11
T_LISTEN = slice(0, 7)
T_ACTIVE = 7
T_PAY = 8
T_AMOUNT = slice(9, 11)

# Loss terms. Their weights are fixed before Optuna (see ``loss.calibrate_weights``)
# so that the search cannot win by down-weighting the hard term.
LOSS_TERMS = ("active", "listen", "pay", "amount")
HUBER_DELTA = 1.0

# Optuna. Wider hidden sizes than the classifier: the head now emits 7 x 11 values.
N_TRIALS = 10
EPOCHS = 40
PATIENCE = 5
HIDDEN_DIMS = (64, 128, 256)
LAYER_DIM_RANGE = (2, 3)

ARTIFACTS_ROOT = Path("artifacts") / f"{ARTIFACTS_NAME}_traj"   # sherlock_w2_v6_manual_traj
COMPARISON_DIR = ARTIFACTS_ROOT / "comparison"


def model_dirs(model_name):
    """Artifact folders for one family: ``model_dirs("lstm")`` or ``("gru")``."""
    base = ARTIFACTS_ROOT / model_name.lower()
    return {
        "artifacts": base,
        "optuna": base / "optuna",
        "checkpoints": base / "checkpoints",
        "evaluation": base / "evaluation",
    }
