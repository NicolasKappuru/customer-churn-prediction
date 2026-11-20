"""Fixed values shared by the w2 sequence modeling modules."""

DATASET_VERSION = 6   # 14 channels, manual segment (src/data_processing/constants.py)
EXPECTED_CUTOFFS = {
    "train_feb": "2017-01-31",
    "test": "2017-02-28",
}
DEV_COHORT = "train_feb"
TEST_COHORT = "test"

N_DAYS = 90
N_CHANNELS = 14
LISTENING_SLICE = slice(0, 8)
TRANSACTION_SLICE = slice(8, 14)
SEED = 110
EPS = 1e-7
CLASS_LABELS = ("Stay (0)", "Churn (1)")

DEV_PARTS = ("fit", "stop", "tune")
SPLIT_SHARES = {"fit": 0.75, "stop": 0.125, "tune": 0.125}
SPLIT_HASH_SALT = "salt110:"
SPLIT_PROTOCOL = "cross_month_feb_to_mar"

# Bounded sample of day-rows used to estimate scaling quantiles.
QUANTILE_SAMPLE = 2_000_000
BLOCK_ROWS = 4096

PHASES = ("fit", "development")

# ---------------------------------------------------------------------------
# Run selection (read once, at import time). Set the environment variables
# BEFORE the first `import src.models...`; in a notebook, restart the kernel
# after changing them.
#
#   W2_ALIGNMENT    calendar | cycle      (default calendar)
#   W2_CYCLE_K      days before expiry    (default 7, only with cycle)
#
# Every run uses the manual segment and the 14 channels of the v6 exports:
#   calendar  -> export-sequential-v6-manual            artifacts/sherlock_w2_v6_manual
#   cycle     -> export-sequential-v6-cycle-k{k}-manual  artifacts/sherlock_w2_v6_manual_cycle_k{k}
# (names must match w2_sequence_preprocessing / src/data_processing/sequences/cycle_export.py)
# ---------------------------------------------------------------------------
import os as _os

SEGMENTS = ("all", "manual")          # used by feature_preparation/segment.py
ALIGNMENTS = ("calendar", "cycle")

SEGMENT = "manual"
ALIGNMENT = _os.environ.get("W2_ALIGNMENT", "calendar")
if ALIGNMENT not in ALIGNMENTS:
    raise ValueError(f"W2_ALIGNMENT={ALIGNMENT!r}; expected one of {ALIGNMENTS}.")
CYCLE_K = int(_os.environ.get("W2_CYCLE_K", "7")) if ALIGNMENT == "cycle" else None

# The 14 channels of the export, in tensor order. The models read all of them.
MODEL_CHANNELS = (
    "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs", "active",
    "tx_count", "tx_paid_sum", "tx_plan_days_sum",
    "tx_state_known", "tx_days_since_last", "tx_days_to_expire",
)
assert len(MODEL_CHANNELS) == N_CHANNELS

CALENDAR_EXPORT_NAME = f"export-sequential-v{DATASET_VERSION}-{SEGMENT}"
CALENDAR_ARTIFACTS_NAME = f"sherlock_w2_v{DATASET_VERSION}_{SEGMENT}"

if ALIGNMENT == "cycle":
    EXPORT_NAME = f"export-sequential-v{DATASET_VERSION}-cycle-k{CYCLE_K}-{SEGMENT}"
    ARTIFACTS_NAME = f"{CALENDAR_ARTIFACTS_NAME}_cycle_k{CYCLE_K}"
else:
    EXPORT_NAME = CALENDAR_EXPORT_NAME
    ARTIFACTS_NAME = CALENDAR_ARTIFACTS_NAME
