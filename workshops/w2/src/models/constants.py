"""Fixed values shared by the w2 sequence modeling modules."""

DATASET_VERSION = 5
EXPECTED_CUTOFFS = {
    "train_feb": "2017-01-31",
    "test": "2017-02-28",
}
DEV_COHORT = "train_feb"
TEST_COHORT = "test"

N_DAYS = 90
N_CHANNELS = 20
LISTENING_SLICE = slice(0, 8)
TRANSACTION_SLICE = slice(8, 20)
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
