"""Fixed values shared by the w1 tabular modeling modules."""

DATASET_VERSION = 5
EXPECTED_CUTOFFS = {
    "train_feb": "2017-01-31",
    "train": "2017-02-28",
    "test": "2017-03-31",
}
DEV_COHORT = "train_feb"
HOLD_COHORT = "train"

SEED = 110
EPS = 1e-7
MODEL_IDS = ("M1", "M2", "M3")
LOGISTIC_MODEL_ID = "M1"
RANDOM_FOREST_MODEL_ID = "M2"
XGBOOST_MODEL_ID = "M3"
CLASS_LABELS = ("Permanece (0)", "Churn (1)")
SPLIT_SHARES = {"fit": 0.75, "stop": 0.125, "tune": 0.125}
SPLIT_HASH_SALT = "salt110:"
SPLIT_PROTOCOL = "cross_month_feb_to_mar"

OPTUNA_MAX_TRIALS = 50
TRIALS_FULL = {"M1": 3, "M2": 3, "M3": 8}
TRIALS_SMOKE = {model_id: 2 for model_id in MODEL_IDS}
# RF v2 trains the recorded parameters exactly; v1 silently clamped some trials.
STUDY_TAGS = {"M1": "newton_cholesky_v1", "M2": "rf_bounded_v2"}

TRAIN_VERBOSE = 1
SMOKE_USERS = 10
SMOKE_TEST_ROWS = 10

