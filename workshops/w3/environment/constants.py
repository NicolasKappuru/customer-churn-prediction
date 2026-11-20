"""Constantes que definen el estado y las acciones del MDP KKBox."""

import datetime
from dataclasses import dataclass

@dataclass(frozen=True)
class RetentionAction:
    name: str
    acceptance_probability: float                        # base / fallback
    acceptance_probability_high: float | None = None    # conditional high (see ActionsManager)
    cost: float = 0.0


ACTION_DEFINITIONS = (
    RetentionAction("Reactivate dormant listener",    0.20, 0.70, 0.04),
    RetentionAction("Send content re-engagement",     0.10, 0.70, 0.05),
    RetentionAction("Manual renewal nudge",            0.30, 0.70, 0.07),
    RetentionAction("Do nothing",                     1.00, None, 0.00),
)

# Raw customer data structure keys
MEMBER_FIELDS: tuple[str, ...] = (
    "city",
    "registered_via",
    "bd",
    "registration_init_time",
)

TRANSACTION_FIELDS: tuple[str, ...] = (
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "actual_amount_paid",
    "is_auto_renew",
    "is_cancel",
    "transaction_date",
    "membership_expire_date",
)

LOG_FIELDS: tuple[str, ...] = (
    "date",
    "num_25",
    "num_50",
    "num_75",
    "num_985",
    "num_100",
    "num_unq",
    "total_secs",
)

# Observation space: 59 preprocessed XGBoost features + 1 churn probability
N_MODEL_FEATURES: int = 59
N_OBSERVATION: int = N_MODEL_FEATURES + 1

# Default transaction values for customers with no prior history
DEFAULT_PAYMENT_METHOD_ID: int = 38
DEFAULT_PLAN_DAYS: int = 30
DEFAULT_PLAN_PRICE: float = 149.0

# Action 2 — content re-engagement: adds 2-8 scaled log entries.
# Valid when Jan activity < Dec activity; logs scaled to customer's Dec daily average.
REENGAGEMENT_DISCOUNT_NTD: float = 50.0
REENGAGEMENT_TRANSACTION_DATE: int = 20170131

# Pool of 8 log entries spread across January 2017.
REENGAGEMENT_LOG_POOL: tuple[dict, ...] = (
    {"date": 20170102, "num_25": 3, "num_50": 2, "num_75": 1, "num_985": 1, "num_100": 8,  "num_unq": 12, "total_secs": 2400.0},
    {"date": 20170105, "num_25": 4, "num_50": 2, "num_75": 2, "num_985": 1, "num_100": 10, "num_unq": 15, "total_secs": 2800.0},
    {"date": 20170109, "num_25": 5, "num_50": 3, "num_75": 1, "num_985": 0, "num_100": 12, "num_unq": 18, "total_secs": 3200.0},
    {"date": 20170112, "num_25": 2, "num_50": 1, "num_75": 1, "num_985": 0, "num_100": 6,  "num_unq": 8,  "total_secs": 1800.0},
    {"date": 20170115, "num_25": 6, "num_50": 3, "num_75": 2, "num_985": 1, "num_100": 14, "num_unq": 20, "total_secs": 3600.0},
    {"date": 20170119, "num_25": 3, "num_50": 2, "num_75": 1, "num_985": 0, "num_100": 9,  "num_unq": 13, "total_secs": 2200.0},
    {"date": 20170123, "num_25": 5, "num_50": 3, "num_75": 2, "num_985": 1, "num_100": 11, "num_unq": 16, "total_secs": 3000.0},
    {"date": 20170127, "num_25": 4, "num_50": 2, "num_75": 1, "num_985": 0, "num_100": 8,  "num_unq": 11, "total_secs": 2600.0},
)

# Action 3 — renewal nudge: transaction date for the simulated renewal
RENEWAL_TRANSACTION_DATE: int = 20170131

# Action 3 — do nothing: no state change, no cost, always valid


# ── Cohort / cutoff configuration ─────────────────────────────────────────────

CUTOFF_DATE: datetime.date = datetime.date(2017, 1, 31)
CUTOFF_INT: int = 20170131
CUTOFF_MONTH: str = "201701"
PREV_MONTH: str = "201612"

LOG_WINDOW_START: int = 20161101
LOG_WINDOW_END: int = 20170131
