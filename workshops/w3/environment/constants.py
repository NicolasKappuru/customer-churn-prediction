"""Constantes que definen el estado y las acciones del MDP KKBox."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionAction:
    name: str
    acceptance_probability: float                        # base / fallback
    acceptance_probability_high: float | None = None    # conditional high (see ActionsManager)
    cost: float = 0.0


ACTION_DEFINITIONS = (
    RetentionAction("Send auto-renewal reminder",           0.10, 0.60, 0.01),
    RetentionAction("Offer loyalty renewal with discount",  0.15, 0.90, 0.15),
    RetentionAction("Send content re-engagement",           0.10, 0.80, 0.05),
    RetentionAction("Offer downgrade to 99 NTD plan",       0.15, 0.90, 0.20),
    RetentionAction("Do nothing",                           1.00, None, 0.00),
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

# Action 1 — loyalty renewal: 30-day plan at 149 NTD with 20 NTD fixed discount + auto-renew
# Valid for tenure_days > 365 with auto-renew on; last_discount=20 stays in the low-churn zone
FIDELITY_PLAN_DAYS: int = 30
FIDELITY_PLAN_PRICE: float = 149.0
FIDELITY_DISCOUNT_NTD: float = 20.0         # fixed NTD discount → customer pays 129
FIDELITY_TRANSACTION_DATE: int = 20170131   # cutoff date

# Action 2 — content re-engagement: adds 2–8 scaled log entries + transaction with 50 NTD discount.
# Valid when Jan activity < Dec activity; logs scaled to customer's Dec daily average.
REENGAGEMENT_DISCOUNT_NTD: float = 50.0
REENGAGEMENT_TRANSACTION_DATE: int = 20170131   # cutoff date

# Pool of 8 log entries spread across January 2017.
# ActionsManager randomly selects 2–8 of these to simulate re-engagement activity.
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

# Action 3 — downgrade: switch to 99 NTD plan (no discount, last_discount=0)
# Valid when current plan_list_price > 99; targets customers near expiry considering cancellation
DOWNGRADE_PLAN_PRICE: float = 99.0
DOWNGRADE_PLAN_DAYS: int = 30
DOWNGRADE_TRANSACTION_DATE: int = 20170131   # cutoff date

# Action 4 — do nothing: no state change, no cost, always valid
# acceptance and cost are unused — apply_action short-circuits for this action
