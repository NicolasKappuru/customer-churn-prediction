"""Constantes que definen el estado y las acciones del MDP KKBox."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionAction:
    name: str
    acceptance_probability: float
    cost: float = 0.0
    acceptance_probability_lapsed: float | None = None  # if set, used when customer is cancelled/lapsed


# Actions defined with name, acceptance probability, cost, and optional lapsed acceptance.
# Each action targets a distinct feature group to prevent the agent from
# learning to spam a single dominant strategy.
ACTION_DEFINITIONS = (
    RetentionAction("Offer subscription renewal", 0.40),              # targets last_is_cancel / last_is_auto_renew
    RetentionAction("Offer annual plan + re-engagement", 0.70, 0.10), # targets last_plan_days + engagement logs
    RetentionAction("Offer 5% discount on renewal", 0.85, 0.15, 0.20), # active: 0.85 | lapsed: 0.20
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

# Action 0 — renewal_offer: restores auto-renew with the same monthly plan
RENEWAL_TRANSACTION_DATE: int = 20170128
RENEWAL_EXPIRE_DATE: int = 20170227      # +30 days from transaction date

# Action 1 — plan_upgrade: annual plan at 20% off list price
UPGRADE_PLAN_DAYS: int = 365
UPGRADE_PLAN_PRICE: float = 1490.0       # list price (~10x monthly)
UPGRADE_DISCOUNT_RATIO: float = 0.20     # 20% off → customer pays 1192
UPGRADE_TRANSACTION_DATE: int = 20170128
UPGRADE_EXPIRE_DATE: int = 20180127      # +365 days from transaction date

# Action 2 — discount_offer: 5% off current monthly plan
DISCOUNT_RATIO: float = 0.05
DISCOUNT_TRANSACTION_DATE: int = 20170128
DISCOUNT_EXPIRE_DATE: int = 20170227     # +30 days from transaction date

# Action 1 — engagement logs: 4 sessions in January 2017 simulating re-engagement
# after accepting the annual plan offer
ENGAGEMENT_LOGS: tuple[dict, ...] = (
    {"date": 20170126, "num_25": 8,  "num_50": 5, "num_75": 3, "num_985": 2, "num_100": 20, "num_unq": 30, "total_secs": 5400.0},
    {"date": 20170128, "num_25": 5,  "num_50": 3, "num_75": 2, "num_985": 1, "num_100": 35, "num_unq": 40, "total_secs": 9000.0},
    {"date": 20170129, "num_25": 10, "num_50": 6, "num_75": 4, "num_985": 2, "num_100": 25, "num_unq": 35, "total_secs": 7200.0},
    {"date": 20170131, "num_25": 3,  "num_50": 2, "num_75": 1, "num_985": 0, "num_100": 15, "num_unq": 20, "total_secs": 3600.0},
)
