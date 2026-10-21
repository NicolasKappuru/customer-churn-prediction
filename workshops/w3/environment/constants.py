"""Constantes que definen el estado y las acciones del MDP KKBox."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionAction:
    name: str
    acceptance_probability: float
    cost: float = 0.0


# Actions defined with name, acceptance probability and cost.
# Each action targets a distinct feature group to prevent the agent from
# learning to spam a single dominant strategy.
ACTION_DEFINITIONS = (
    RetentionAction("Offer subscription renewal", 0.90),       # targets last_is_cancel / last_is_auto_renew
    RetentionAction("Offer annual plan upgrade", 0.90),        # targets last_plan_days / days_to_expire
    RetentionAction("Offer 50% discount on renewal", 0.90, 0.10),  # targets average_amount_paid / last_discount
    RetentionAction("Send engagement campaign", 0.90),         # targets days_since_last_log / trend_songs
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

# Action 1 — plan_upgrade: commits customer to an annual plan
UPGRADE_PLAN_DAYS: int = 365
UPGRADE_PLAN_PRICE: float = 1490.0       # ~10x the default monthly price
UPGRADE_TRANSACTION_DATE: int = 20170128
UPGRADE_EXPIRE_DATE: int = 20180127      # +365 days from transaction date

# Action 2 — discount_offer: 50% off on the current plan
DISCOUNT_RATIO: float = 0.50
DISCOUNT_TRANSACTION_DATE: int = 20170128
DISCOUNT_EXPIRE_DATE: int = 20170227     # +30 days from transaction date

# Action 3 — engagement_boost: simulated high-activity listening session
# Date is within January 2017 (the last eligible month for trend_songs)
ENGAGEMENT_LOG_DATE: int = 20170129
ENGAGEMENT_NUM_25: int = 5
ENGAGEMENT_NUM_50: int = 3
ENGAGEMENT_NUM_75: int = 2
ENGAGEMENT_NUM_985: int = 1
ENGAGEMENT_NUM_100: int = 35
ENGAGEMENT_NUM_UNQ: int = 40
ENGAGEMENT_TOTAL_SECS: float = 9000.0

# Minimum days of inactivity required for engagement_boost to be valid
ENGAGEMENT_INACTIVITY_THRESHOLD: int = 7
