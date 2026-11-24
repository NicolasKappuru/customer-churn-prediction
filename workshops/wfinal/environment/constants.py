"""Constantes que definen el estado y las acciones del MDP KKBox."""

import datetime
from dataclasses import dataclass

@dataclass(frozen=True)
class RetentionAction:
    name: str
    acceptance_probability: float                        # base / fallback
    acceptance_probability_high: float | None = None    # conditional high (see ActionsManager)
    cost: float = 0.0


# 5 log-based retention strategies + Do nothing.
# All actions add user_log entries — no transactions added (transaction features
# increase churn probability for the manual segment in the XGB model).
ACTION_DEFINITIONS = (
    RetentionAction("Send content re-engagement",   0.05, 0.50, 0.05),  # 0
    RetentionAction("Reactivate dormant listener",  0.20, 0.55, 0.04),  # 1
    RetentionAction("Trend recovery push",          0.15, 0.78, 0.06),  # 2
    RetentionAction("Completion booster",           0.20, 0.60, 0.05),  # 3
    RetentionAction("Listening intensifier",        0.30, 0.72, 0.04),  # 4
    RetentionAction("Do nothing",                   1.00, None, 0.00),  # 5
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

# Observation space: 48 preprocessed XGBoost features + 1 churn probability
N_MODEL_FEATURES: int = 48
N_OBSERVATION: int = N_MODEL_FEATURES + 1

# ── Action 0 — Send content re-engagement ─────────────────────────────────────
# Adds 2–8 scaled log entries spread across January 2017.
# Valid when customer has logs within the last 90 days (still reachable).
# Favorable when January activity is already declining vs December.
REENGAGEMENT_WINDOW_DAYS: int = 90
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

# ── Action 1 — Reactivate dormant listener ────────────────────────────────────
# Adds 2–3 scaled log entries in mid-January for customers inactive >7 days.
# Favorable when customer has long tenure (loyal user who lapsed).
DORMANT_INACTIVE_DAYS: int = 7
DORMANT_LOG_POOL: tuple[dict, ...] = (
    {"date": 20170104, "num_25": 3, "num_50": 2, "num_75": 1, "num_985": 1, "num_100": 8,  "num_unq": 12, "total_secs": 2400.0},
    {"date": 20170110, "num_25": 4, "num_50": 2, "num_75": 1, "num_985": 1, "num_100": 9,  "num_unq": 13, "total_secs": 2600.0},
    {"date": 20170117, "num_25": 3, "num_50": 2, "num_75": 1, "num_985": 0, "num_100": 7,  "num_unq": 11, "total_secs": 2200.0},
    {"date": 20170122, "num_25": 4, "num_50": 2, "num_75": 2, "num_985": 1, "num_100": 10, "num_unq": 14, "total_secs": 2800.0},
)
DORMANT_TENURE_DAYS: int = 365   # favorable if tenure >= this

# ── Action 2 — Trend recovery push ────────────────────────────────────────────
# Adds 3–5 January logs scaled to 1.5× December daily average to reverse decline.
# Valid when January activity is already below December (declining trend).
# Favorable when customer had good December engagement (3+ logs = meaningful base).
TREND_RECOVERY_LOG_POOL: tuple[dict, ...] = (
    {"date": 20170106, "num_25": 5, "num_50": 4, "num_75": 3, "num_985": 2, "num_100": 14, "num_unq": 20, "total_secs": 3800.0},
    {"date": 20170111, "num_25": 6, "num_50": 4, "num_75": 3, "num_985": 2, "num_100": 16, "num_unq": 22, "total_secs": 4200.0},
    {"date": 20170116, "num_25": 5, "num_50": 3, "num_75": 3, "num_985": 2, "num_100": 13, "num_unq": 19, "total_secs": 3600.0},
    {"date": 20170121, "num_25": 6, "num_50": 4, "num_75": 3, "num_985": 2, "num_100": 15, "num_unq": 21, "total_secs": 4000.0},
    {"date": 20170126, "num_25": 5, "num_50": 4, "num_75": 2, "num_985": 2, "num_100": 14, "num_unq": 20, "total_secs": 3800.0},
)
TREND_RECOVERY_BOOST: float = 1.5   # scale logs to 150% of December daily average
TREND_RECOVERY_MIN_DEC_LOGS: int = 3

# ── Action 3 — Completion booster ─────────────────────────────────────────────
# Adds 2–4 high-completion (num_100 heavy) log entries for active January customers
# with low completion ratios. Targets last_ratio_100 and trend_ratio100.
# Valid when customer has ≥1 January login and completion ratio < 50%.
# Favorable when customer has ≥2 January logins (already re-engaged).
COMPLETION_LOG_POOL: tuple[dict, ...] = (
    {"date": 20170107, "num_25": 1, "num_50": 1, "num_75": 1, "num_985": 2, "num_100": 16, "num_unq": 16, "total_secs": 3800.0},
    {"date": 20170114, "num_25": 1, "num_50": 1, "num_75": 1, "num_985": 1, "num_100": 14, "num_unq": 14, "total_secs": 3400.0},
    {"date": 20170120, "num_25": 1, "num_50": 1, "num_75": 1, "num_985": 2, "num_100": 15, "num_unq": 15, "total_secs": 3600.0},
    {"date": 20170128, "num_25": 1, "num_50": 1, "num_75": 1, "num_985": 1, "num_100": 13, "num_unq": 13, "total_secs": 3200.0},
)
LOW_COMPLETION_RATIO: float = 0.50   # valid when ratio_100 < this
COMPLETION_MIN_LOGINS: int = 2       # favorable when January logins >= this

# ── Action 4 — Listening intensifier ──────────────────────────────────────────
# Adds 1–3 high-duration (high total_secs) log entries for active January customers
# with short average sessions. Targets sum_last_month_total_secs.
# Valid when customer has ≥1 January login and mean session < 2000 secs.
# Favorable when customer has ≥2 January logins.
INTENSIFIER_LOG_POOL: tuple[dict, ...] = (
    {"date": 20170108, "num_25": 2, "num_50": 2, "num_75": 3, "num_985": 3, "num_100": 10, "num_unq": 15, "total_secs": 7200.0},
    {"date": 20170113, "num_25": 2, "num_50": 2, "num_75": 3, "num_985": 2, "num_100": 9,  "num_unq": 14, "total_secs": 6800.0},
    {"date": 20170124, "num_25": 2, "num_50": 2, "num_75": 2, "num_985": 3, "num_100": 10, "num_unq": 15, "total_secs": 7000.0},
)
LOW_INTENSITY_SECS: float = 2000.0   # valid when mean total_secs/login < this
INTENSIFIER_MIN_LOGINS: int = 2      # favorable when January logins >= this

# ── Misc ───────────────────────────────────────────────────────────────────────
MAX_STEPS_PER_EPISODE: int = 1
DO_NOTHING_ACTION: int = 5

# ── Cohort / cutoff configuration ─────────────────────────────────────────────
CUTOFF_DATE: datetime.date = datetime.date(2017, 1, 31)
CUTOFF_INT: int = 20170131
CUTOFF_MONTH: str = "201701"
PREV_MONTH: str = "201612"
LOG_WINDOW_START: int = 20161101
LOG_WINDOW_END: int = 20170131

NOISE_STD: float = 0.10
CUSTOMER_GAIN: tuple[float, ...] = (0.3, 0.4, 0.4, 0.3, 0.3)   # actions 0..4
CUSTOMER_LOSS: tuple[float, ...] = (0.1, 0.2, 0.2, 0.1, 0.1)
REJECT_PENALTY: float = 0.10
REJECT_BONUS: float  = 0.01
CUSTOMER_N_ACTIONS: int = 2
CUSTOMER_OBS_SIZE: int = 71
