"""Customer data model — holds the 59 preprocessed features + churn prob.

The 59 features are split across the three display tabs exactly as they appear
in the XGBoost feature set used by the supervised model.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ── feature groupings (order = display order) ────────────────────────────────

MEMBER_FEATURES: list[str] = [
    "city", "registered_via", "bd", "bd_missing",
    "tenure_days", "registration_year", "registration_month", "has_member_record",
]

TRANSACTION_FEATURES: list[str] = [
    "is_cancel_change_flag", "is_auto_renew_change_flag", "transaction_count",
    "average_amount_paid", "average_amount_charged", "average_plan",
    "n_payment_methods", "n_plans", "n_list_prices", "n_amounts_paid",
    "duration", "plan_coverage_ratio", "sum_plan_days", "sum_list_price",
    "sum_amount_paid", "more_than_30", "autorenew_not_cancel", "not_autorenew_cancel",
    "max_plan_days", "last_payment_method_id", "last_plan_days",
    "last_is_auto_renew", "last_is_cancel", "last_discount",
    "days_to_expire", "expire_dayofweek", "days_since_last_transaction",
    "difference", "has_transactions",
]

LOG_FEATURES: list[str] = [
    "sum_last_month_unq", "sum_last_month_total_secs", "mean_last_month_total_secs",
    "last_login", "last_sum_total_songs", "last_ratio_100", "last_ratio_25",
    "sum_last_second_month_total_secs", "last_second_login",
    "last_second_sum_total_songs", "last_second_ratio_100", "last_second_ratio_25",
    "last_third_sum_total_songs", "last_third_ratio_100", "last_third_ratio_25",
    "login_difference", "trend_songs", "trend_secs", "trend_ratio100", "trend_ratio25",
    "days_since_last_log", "has_listening_logs",
]

ALL_FEATURES: list[str] = MEMBER_FEATURES + TRANSACTION_FEATURES + LOG_FEATURES

# ── display colours ───────────────────────────────────────────────────────────
COL_DEFAULT = (208, 208, 208)

ACTION_COLORS = [
    (192, 160, 255),   # 0 — purple     (auto-renewal reminder)
    (100, 200, 255),   # 1 — light blue (loyalty renewal)
    (255, 220,  80),   # 2 — yellow     (content re-engagement)
    (144, 238, 144),   # 3 — light green (downgrade to 99 NTD)
    (200, 200, 215),   # 4 — slate grey  (do nothing)
]


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt(name: str, val: float | int | None) -> str:
    if val is None:
        return "—"
    # Boolean-like flags
    if name.endswith(("_flag", "_record", "_logs", "has_transactions",
                       "more_than_30", "bd_missing")):
        return "Yes" if val else "No"
    # Ratio / small float
    if name.startswith(("last_ratio", "trend_ratio", "plan_coverage")):
        return f"{float(val):.3f}"
    # Large sums
    if name in ("sum_last_month_total_secs", "sum_last_second_month_total_secs",
                "mean_last_month_total_secs", "trend_secs"):
        return f"{float(val):,.0f}s"
    if "price" in name or "amount" in name or "paid" in name or "charged" in name:
        return f"${float(val):,.1f}"
    # Integer-like
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


# ── model ─────────────────────────────────────────────────────────────────────

@dataclass
class CustomerDataModel:
    features: dict[str, float | int]    # the 59 preprocessed features
    churn_probability: float = 0.0
    reward: float = 0.0
    action_selected: int | None = None
    action_accepted: bool | None = None
    _changed: set = field(default_factory=set)

    # ── tab data accessors ────────────────────────────────────────────────────

    def member_rows(self) -> list[tuple[str, str]]:
        return [(k, _fmt(k, self.features.get(k))) for k in MEMBER_FEATURES]

    def transaction_rows(self) -> list[tuple[str, str]]:
        return [(k, _fmt(k, self.features.get(k))) for k in TRANSACTION_FEATURES]

    def log_rows(self) -> list[tuple[str, str]]:
        return [(k, _fmt(k, self.features.get(k))) for k in LOG_FEATURES]

    # ── change tracking ───────────────────────────────────────────────────────

    def apply_action_result(
        self,
        new_features: dict[str, float | int],
        new_churn: float,
        reward: float,
        action: int,
        accepted: bool,
    ) -> None:
        self._changed        = {k for k in new_features if new_features.get(k) != self.features.get(k)}
        self.features        = new_features
        self.churn_probability = new_churn
        self.reward          = reward
        self.action_selected = action
        self.action_accepted = accepted

    def field_color(self, key: str) -> tuple:
        if key in self._changed and self.action_selected is not None:
            return ACTION_COLORS[self.action_selected]
        return COL_DEFAULT

    def reset(self, features: dict, churn: float) -> None:
        self.features          = features
        self.churn_probability = churn
        self.reward            = 0.0
        self.action_selected   = None
        self.action_accepted   = None
        self._changed          = set()
