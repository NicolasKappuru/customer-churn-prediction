"""Determine and apply valid stochastic customer-retention actions for KKBox."""

from __future__ import annotations

import copy
import datetime
import random
from numbers import Integral

from constants import (
    ACTION_DEFINITIONS,
    DEFAULT_PAYMENT_METHOD_ID,
    DEFAULT_PLAN_DAYS,
    DEFAULT_PLAN_PRICE,
    FIDELITY_PLAN_DAYS,
    FIDELITY_PLAN_PRICE,
    FIDELITY_DISCOUNT_NTD,
    FIDELITY_TRANSACTION_DATE,
    REENGAGEMENT_LOG_POOL,
    REENGAGEMENT_DISCOUNT_NTD,
    REENGAGEMENT_TRANSACTION_DATE,
    DOWNGRADE_PLAN_PRICE,
    DOWNGRADE_PLAN_DAYS,
    DOWNGRADE_TRANSACTION_DATE,
)

_CUTOFF_DATE = datetime.date(2017, 1, 31)


def _date_from_int(d: int) -> datetime.date:
    return datetime.date(d // 10000, (d // 100) % 100, d % 100)


class ActionsManager:
    """Return valid action indices and apply accepted retention offers to a KKBox customer dict."""

    _SONG_KEYS = ("num_25", "num_50", "num_75", "num_985", "num_100")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_favorable(self, action_index: int, customer: dict) -> bool:
        """Return True if the customer meets the high-acceptance condition for this action."""
        if action_index == 0:
            return self._days_since_last_log(customer) <= 30
        if action_index == 1:
            return self._last_month_logins(customer) >= 15
        if action_index == 2:
            return self._days_since_last_log(customer) <= 30
        if action_index == 3:
            return self._tenure_days(customer) < 365
        return False

    def get_valid_actions(self, customer: dict) -> list[int]:
        """Return valid zero-based action indices for the customer's current state."""
        last_tx = self._get_last_transaction(customer)
        valid = []

        # Action 0: reminder — auto-renew off and membership expiring within the next 15 days
        if (last_tx is not None
                and int(last_tx["is_auto_renew"]) == 0
                and 0 <= (_date_from_int(int(last_tx["membership_expire_date"])) - _CUTOFF_DATE).days <= 15):
            valid.append(0)

        # Action 1: loyalty renewal — long-tenure customers with auto-renew active
        if self._tenure_days(customer) > 365 and last_tx is not None and int(last_tx["is_auto_renew"]) == 1:
            valid.append(1)

        # Action 2: reactivation — only when activity is declining
        if self._is_declining_active(customer):
            valid.append(2)

        # Action 3: downgrade — only when current plan is more expensive than 99 NTD
        if last_tx is not None and float(last_tx["plan_list_price"]) > DOWNGRADE_PLAN_PRICE:
            valid.append(3)

        # Action 4: do nothing — always valid
        valid.append(4)

        return valid

    def apply_action(self, customer: dict, action: int) -> tuple[dict, float]:
        """Return the updated customer dict and action cost."""
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer action index.")
        action_index = int(action)
        if action_index < 0 or action_index >= len(ACTION_DEFINITIONS):
            raise ValueError(
                f"action must be between 0 and {len(ACTION_DEFINITIONS) - 1}."
            )
        if action_index not in self.get_valid_actions(customer):
            raise ValueError(
                f"Action {action_index} ({ACTION_DEFINITIONS[action_index].name}) "
                "is not valid for this customer."
            )

        # Do nothing — no state change, no cost, skip stochastic acceptance
        if action_index == 4:
            return customer, 0.0

        action_definition = ACTION_DEFINITIONS[action_index]
        acceptance = self._get_acceptance_probability(action_index, customer)
        if random.random() >= acceptance:
            return customer, 0.0

        updated = copy.deepcopy(customer)
        _dispatch = {
            0: self._apply_reminder_autorenew,
            1: self._apply_fidelity_renewal,
            2: self._apply_content_reengagement,
            3: self._apply_downgrade,
        }
        _dispatch[action_index](updated)
        return updated, action_definition.cost

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _apply_reminder_autorenew(self, customer: dict) -> None:
        """Activate auto-renew on the last transaction — no new transaction added."""
        last_tx = self._get_last_transaction(customer)
        if last_tx is not None:
            last_tx["is_auto_renew"] = 1

    def _apply_fidelity_renewal(self, customer: dict) -> None:
        """Add a 30-day loyalty transaction at 149 NTD with 20 NTD fixed discount + auto-renew."""
        last_tx = self._get_last_transaction(customer)
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": FIDELITY_PLAN_DAYS,
            "plan_list_price":   FIDELITY_PLAN_PRICE,
            "actual_amount_paid": FIDELITY_PLAN_PRICE - FIDELITY_DISCOUNT_NTD,
            "is_auto_renew": 1,
            "is_cancel": 0,
            "transaction_date": FIDELITY_TRANSACTION_DATE,
            "membership_expire_date": self._compute_expire_date(last_tx, FIDELITY_PLAN_DAYS),
        })

    def _apply_content_reengagement(self, customer: dict) -> None:
        """Add 2–8 scaled log entries and a transaction with 50 NTD discount."""
        n = random.randint(2, len(REENGAGEMENT_LOG_POOL))
        selected = random.sample(REENGAGEMENT_LOG_POOL, n)
        customer["user_logs"].extend(self._scale_logs(selected, customer))

        last_tx = self._get_last_transaction(customer)
        price = float(last_tx["plan_list_price"]) if last_tx else DEFAULT_PLAN_PRICE
        plan_days = int(last_tx["payment_plan_days"]) if last_tx else DEFAULT_PLAN_DAYS
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": plan_days,
            "plan_list_price":   price,
            "actual_amount_paid": price - REENGAGEMENT_DISCOUNT_NTD,
            "is_auto_renew": int(last_tx["is_auto_renew"]) if last_tx else 0,
            "is_cancel":     0,
            "transaction_date": REENGAGEMENT_TRANSACTION_DATE,
            "membership_expire_date": self._compute_expire_date(last_tx, plan_days),
        })

    def _apply_downgrade(self, customer: dict) -> None:
        """Add a 30-day transaction at 99 NTD (no discount, last_discount=0)."""
        last_tx = self._get_last_transaction(customer)
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": DOWNGRADE_PLAN_DAYS,
            "plan_list_price":   DOWNGRADE_PLAN_PRICE,
            "actual_amount_paid": DOWNGRADE_PLAN_PRICE,
            "is_auto_renew": int(last_tx["is_auto_renew"]) if last_tx else 0,
            "is_cancel": 0,
            "transaction_date": DOWNGRADE_TRANSACTION_DATE,
            "membership_expire_date": self._compute_expire_date(last_tx, DOWNGRADE_PLAN_DAYS),
        })

    # ------------------------------------------------------------------
    # Acceptance probability
    # ------------------------------------------------------------------

    def _get_acceptance_probability(self, action_index: int, customer: dict) -> float:
        action_def = ACTION_DEFINITIONS[action_index]
        if action_def.acceptance_probability_high is None:
            return action_def.acceptance_probability

        if action_index == 0:
            high = self._days_since_last_log(customer) <= 30
        elif action_index == 1:
            high = self._last_month_logins(customer) >= 15
        elif action_index == 2:
            high = self._days_since_last_log(customer) <= 30
        elif action_index == 3:
            high = self._tenure_days(customer) < 365
        else:
            high = False

        return action_def.acceptance_probability_high if high else action_def.acceptance_probability

    # ------------------------------------------------------------------
    # Condition helpers (computed from raw customer dict, no preprocessor)
    # ------------------------------------------------------------------

    def _days_since_last_log(self, customer: dict) -> int:
        logs = customer.get("user_logs") or []
        if not logs:
            return 9999
        return (_CUTOFF_DATE - _date_from_int(max(int(log["date"]) for log in logs))).days

    def _tenure_days(self, customer: dict) -> int:
        rit = customer.get("registration_init_time")
        return 0 if rit is None else (_CUTOFF_DATE - _date_from_int(int(rit))).days

    def _last_month_logins(self, customer: dict) -> int:
        logs = customer.get("user_logs") or []
        return sum(1 for log in logs if str(int(log["date"]))[:6] == "201701")

    def _is_declining_active(self, customer: dict) -> bool:
        logs = customer.get("user_logs") or []
        if not any(20161101 <= int(log["date"]) <= 20170131 for log in logs):
            return False

        def _songs(prefix: str) -> int:
            return sum(
                int(log.get("num_25", 0)) + int(log.get("num_50", 0)) +
                int(log.get("num_75", 0)) + int(log.get("num_985", 0)) + int(log.get("num_100", 0))
                for log in logs if str(int(log["date"]))[:6] == prefix
            )

        return (_songs("201701") / (_songs("201612") + 1)) < 1.0

    def _avg_difference(self, customer: dict) -> float:
        """Return avg_actual_amount_paid - avg_plan_list_price across all transactions."""
        txns = customer.get("transactions") or []
        if not txns:
            return 0.0
        avg_paid    = sum(float(t["actual_amount_paid"]) for t in txns) / len(txns)
        avg_charged = sum(float(t["plan_list_price"])    for t in txns) / len(txns)
        return avg_paid - avg_charged

    def _in_decision_window(self, customer: dict) -> bool:
        last_tx = self._get_last_transaction(customer)
        if last_tx is None:
            return False
        days = (_date_from_int(int(last_tx["membership_expire_date"])) - _CUTOFF_DATE).days
        return -30 <= days <= 15

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_expire_date(self, last_tx: dict | None, plan_days: int) -> int:
        """Compute expire date as max(last_expire, cutoff) + plan_days."""
        if last_tx is not None:
            last_expire = _date_from_int(int(last_tx["membership_expire_date"]))
            start = max(last_expire, _CUTOFF_DATE)
        else:
            start = _CUTOFF_DATE
        expire = start + datetime.timedelta(days=plan_days)
        return expire.year * 10000 + expire.month * 100 + expire.day

    def _scale_logs(self, entries: list[dict], customer: dict) -> list[dict]:
        """Scale log song counts to match the customer's December 2016 daily average."""
        logs = customer.get("user_logs") or []
        dec_logs = [log for log in logs if str(int(log["date"]))[:6] == "201612"]
        if not dec_logs:
            return list(entries)
        dec_songs = sum(sum(int(log.get(k, 0)) for k in self._SONG_KEYS) for log in dec_logs)
        dec_avg = dec_songs / len(dec_logs)
        scaled = []
        for entry in entries:
            entry_songs = sum(int(entry.get(k, 0)) for k in self._SONG_KEYS)
            if entry_songs == 0:
                scaled.append(dict(entry))
                continue
            factor = dec_avg / entry_songs
            e = dict(entry)
            for k in self._SONG_KEYS:
                e[k] = max(0, round(int(entry.get(k, 0)) * factor))
            e["num_unq"] = max(0, round(int(entry.get("num_unq", 0)) * factor))
            e["total_secs"] = max(0.0, float(entry.get("total_secs", 0.0)) * factor)
            scaled.append(e)
        return scaled

    def _get_last_transaction(self, customer: dict) -> dict | None:
        txns = customer.get("transactions") or []
        if not txns:
            return None
        return max(txns, key=lambda t: int(t["transaction_date"]))
