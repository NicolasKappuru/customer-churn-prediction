"""Determine and apply valid stochastic customer-retention actions for KKBox."""

from __future__ import annotations

import copy
import datetime
import random
from numbers import Integral

from constants import (
    ACTION_DEFINITIONS,
    CUTOFF_DATE,
    CUTOFF_MONTH,
    PREV_MONTH,
    LOG_WINDOW_START,
    LOG_WINDOW_END,
    DEFAULT_PAYMENT_METHOD_ID,
    DEFAULT_PLAN_DAYS,
    DEFAULT_PLAN_PRICE,
<<<<<<< HEAD
    REENGAGEMENT_LOG_POOL,
    RENEWAL_TRANSACTION_DATE,
=======
    FIDELITY_PLAN_DAYS,
    FIDELITY_PLAN_PRICE,
    FIDELITY_DISCOUNT_NTD,
    FIDELITY_TRANSACTION_DATE,
    REENGAGEMENT_LOG_POOL,
    DOWNGRADE_PLAN_PRICE,
    DOWNGRADE_PLAN_DAYS,
    DOWNGRADE_TRANSACTION_DATE,
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
)

_CUTOFF_DATE = CUTOFF_DATE

<<<<<<< HEAD
# Days-since-last-log threshold for "recently active" (Action 3 validity)
_ACTIVE_WINDOW_DAYS: int = 7
_DORMANT_WINDOW_DAYS: int = 30

=======
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760

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
<<<<<<< HEAD
            return self._tenure_days(customer) >= 365
        if action_index == 1:
            return self._days_since_last_log(customer) <= 30
        if action_index == 2:
            return self._last_month_logins(customer) >= 3
=======
            return self._days_since_last_log(customer) <= 30
        if action_index == 1:
            return self._last_month_logins(customer) >= 15
        if action_index == 2:
            return self._days_since_last_log(customer) <= 30
        if action_index == 3:
            return self._tenure_days(customer) < 365
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
        return False

    def get_valid_actions(self, customer: dict) -> list[int]:
        """Return valid zero-based action indices for the customer's current state."""
<<<<<<< HEAD
        valid = []
        logs = customer.get("user_logs") or []

        # Action 0: reactivate dormant — no logs at all, or inactive for 30+ days
        if not logs or self._days_since_last_log(customer) > _DORMANT_WINDOW_DAYS:
            valid.append(0)

        # Action 1: re-engagement — any customer with logs in the Nov-Jan window
        if self._has_window_logs(customer):
            valid.append(1)

        # Action 2: renewal nudge — actively listening but subscription lapsed
        if logs and self._days_since_last_log(customer) <= _ACTIVE_WINDOW_DAYS:
            valid.append(2)

        # Action 3: do nothing — always valid
        valid.append(3)
=======
        last_tx = self._get_last_transaction(customer)
        valid = []

        # Action 0: reminder — auto-renew off and membership expiring within the next 15 days
        if (last_tx is not None
                and int(last_tx["is_auto_renew"]) == 0
                and 0 <= (_date_from_int(int(last_tx["membership_expire_date"])) - _CUTOFF_DATE).days <= 15):
            valid.append(0)

        # Action 1: loyalty renewal — long-tenure customers with auto-renew active
        if self._tenure_days(customer) > 730 and last_tx is not None and int(last_tx["is_auto_renew"]) == 1:
            valid.append(1)

        # Action 2: reactivation — only when activity is declining
        if self._is_declining_active(customer):
            valid.append(2)

        # Action 3: downgrade — only when current plan is more expensive than 99 NTD
        if last_tx is not None and float(last_tx["plan_list_price"]) > DOWNGRADE_PLAN_PRICE:
            valid.append(3)

        # Action 4: do nothing — always valid
        valid.append(4)
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760

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

<<<<<<< HEAD
        if action_index == 3:
=======
        # Do nothing — no state change, no cost, skip stochastic acceptance
        if action_index == 4:
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
            return customer, 0.0

        action_definition = ACTION_DEFINITIONS[action_index]
        acceptance = self._get_acceptance_probability(action_index, customer)
        if random.random() >= acceptance:
            return customer, 0.0

        updated = copy.deepcopy(customer)
        _dispatch = {
<<<<<<< HEAD
            0: self._apply_reactivate_dormant,
            1: self._apply_content_reengagement,
            2: self._apply_renewal_nudge,
=======
            0: self._apply_reminder_autorenew,
            1: self._apply_fidelity_renewal,
            2: self._apply_content_reengagement,
            3: self._apply_downgrade,
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
        }
        _dispatch[action_index](updated)
        return updated, action_definition.cost

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

<<<<<<< HEAD
    def _apply_reactivate_dormant(self, customer: dict) -> None:
        """Add 2-3 January log entries for customers with no listening history in the window."""
        n = random.randint(2, 3)
        selected = random.sample(list(REENGAGEMENT_LOG_POOL), n)
        customer["user_logs"].extend([dict(e) for e in selected])

    def _apply_content_reengagement(self, customer: dict) -> None:
        """Add 2-8 scaled log entries simulating a content re-engagement campaign."""
        n = random.randint(2, len(REENGAGEMENT_LOG_POOL))
        selected = random.sample(list(REENGAGEMENT_LOG_POOL), n)
        customer["user_logs"].extend(self._scale_logs(selected, customer))

    def _apply_renewal_nudge(self, customer: dict) -> None:
        """Add a renewal transaction that changes last_is_cancel 1→0 without extending expiry.

        Preserves days_to_expire by reusing the existing membership_expire_date.
        """
        last_tx = self._get_last_transaction(customer)
        price = min(
            float(last_tx["plan_list_price"]) if last_tx else DEFAULT_PLAN_PRICE,
            DEFAULT_PLAN_PRICE,
        )
        expire_date = (
            int(last_tx["membership_expire_date"]) if last_tx
            else RENEWAL_TRANSACTION_DATE
        )
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": DEFAULT_PLAN_DAYS,
            "plan_list_price": price,
            "actual_amount_paid": price,
            "is_auto_renew": 0,
            "is_cancel": 0,
            "transaction_date": RENEWAL_TRANSACTION_DATE,
            "membership_expire_date": expire_date,
=======
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
        """Add 2–8 scaled log entries simulating a content re-engagement campaign."""
        n = random.randint(2, len(REENGAGEMENT_LOG_POOL))
        selected = random.sample(REENGAGEMENT_LOG_POOL, n)
        customer["user_logs"].extend(self._scale_logs(selected, customer))

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
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
        })

    # ------------------------------------------------------------------
    # Acceptance probability
    # ------------------------------------------------------------------

    def _get_acceptance_probability(self, action_index: int, customer: dict) -> float:
        action_def = ACTION_DEFINITIONS[action_index]
        if action_def.acceptance_probability_high is None:
            return action_def.acceptance_probability

        if action_index == 0:
<<<<<<< HEAD
            high = self._tenure_days(customer) >= 365
        elif action_index == 1:
            high = self._days_since_last_log(customer) <= 30
        elif action_index == 2:
            high = self._last_month_logins(customer) >= 3
=======
            high = self._days_since_last_log(customer) <= 30
        elif action_index == 1:
            high = self._last_month_logins(customer) >= 15
        elif action_index == 2:
            high = self._days_since_last_log(customer) <= 30
        elif action_index == 3:
            high = self._tenure_days(customer) < 365
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
        else:
            high = False

        return action_def.acceptance_probability_high if high else action_def.acceptance_probability

    # ------------------------------------------------------------------
<<<<<<< HEAD
    # Condition helpers
=======
    # Condition helpers (computed from raw customer dict, no preprocessor)
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
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
        return sum(1 for log in logs if str(int(log["date"]))[:6] == CUTOFF_MONTH)

<<<<<<< HEAD
    def _has_window_logs(self, customer: dict) -> bool:
        """Return True if the customer has any log entries in the Nov–Jan window."""
        logs = customer.get("user_logs") or []
        return any(LOG_WINDOW_START <= int(log["date"]) <= LOG_WINDOW_END for log in logs)

=======
>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
    def _is_declining_active(self, customer: dict) -> bool:
        logs = customer.get("user_logs") or []
        if not any(LOG_WINDOW_START <= int(log["date"]) <= LOG_WINDOW_END for log in logs):
            return False

        def _songs(prefix: str) -> int:
            return sum(
                int(log.get("num_25", 0)) + int(log.get("num_50", 0)) +
                int(log.get("num_75", 0)) + int(log.get("num_985", 0)) + int(log.get("num_100", 0))
                for log in logs if str(int(log["date"]))[:6] == prefix
            )

        return (_songs(CUTOFF_MONTH) / (_songs(PREV_MONTH) + 1)) < 1.0

<<<<<<< HEAD
=======
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

>>>>>>> 6ab4a8f759c56beac43402964f7a82ffa919a760
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
        dec_logs = [log for log in logs if str(int(log["date"]))[:6] == PREV_MONTH]
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
