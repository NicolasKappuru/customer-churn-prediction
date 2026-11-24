"""Determine and apply valid log-based customer-retention actions for KKBox."""

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
    DO_NOTHING_ACTION,
    REENGAGEMENT_LOG_POOL,
    REENGAGEMENT_WINDOW_DAYS,
    DORMANT_LOG_POOL,
    DORMANT_INACTIVE_DAYS,
    DORMANT_TENURE_DAYS,
    TREND_RECOVERY_LOG_POOL,
    TREND_RECOVERY_BOOST,
    TREND_RECOVERY_MIN_DEC_LOGS,
    COMPLETION_LOG_POOL,
    LOW_COMPLETION_RATIO,
    COMPLETION_MIN_LOGINS,
    INTENSIFIER_LOG_POOL,
    LOW_INTENSITY_SECS,
    INTENSIFIER_MIN_LOGINS,
)

_CUTOFF_DATE = CUTOFF_DATE


def _date_from_int(d: int) -> datetime.date:
    return datetime.date(d // 10000, (d // 100) % 100, d % 100)


class ActionsManager:
    """Return valid action indices and apply accepted log-based retention actions."""

    _SONG_KEYS = ("num_25", "num_50", "num_75", "num_985", "num_100")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_favorable(self, action_index: int, customer: dict) -> bool:
        """Return True if the customer meets the high-acceptance condition for this action."""
        if action_index == 0:
            return self._is_declining_active(customer)
        if action_index == 1:
            return self._tenure_days(customer) >= DORMANT_TENURE_DAYS
        if action_index == 2:
            logs = customer.get("user_logs") or []
            dec_days = sum(1 for log in logs if str(int(log["date"]))[:6] == PREV_MONTH)
            return dec_days >= TREND_RECOVERY_MIN_DEC_LOGS
        if action_index == 3:
            return self._last_month_logins(customer) >= COMPLETION_MIN_LOGINS
        if action_index == 4:
            return self._last_month_logins(customer) >= INTENSIFIER_MIN_LOGINS
        return False

    def get_valid_actions(self, customer: dict) -> list[int]:
        """Return valid zero-based action indices for the customer's current state."""
        logs = customer.get("user_logs") or []
        valid = []

        # Action 0: content re-engagement — has logs and was active within 90 days
        if logs and self._days_since_last_log(customer) <= REENGAGEMENT_WINDOW_DAYS:
            valid.append(0)

        # Action 1: reactivate dormant — has logs but inactive for >7 days
        if logs and self._days_since_last_log(customer) > DORMANT_INACTIVE_DAYS:
            valid.append(1)

        # Action 2: trend recovery — January activity below December
        if self._is_declining_active(customer):
            valid.append(2)

        # Action 3: completion booster — active in January with low completion ratio
        if self._last_month_logins(customer) > 0 and self._last_month_ratio_100(customer) < LOW_COMPLETION_RATIO:
            valid.append(3)

        # Action 4: listening intensifier — active in January with short sessions
        if self._last_month_logins(customer) > 0 and self._last_month_mean_secs(customer) < LOW_INTENSITY_SECS:
            valid.append(4)

        # Action 5: do nothing — always valid
        valid.append(DO_NOTHING_ACTION)

        return valid

    def apply_action(self, customer: dict, action: int) -> tuple[dict, float]:
        """Return the updated customer dict and action cost (stochastic acceptance)."""
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer action index.")
        action_index = int(action)
        if action_index < 0 or action_index >= len(ACTION_DEFINITIONS):
            raise ValueError(f"action must be between 0 and {len(ACTION_DEFINITIONS) - 1}.")
        if action_index not in self.get_valid_actions(customer):
            raise ValueError(
                f"Action {action_index} ({ACTION_DEFINITIONS[action_index].name}) "
                "is not valid for this customer."
            )
        if action_index == DO_NOTHING_ACTION:
            return customer, 0.0

        action_definition = ACTION_DEFINITIONS[action_index]
        acceptance = self._get_acceptance_probability(action_index, customer)
        if random.random() >= acceptance:
            return customer, 0.0

        updated = copy.deepcopy(customer)
        self._dispatch(action_index, updated)
        return updated, action_definition.cost

    def apply_effect(self, customer: dict, action: int) -> tuple[dict, float]:
        """Apply the action effect unconditionally (no acceptance randomness)."""
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer action index.")
        action_index = int(action)
        if action_index < 0 or action_index >= len(ACTION_DEFINITIONS):
            raise ValueError(f"action must be between 0 and {len(ACTION_DEFINITIONS) - 1}.")
        if action_index == DO_NOTHING_ACTION:
            return customer, 0.0
        action_definition = ACTION_DEFINITIONS[action_index]
        updated = copy.deepcopy(customer)
        self._dispatch(action_index, updated)
        return updated, action_definition.cost

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action_index: int, customer: dict) -> None:
        {
            0: self._apply_content_reengagement,
            1: self._apply_reactivate_dormant,
            2: self._apply_trend_recovery,
            3: self._apply_completion_booster,
            4: self._apply_listening_intensifier,
        }[action_index](customer)

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _apply_content_reengagement(self, customer: dict) -> None:
        """Add 2–8 scaled log entries simulating a content re-engagement campaign."""
        existing_dates = {int(log["date"]) for log in (customer.get("user_logs") or [])}
        pool = [e for e in REENGAGEMENT_LOG_POOL if e["date"] not in existing_dates]
        if not pool:
            return
        n = random.randint(min(2, len(pool)), len(pool))
        selected = random.sample(pool, n)
        customer["user_logs"].extend(self._scale_logs(selected, customer))

    def _apply_reactivate_dormant(self, customer: dict) -> None:
        """Add 2–3 scaled log entries in mid-January for long-inactive customers."""
        existing_dates = {int(log["date"]) for log in (customer.get("user_logs") or [])}
        pool = [e for e in DORMANT_LOG_POOL if e["date"] not in existing_dates]
        if not pool:
            return
        n = random.randint(min(2, len(pool)), min(3, len(pool)))
        selected = random.sample(pool, n)
        customer["user_logs"].extend(self._scale_logs(selected, customer))

    def _apply_trend_recovery(self, customer: dict) -> None:
        """Add 3–5 January logs scaled to 150% of December daily average to reverse decline."""
        existing_dates = {int(log["date"]) for log in (customer.get("user_logs") or [])}
        pool = [e for e in TREND_RECOVERY_LOG_POOL if e["date"] not in existing_dates]
        if not pool:
            return
        n = random.randint(min(3, len(pool)), min(5, len(pool)))
        selected = random.sample(pool, n)
        customer["user_logs"].extend(self._scale_logs(selected, customer, boost_factor=TREND_RECOVERY_BOOST))

    def _apply_completion_booster(self, customer: dict) -> None:
        """Add 2–4 high-completion (num_100 heavy) log entries."""
        existing_dates = {int(log["date"]) for log in (customer.get("user_logs") or [])}
        pool = [e for e in COMPLETION_LOG_POOL if e["date"] not in existing_dates]
        if not pool:
            return
        n = random.randint(min(2, len(pool)), min(4, len(pool)))
        selected = random.sample(pool, n)
        customer["user_logs"].extend(self._scale_logs(selected, customer))

    def _apply_listening_intensifier(self, customer: dict) -> None:
        """Add 1–3 high-duration log entries to boost total listening time."""
        existing_dates = {int(log["date"]) for log in (customer.get("user_logs") or [])}
        pool = [e for e in INTENSIFIER_LOG_POOL if e["date"] not in existing_dates]
        if not pool:
            return
        n = random.randint(1, min(3, len(pool)))
        selected = random.sample(pool, n)
        customer["user_logs"].extend(self._scale_logs(selected, customer))

    # ------------------------------------------------------------------
    # Acceptance probability
    # ------------------------------------------------------------------

    def _get_acceptance_probability(self, action_index: int, customer: dict) -> float:
        action_def = ACTION_DEFINITIONS[action_index]
        if action_def.acceptance_probability_high is None:
            return action_def.acceptance_probability
        return (
            action_def.acceptance_probability_high
            if self.is_favorable(action_index, customer)
            else action_def.acceptance_probability
        )

    # ------------------------------------------------------------------
    # Condition helpers (computed from raw customer dict)
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

    def _last_month_ratio_100(self, customer: dict) -> float:
        logs = customer.get("user_logs") or []
        jan_logs = [log for log in logs if str(int(log["date"]))[:6] == CUTOFF_MONTH]
        if not jan_logs:
            return 0.0
        total = sum(
            sum(int(log.get(k, 0)) for k in self._SONG_KEYS)
            for log in jan_logs
        )
        num_100 = sum(int(log.get("num_100", 0)) for log in jan_logs)
        return num_100 / total if total > 0 else 0.0

    def _last_month_mean_secs(self, customer: dict) -> float:
        logs = customer.get("user_logs") or []
        jan_logs = [log for log in logs if str(int(log["date"]))[:6] == CUTOFF_MONTH]
        if not jan_logs:
            return 0.0
        return sum(float(log.get("total_secs", 0.0)) for log in jan_logs) / len(jan_logs)

    def _is_declining_active(self, customer: dict) -> bool:
        logs = customer.get("user_logs") or []
        if not any(LOG_WINDOW_START <= int(log["date"]) <= LOG_WINDOW_END for log in logs):
            return False

        def _songs(prefix: str) -> int:
            return sum(
                sum(int(log.get(k, 0)) for k in self._SONG_KEYS)
                for log in logs if str(int(log["date"]))[:6] == prefix
            )

        return (_songs(CUTOFF_MONTH) / (_songs(PREV_MONTH) + 1)) < 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scale_logs(self, entries: list[dict], customer: dict, boost_factor: float = 1.0) -> list[dict]:
        """Scale log song counts to customer's December daily average × boost_factor."""
        logs = customer.get("user_logs") or []
        dec_logs = [log for log in logs if str(int(log["date"]))[:6] == PREV_MONTH]
        if not dec_logs:
            return list(entries)
        dec_songs = sum(sum(int(log.get(k, 0)) for k in self._SONG_KEYS) for log in dec_logs)
        dec_avg = dec_songs / len(dec_logs) * boost_factor
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
