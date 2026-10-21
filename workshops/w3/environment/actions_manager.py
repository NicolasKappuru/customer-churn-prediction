"""Determine and apply valid stochastic customer-retention actions for KKBox."""

from __future__ import annotations

import copy
import random
from numbers import Integral

import pandas as pd

from constants import (
    ACTION_DEFINITIONS,
    DEFAULT_PAYMENT_METHOD_ID,
    DEFAULT_PLAN_DAYS,
    DEFAULT_PLAN_PRICE,
    RENEWAL_TRANSACTION_DATE,
    RENEWAL_EXPIRE_DATE,
    UPGRADE_PLAN_DAYS,
    UPGRADE_PLAN_PRICE,
    UPGRADE_TRANSACTION_DATE,
    UPGRADE_EXPIRE_DATE,
    DISCOUNT_RATIO,
    DISCOUNT_TRANSACTION_DATE,
    DISCOUNT_EXPIRE_DATE,
    ENGAGEMENT_LOG_DATE,
    ENGAGEMENT_NUM_25,
    ENGAGEMENT_NUM_50,
    ENGAGEMENT_NUM_75,
    ENGAGEMENT_NUM_985,
    ENGAGEMENT_NUM_100,
    ENGAGEMENT_NUM_UNQ,
    ENGAGEMENT_TOTAL_SECS,
    ENGAGEMENT_INACTIVITY_THRESHOLD,
)


class ActionsManager:
    """Return valid action indices and apply accepted retention offers to a KKBox customer dict."""

    _CUTOFF = pd.Timestamp("2017-01-31")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_valid_actions(self, customer: dict) -> list[int]:
        """Return valid zero-based action indices for the customer's current state."""
        last_tx = self._get_last_transaction(customer)
        last_log_date = self._get_last_log_date(customer)
        valid = []

        # Action 0: renewal_offer — valid if subscription was canceled or auto-renew is off
        if last_tx is None or last_tx["is_cancel"] == 1 or last_tx["is_auto_renew"] == 0:
            valid.append(0)

        # Action 1: plan_upgrade — valid if not already on an annual plan
        if last_tx is None or int(last_tx["payment_plan_days"]) < UPGRADE_PLAN_DAYS:
            valid.append(1)

        # Action 2: discount_offer — valid if no discount was applied on the last transaction
        if last_tx is None or float(last_tx["actual_amount_paid"]) >= float(last_tx["plan_list_price"]):
            valid.append(2)

        # Action 3: engagement_boost — valid if user has been inactive for too long
        if last_log_date is None or (self._CUTOFF - last_log_date).days > ENGAGEMENT_INACTIVITY_THRESHOLD:
            valid.append(3)

        return valid

    def apply_action(self, customer: dict, action: int) -> tuple[dict, float, float]:
        """Return the updated customer dict, action cost, and action benefit."""
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

        action_definition = ACTION_DEFINITIONS[action_index]
        if random.random() >= action_definition.acceptance_probability:
            return customer, 0.0, 0.0

        updated = copy.deepcopy(customer)
        _dispatch = {
            0: self._apply_renewal_offer,
            1: self._apply_plan_upgrade,
            2: self._apply_discount_offer,
            3: self._apply_engagement_boost,
        }
        action_benefit = _dispatch[action_index](updated)
        return updated, action_definition.cost, action_benefit

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _apply_renewal_offer(self, customer: dict) -> float:
        last_tx = self._get_last_transaction(customer)
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": int(last_tx["payment_plan_days"]) if last_tx else DEFAULT_PLAN_DAYS,
            "plan_list_price": float(last_tx["plan_list_price"]) if last_tx else DEFAULT_PLAN_PRICE,
            "actual_amount_paid": float(last_tx["plan_list_price"]) if last_tx else DEFAULT_PLAN_PRICE,
            "is_auto_renew": 1,
            "is_cancel": 0,
            "transaction_date": RENEWAL_TRANSACTION_DATE,
            "membership_expire_date": RENEWAL_EXPIRE_DATE,
        })
        return 0.0

    def _apply_plan_upgrade(self, customer: dict) -> float:
        last_tx = self._get_last_transaction(customer)
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": UPGRADE_PLAN_DAYS,
            "plan_list_price": UPGRADE_PLAN_PRICE,
            "actual_amount_paid": UPGRADE_PLAN_PRICE,
            "is_auto_renew": 1,
            "is_cancel": 0,
            "transaction_date": UPGRADE_TRANSACTION_DATE,
            "membership_expire_date": UPGRADE_EXPIRE_DATE,
        })
        return 0.0

    def _apply_discount_offer(self, customer: dict) -> float:
        last_tx = self._get_last_transaction(customer)
        price = float(last_tx["plan_list_price"]) if last_tx else DEFAULT_PLAN_PRICE
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": int(last_tx["payment_plan_days"]) if last_tx else DEFAULT_PLAN_DAYS,
            "plan_list_price": price,
            "actual_amount_paid": price * (1.0 - DISCOUNT_RATIO),
            "is_auto_renew": 1,
            "is_cancel": 0,
            "transaction_date": DISCOUNT_TRANSACTION_DATE,
            "membership_expire_date": DISCOUNT_EXPIRE_DATE,
        })
        return 0.0

    def _apply_engagement_boost(self, customer: dict) -> float:
        customer["user_logs"].append({
            "date": ENGAGEMENT_LOG_DATE,
            "num_25": ENGAGEMENT_NUM_25,
            "num_50": ENGAGEMENT_NUM_50,
            "num_75": ENGAGEMENT_NUM_75,
            "num_985": ENGAGEMENT_NUM_985,
            "num_100": ENGAGEMENT_NUM_100,
            "num_unq": ENGAGEMENT_NUM_UNQ,
            "total_secs": ENGAGEMENT_TOTAL_SECS,
        })
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_last_transaction(self, customer: dict) -> dict | None:
        txns = customer.get("transactions") or []
        if not txns:
            return None
        return max(txns, key=lambda t: int(t["transaction_date"]))

    def _get_last_log_date(self, customer: dict) -> pd.Timestamp | None:
        logs = customer.get("user_logs") or []
        if not logs:
            return None
        last_date_int = max(int(log["date"]) for log in logs)
        return pd.Timestamp(str(last_date_int))
