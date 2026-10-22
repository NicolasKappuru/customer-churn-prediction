"""Determine and apply valid stochastic customer-retention actions for KKBox."""

from __future__ import annotations

import copy
import random
from numbers import Integral

from constants import (
    ACTION_DEFINITIONS,
    DEFAULT_PAYMENT_METHOD_ID,
    DEFAULT_PLAN_DAYS,
    DEFAULT_PLAN_PRICE,
    RENEWAL_TRANSACTION_DATE,
    RENEWAL_EXPIRE_DATE,
    UPGRADE_PLAN_DAYS,
    UPGRADE_PLAN_PRICE,
    UPGRADE_DISCOUNT_RATIO,
    UPGRADE_TRANSACTION_DATE,
    UPGRADE_EXPIRE_DATE,
    DISCOUNT_RATIO,
    DISCOUNT_TRANSACTION_DATE,
    DISCOUNT_EXPIRE_DATE,
    ENGAGEMENT_LOGS,
)


class ActionsManager:
    """Return valid action indices and apply accepted retention offers to a KKBox customer dict."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_valid_actions(self, customer: dict) -> list[int]:
        """Return valid zero-based action indices for the customer's current state."""
        last_tx = self._get_last_transaction(customer)
        valid = []

        # Action 0: renewal_offer — valid if subscription was canceled or auto-renew is off
        if last_tx is None or last_tx["is_cancel"] == 1 or last_tx["is_auto_renew"] == 0:
            valid.append(0)

        # Action 1: annual_plan_engagement — valid if not already on an annual plan
        if last_tx is None or int(last_tx["payment_plan_days"]) < UPGRADE_PLAN_DAYS:
            valid.append(1)

        # Action 2: discount_offer — valid for any customer without an existing discount.
        # Acceptance probability is dynamic: lower for lapsed/cancelled customers.
        if last_tx is None or float(last_tx["actual_amount_paid"]) >= float(last_tx["plan_list_price"]):
            valid.append(2)

        return valid if valid else [0]

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

        action_definition = ACTION_DEFINITIONS[action_index]
        acceptance = self._get_acceptance_probability(action_index, customer)
        if random.random() >= acceptance:
            return customer, 0.0

        updated = copy.deepcopy(customer)
        _dispatch = {
            0: self._apply_renewal_offer,
            1: self._apply_annual_plan_engagement,
            2: self._apply_discount_offer,
        }
        _dispatch[action_index](updated)
        return updated, action_definition.cost

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _apply_renewal_offer(self, customer: dict) -> None:
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

    def _apply_annual_plan_engagement(self, customer: dict) -> None:
        last_tx = self._get_last_transaction(customer)
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": UPGRADE_PLAN_DAYS,
            "plan_list_price": UPGRADE_PLAN_PRICE,
            "actual_amount_paid": UPGRADE_PLAN_PRICE * (1.0 - UPGRADE_DISCOUNT_RATIO),
            "is_auto_renew": 1,
            "is_cancel": 0,
            "transaction_date": UPGRADE_TRANSACTION_DATE,
            "membership_expire_date": UPGRADE_EXPIRE_DATE,
        })
        customer["user_logs"].extend(ENGAGEMENT_LOGS)

    def _apply_discount_offer(self, customer: dict) -> None:
        last_tx = self._get_last_transaction(customer)
        price = float(last_tx["plan_list_price"]) if last_tx else DEFAULT_PLAN_PRICE
        customer["transactions"].append({
            "payment_method_id": int(last_tx["payment_method_id"]) if last_tx else DEFAULT_PAYMENT_METHOD_ID,
            "payment_plan_days": int(last_tx["payment_plan_days"]) if last_tx else DEFAULT_PLAN_DAYS,
            "plan_list_price": price,
            "actual_amount_paid": price * (1.0 - DISCOUNT_RATIO),
            "is_auto_renew": int(last_tx["is_auto_renew"]) if last_tx else 0,
            "is_cancel": int(last_tx["is_cancel"]) if last_tx else 0,
            "transaction_date": DISCOUNT_TRANSACTION_DATE,
            "membership_expire_date": DISCOUNT_EXPIRE_DATE,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_acceptance_probability(self, action_index: int, customer: dict) -> float:
        action_definition = ACTION_DEFINITIONS[action_index]
        if action_definition.acceptance_probability_lapsed is None:
            return action_definition.acceptance_probability
        last_tx = self._get_last_transaction(customer)
        if last_tx is not None and (
            int(last_tx["is_cancel"]) == 1 or int(last_tx["is_auto_renew"]) == 0
        ):
            return action_definition.acceptance_probability_lapsed
        return action_definition.acceptance_probability

    def _get_last_transaction(self, customer: dict) -> dict | None:
        txns = customer.get("transactions") or []
        if not txns:
            return None
        return max(txns, key=lambda t: int(t["transaction_date"]))
