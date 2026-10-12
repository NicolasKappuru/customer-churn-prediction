"""Determine and apply valid stochastic customer-retention actions."""

from __future__ import annotations

import random
from numbers import Integral

import pandas as pd

from w3_reinforcement_leraning.environment.constants import (
    ACTION_DEFINITIONS,
    RAW_FEATURES,
    DSL_SERVICE_PRICE,
    DISCOUNT_PERCENTAGE,
    OTHER_ACTION_BENEFIT,
    SECURE_PLAN_BENEFIT_PER_SERVICE,
    SECURE_PLAN_SERVICE_PRICE,
    SECURE_PLAN_SERVICES,
)


class ActionsManager:
    """Return valid action indices and apply accepted offers to a customer row."""

    def get_valid_actions(self, df_customer_data: pd.DataFrame) -> list[int]:
        """Return valid zero-based action indices for a raw one-row customer DataFrame."""
        customer = self._get_customer_row(df_customer_data)
        valid_actions = []

        if customer["Contract"] in {"Month-to-month", "One year"}:
            valid_actions.append(0)

        internet_service = customer["InternetService"]
        if internet_service not in {"No", "DSL", "Fiber optic"}:
            raise ValueError(
                "InternetService must be 'No', 'DSL', or 'Fiber optic'."
            )

        if internet_service == "No" or any(
            customer[service] != "Yes" for service in SECURE_PLAN_SERVICES
        ):
            valid_actions.append(1)

        if (
            customer["PaperlessBilling"] != "No"
            or customer["PaymentMethod"] != "Credit card (automatic)"
        ):
            valid_actions.append(2)

        valid_actions.append(3)
        return sorted(valid_actions)

    def apply_action(
        self, df_customer_data: pd.DataFrame, action: int
    ) -> tuple[pd.DataFrame, float, float]:
        """Return the updated customer, action cost, and action benefit."""
        customer = self._get_customer_row(df_customer_data)
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer action index.")
        action_index = int(action)
        if action_index < 0 or action_index >= len(ACTION_DEFINITIONS):
            raise ValueError(
                f"action must be between 0 and {len(ACTION_DEFINITIONS) - 1}."
            )
        if action_index not in self.get_valid_actions(df_customer_data):
            raise ValueError(
                f"Action {action_index} ({ACTION_DEFINITIONS[action_index].name}) "
                "is not valid for this customer."
            )

        action_definition = ACTION_DEFINITIONS[action_index]
        if random.random() >= action_definition.acceptance_probability:
            return df_customer_data, 0.0, 0.0

        updated_customer = df_customer_data.copy(deep=True)
        action_benefit = OTHER_ACTION_BENEFIT
        if action_index == 0:
            updated_customer.loc[:, "Contract"] = (
                "One year"
                if customer["Contract"] == "Month-to-month"
                else "Two year"
            )
        elif action_index == 1:
            if customer["InternetService"] == "No":
                updated_customer.loc[:, "InternetService"] = "DSL"
                monthly_charges = pd.to_numeric(
                    updated_customer["MonthlyCharges"], errors="raise"
                )
                updated_customer.loc[:, "MonthlyCharges"] = (
                    monthly_charges + DSL_SERVICE_PRICE
                )

            missing_services = [
                service
                for service in SECURE_PLAN_SERVICES
                if customer[service] != "Yes"
            ]
            if missing_services:
                updated_customer.loc[:, missing_services] = "Yes"
                monthly_charges = pd.to_numeric(
                    updated_customer["MonthlyCharges"], errors="raise"
                )
                updated_customer.loc[:, "MonthlyCharges"] = (
                    monthly_charges
                    + SECURE_PLAN_SERVICE_PRICE * len(missing_services)
                )
                action_benefit = (
                    SECURE_PLAN_BENEFIT_PER_SERVICE * len(missing_services)
                )
            else:
                action_benefit = 0.0
        elif action_index == 2:
            updated_customer.loc[:, "PaperlessBilling"] = "No"
            updated_customer.loc[:, "PaymentMethod"] = "Credit card (automatic)"
        else:
            updated_customer.loc[:, "MonthlyCharges"] = (
                pd.to_numeric(updated_customer["MonthlyCharges"], errors="raise")
                * (1 - DISCOUNT_PERCENTAGE)
            )

        return updated_customer, action_definition.cost, action_benefit

    @staticmethod
    def _get_customer_row(df_customer_data: pd.DataFrame) -> pd.Series:
        if not isinstance(df_customer_data, pd.DataFrame):
            raise TypeError("df_customer_data must be a pandas DataFrame.")
        if len(df_customer_data) != 1:
            raise ValueError("df_customer_data must contain exactly one customer row.")

        missing_features = [
            feature for feature in RAW_FEATURES if feature not in df_customer_data.columns
        ]
        if missing_features:
            raise ValueError(
                "df_customer_data is missing required features: "
                + ", ".join(missing_features)
            )
        return df_customer_data.iloc[0]