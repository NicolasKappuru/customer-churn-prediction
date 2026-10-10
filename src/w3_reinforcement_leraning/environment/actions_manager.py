"""Determine and apply valid stochastic customer-retention actions."""

from __future__ import annotations

import random
from numbers import Integral

import pandas as pd

from w3_reinforcement_leraning.environment.constants import (
    ACTION_DEFINITIONS,
    INTERNET_DEPENDENT_SERVICES,
    RAW_FEATURES,
    SERVICE_PRICES,
)


class ActionsManager:
    """Return valid action indices and apply accepted offers to a customer row."""

    def get_valid_actions(self, df_customer_data: pd.DataFrame) -> list[int]:
        """Return valid zero-based action indices for a raw one-row customer DataFrame."""
        customer = self._get_customer_row(df_customer_data)
        valid_actions = [2, 3, 4]

        if customer["Contract"] == "Month-to-month":
            valid_actions.extend((0, 1))

        if customer["PhoneService"] == "No":
            valid_actions.append(5)
        elif customer["MultipleLines"] == "No":
            valid_actions.append(6)

        internet_service = customer["InternetService"]
        if internet_service == "No":
            valid_actions.extend((7, 8))
        elif internet_service in {"DSL", "Fiber optic"}:
            for action_index, service in enumerate(
                INTERNET_DEPENDENT_SERVICES, start=9
            ):
                if customer[service] == "No":
                    valid_actions.append(action_index)
        else:
            raise ValueError(
                "InternetService must be 'No', 'DSL', or 'Fiber optic'."
            )

        return sorted(valid_actions)

    def apply_action(
        self, df_customer_data: pd.DataFrame, action: int
    ) -> tuple[pd.DataFrame, float]:
        """Return the resulting customer row and the cost of an accepted action."""
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
            return df_customer_data, 0.0

        updated_customer = df_customer_data.copy(deep=True)
        if action_index == 0:
            updated_customer.loc[:, "Contract"] = "One year"
        elif action_index == 1:
            updated_customer.loc[:, "Contract"] = "Two year"
        elif action_index in (2, 3, 4):
            discount = action_definition.cost
            updated_customer.loc[:, "MonthlyCharges"] = (
                pd.to_numeric(updated_customer["MonthlyCharges"], errors="raise")
                * (1 - discount)
            )
        elif action_index == 5:
            updated_customer.loc[:, "PhoneService"] = "Yes"
            updated_customer.loc[:, "MultipleLines"] = "No"
            updated_customer.loc[:, "MonthlyCharges"] = (
                pd.to_numeric(updated_customer["MonthlyCharges"], errors="raise")
                + SERVICE_PRICES[action_index]
            )
        elif action_index == 6:
            updated_customer.loc[:, "MultipleLines"] = "Yes"
            updated_customer.loc[:, "MonthlyCharges"] = (
                pd.to_numeric(updated_customer["MonthlyCharges"], errors="raise")
                + SERVICE_PRICES[action_index]
            )
        elif action_index in (7, 8):
            updated_customer.loc[:, "InternetService"] = (
                "DSL" if action_index == 7 else "Fiber optic"
            )
            updated_customer.loc[:, list(INTERNET_DEPENDENT_SERVICES)] = "No"
            updated_customer.loc[:, "MonthlyCharges"] = (
                pd.to_numeric(updated_customer["MonthlyCharges"], errors="raise")
                + SERVICE_PRICES[action_index]
            )
        else:
            service = INTERNET_DEPENDENT_SERVICES[action_index - 9]
            updated_customer.loc[:, service] = "Yes"
            updated_customer.loc[:, "MonthlyCharges"] = (
                pd.to_numeric(updated_customer["MonthlyCharges"], errors="raise")
                + SERVICE_PRICES[action_index]
            )

        return updated_customer, action_definition.cost

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