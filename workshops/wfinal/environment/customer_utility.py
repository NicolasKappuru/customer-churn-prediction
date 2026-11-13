"""Simple is-favorable-based customer utility."""
from __future__ import annotations
import numpy as np
from constants import CUSTOMER_GAIN, CUSTOMER_LOSS, NOISE_STD


class CustomerUtility:
    def __init__(self, actions_manager) -> None:
        self._am = actions_manager

    def utility(self, action: int, customer) -> float:
        """Deterministic utility (used by oracle)."""
        return CUSTOMER_GAIN[action] if self._am.is_favorable(action, customer) else -CUSTOMER_LOSS[action]

    def sample_utility(self, action: int, customer, rng) -> float:
        """Utility with Gaussian noise (used as reward signal)."""
        return self.utility(action, customer) + float(rng.normal(0.0, NOISE_STD))
