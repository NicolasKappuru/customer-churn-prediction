"""Interchangeable acceptance policies for KKBoxRetentionEnv."""
from __future__ import annotations
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from actions_manager import ActionsManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _p in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import ACTION_DEFINITIONS


class AcceptancePolicy:
    """Base class â€” decide whether a customer accepts a retention offer."""

    def decide(
        self,
        customer: dict,
        features_row,  # pd.Series
        action: int,
        turn: int,
        n_rejections: int,
        rng: np.random.Generator,
    ) -> bool:
        raise NotImplementedError


class StaticAcceptance(AcceptancePolicy):
    """Uses the fixed stochastic probabilities defined in ACTION_DEFINITIONS."""

    def __init__(self, action_manager: "ActionsManager") -> None:
        self._am = action_manager

    def decide(self, customer, features_row, action, turn, n_rejections, rng) -> bool:
        p = self._am._get_acceptance_probability(action, customer)
        return bool(rng.random() < p)


class CustomerAgentAcceptance(AcceptancePolicy):
    """Samples accept/reject from a trained CustomerPPOAgent policy."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        from customer_env import CustomerEnv  # lazy import to avoid circular
        self._checkpoint_path = Path(checkpoint_path)
        self._agent = None  # loaded lazily

    def _load_agent(self) -> None:
        if self._agent is not None:
            return
        sys.path.insert(0, str(W3_ROOT / "agent"))
        from customer_agent_ppo import CustomerPPOAgent
        self._agent, _ = CustomerPPOAgent.load_checkpoint(self._checkpoint_path)

    def decide(self, customer, features_row, action, turn, n_rejections, rng,
               times_offered: int = 0, customer_data=None) -> bool:
        self._load_agent()
        from customer_env import CustomerEnv
        obs = CustomerEnv.build_customer_observation(
            features_row, action,
            customer_data=customer_data,
        )
        p = self._agent.accept_probability(obs)
        return bool(rng.random() < p)
