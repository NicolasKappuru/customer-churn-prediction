"""Gymnasium environment for KKBox customer retention."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from constants import ACTION_DEFINITIONS, N_OBSERVATION
from actions_manager import ActionsManager
from supervised_model.customer_selector import CustomerSelector
from supervised_model.preprocessor_customer_data import PreprocessorCustomerData
from supervised_model.supervised_model import SupervisedModel


class KKBoxRetentionEnv(gym.Env):
    """Gymnasium environment for KKBox customer retention.

    Each episode corresponds to a single interaction with one churned customer.
    The agent selects one retention action; the reward is the reduction in
    predicted churn probability, discounted by action cost.

    Action space:  Discrete(4)
        0 — renewal_offer       add a monthly renewal transaction (auto-renew restored)
        1 — plan_upgrade        add an annual plan transaction
        2 — discount_offer      add a 50 % discounted renewal transaction
        3 — engagement_boost    add a high-activity user-log entry

    Observation space: Box(shape=(60,), dtype=float32)
        59 preprocessed XGBoost features + 1 churn probability
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.customer_selector = CustomerSelector()
        self.preprocessor_customer_data = PreprocessorCustomerData()
        self.supervised_model = SupervisedModel()
        self.action_manager = ActionsManager()

        self.customer_data: dict | None = None

        self.probability_churn_before: float = 0.0
        self.probability_churn_after: float = 0.0

        self.action_space = spaces.Discrete(len(ACTION_DEFINITIONS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(N_OBSERVATION,),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.customer_data = self.customer_selector.get_random_customer()

        customer_features = self.preprocessor_customer_data.preprocess_customer(self.customer_data)
        self.probability_churn_before = self.supervised_model.predict_probability(customer_features)

        observation_df = customer_features.copy()
        observation_df["churn_probability"] = self.probability_churn_before
        observation = observation_df.to_numpy(dtype=np.float32).flatten()

        return observation, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.customer_data, action_cost, action_benefit = self.action_manager.apply_action(
            self.customer_data, action
        )

        customer_features = self.preprocessor_customer_data.preprocess_customer(self.customer_data)
        self.probability_churn_after = self.supervised_model.predict_probability(customer_features)

        observation_df = customer_features.copy()
        observation_df["churn_probability"] = self.probability_churn_after
        observation = observation_df.to_numpy(dtype=np.float32).flatten()

        churn_disminution = self.probability_churn_before - self.probability_churn_after
        cost = 1 - action_cost
        benefit_multiplier = 1 + action_benefit

        reward = churn_disminution * cost * benefit_multiplier

        terminated = True
        truncated = False
        info = {
            "probability_before": self.probability_churn_before,
            "probability_after": self.probability_churn_after,
            "action": action,
            "action_cost": action_cost,
            "action_benefit": action_benefit,
            "benefit_multiplier": benefit_multiplier,
        }

        return observation, reward, terminated, truncated, info
