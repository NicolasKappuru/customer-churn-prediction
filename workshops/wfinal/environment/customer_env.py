"""Customer-side Gym environment for training the customer acceptance agent."""
from __future__ import annotations
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _p in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import (
    ACTION_DEFINITIONS, DO_NOTHING_ACTION, N_MODEL_FEATURES,
    CUSTOMER_N_ACTIONS, CUSTOMER_OBS_SIZE
)
from env import KKBoxRetentionEnv
from customer_utility import CustomerUtility, NOISE_STD


_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 6: all except Do nothing


class CustomerEnv(gym.Env):
    """Gym environment where the agent plays the customer role.

    The inner KKBoxRetentionEnv uses customer_mode='static' (acceptance is not used).
    At each step the customer observes a randomly-chosen valid offer and decides to
    accept (1) or reject (0).

    Observation (67 floats):
        59 preprocessed features + 6 one-hot offer encoding + turn/3 + n_rejections/3
    Action space: Discrete(2)  — 0=reject, 1=accept
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self._inner = KKBoxRetentionEnv(customer_mode="static")
        self._utility = self._inner._customer_utility
        self._rng = np.random.default_rng()
        self._current_offer: int = 0
        self._features_row = None
        self.action_space = spaces.Discrete(CUSTOMER_N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(CUSTOMER_OBS_SIZE,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        with redirect_stdout(io.StringIO()):
            _, _ = self._inner.reset(seed=seed)
        self._features_row = self._get_features()
        offer = self._pick_offer()
        if offer is None:
            return np.zeros(CUSTOMER_OBS_SIZE, dtype=np.float32), {"offer": None}
        self._current_offer = offer
        obs = self.build_customer_observation(
            self._features_row, offer, self._inner.turn, self._inner.n_rejections
        )
        return obs.astype(np.float32), {"offer": offer}

    def step(self, customer_action: int):
        offer = self._current_offer
        accepted = (customer_action == 1)
        with redirect_stdout(io.StringIO()):
            _, _, terminated, truncated, info = self._inner.step_with_decision(offer, accepted)

        u_final = self._utility.sample_utility(
            offer, self._features_row, self._inner.turn, self._rng
        ) if accepted else 0.0
        reward = u_final

        self._features_row = self._get_features()
        next_offer = self._pick_offer()
        if next_offer is None or terminated or truncated:
            obs = np.zeros(CUSTOMER_OBS_SIZE, dtype=np.float32)
            terminated = True
        else:
            self._current_offer = next_offer
            obs = self.build_customer_observation(
                self._features_row, next_offer, self._inner.turn, self._inner.n_rejections
            ).astype(np.float32)

        return obs, float(reward), terminated, truncated, {
            "offer": offer,
            "utility_no_noise": self._utility.utility(offer, self._features_row, self._inner.turn),
            "accepted": accepted,
        }

    @staticmethod
    def build_customer_observation(
        features_row,  # pd.Series with 59 features
        offer: int,
        turn: int,
        n_rejections: int,
    ) -> np.ndarray:
        """Build the 67-dim customer observation vector."""
        feat = np.array(list(features_row.values), dtype=np.float32)  # 59
        one_hot = np.zeros(_N_OFFERS, dtype=np.float32)
        if 0 <= offer < _N_OFFERS:
            one_hot[offer] = 1.0
        scalar = np.array([turn / 3.0, n_rejections / 3.0], dtype=np.float32)
        return np.concatenate([feat, one_hot, scalar])

    def _get_features(self):
        with redirect_stdout(io.StringIO()):
            df = self._inner.preprocessor_customer_data.preprocess_customer(
                self._inner.customer_data
            )
        return df.iloc[0]

    def _pick_offer(self):
        """Pick a random valid offer (excludes Do nothing). Returns None if none available."""
        valid = self._inner.get_valid_actions()
        offers = [a for a in valid if a != DO_NOTHING_ACTION]
        if not offers:
            return None
        return int(self._rng.choice(offers))

    def close(self):
        self._inner.close()
        super().close()
