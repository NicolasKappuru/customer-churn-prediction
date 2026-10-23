"""Gymnasium environment for KKBox customer retention."""

from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stdout

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

    Action space:  Discrete(3)
        0 — renewal_offer             add a monthly renewal transaction (auto-renew restored)
        1 — annual_plan_engagement    add an annual plan transaction + engagement logs
        2 — discount_offer            add a 20% discounted renewal transaction

    Observation space: Box(shape=(60,), dtype=float32)
        59 preprocessed XGBoost features + 1 churn probability
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None, render_delay: float = 2.0) -> None:
        super().__init__()
        self.customer_selector = CustomerSelector()
        self.preprocessor_customer_data = PreprocessorCustomerData()
        self.supervised_model = SupervisedModel()
        self.action_manager = ActionsManager()

        self.customer_data: dict | None = None
        self.probability_churn_before: float = 0.0
        self.probability_churn_after: float = 0.0

        # Observation cache: list of (observation, churn_probability) per pool index.
        # Auto-rebuilt whenever the pool size changes (e.g. after train/val split).
        self._obs_cache: list[tuple[np.ndarray, float]] | None = None

        self.render_mode = render_mode
        self._bridge = None
        if render_mode == "human":
            if self._display_available():
                from render.bridge import KKBoxRenderBridge
                self._bridge = KKBoxRenderBridge(render_delay=render_delay)
            else:
                warnings.warn(
                    "render_mode='human' requested but no display found "
                    "(DISPLAY/WAYLAND_DISPLAY not set). Rendering disabled. "
                    "Use 'export DISPLAY=:0' or 'ssh -X' to enable it.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.action_space = spaces.Discrete(len(ACTION_DEFINITIONS))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(N_OBSERVATION,),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        pool = self.customer_selector._pool
        if self._obs_cache is None or len(self._obs_cache) != len(pool):
            self._build_obs_cache()

        idx = int(self.np_random.integers(0, len(pool)))
        self.customer_data = pool[idx]
        observation, self.probability_churn_before = self._obs_cache[idx]

        if self._bridge is not None:
            with redirect_stdout(io.StringIO()):
                features_df = self.preprocessor_customer_data.preprocess_customer(self.customer_data)
            self._bridge.on_reset(
                features=dict(features_df.iloc[0]),
                churn_prob=self.probability_churn_before,
            )

        return observation, {}

    def _build_obs_cache(self) -> None:
        pool = self.customer_selector._pool
        cache = []
        for customer in pool:
            with redirect_stdout(io.StringIO()):
                features = self.preprocessor_customer_data.preprocess_customer(customer)
                prob = float(self.supervised_model.predict_probability(features))
            obs_df = features.copy()
            obs_df["churn_probability"] = prob
            cache.append((obs_df.to_numpy(dtype=np.float32).flatten(), prob))
        self._obs_cache = cache

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.customer_data, action_cost = self.action_manager.apply_action(
            self.customer_data, action
        )

        customer_features = self.preprocessor_customer_data.preprocess_customer(self.customer_data)
        self.probability_churn_after = self.supervised_model.predict_probability(customer_features)

        observation_df = customer_features.copy()
        observation_df["churn_probability"] = self.probability_churn_after
        observation = observation_df.to_numpy(dtype=np.float32).flatten()

        churn_disminution = self.probability_churn_before - self.probability_churn_after
        reward = churn_disminution * (1 - action_cost)

        terminated = True
        truncated = False
        info = {
            "probability_before": self.probability_churn_before,
            "probability_after": self.probability_churn_after,
            "action": action,
            "action_cost": action_cost,
        }

        if self._bridge is not None:
            accepted = float(self.probability_churn_after) < float(self.probability_churn_before)
            self._bridge.on_step(
                new_features=dict(customer_features.iloc[0]),
                new_churn=float(self.probability_churn_after),
                action=action,
                reward=float(reward),
                accepted=accepted,
            )

        return observation, reward, terminated, truncated, info

    def render(self) -> bool:
        """Pump pygame events and redraw. Returns False when the window is closed."""
        if self._bridge is not None:
            return self._bridge.tick()
        return True

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
        super().close()

    @staticmethod
    def _display_available() -> bool:
        if sys.platform == "win32":
            return True
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False
        import subprocess
        try:
            result = subprocess.run(
                ["xdpyinfo"],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
