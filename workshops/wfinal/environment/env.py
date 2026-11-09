"""Gymnasium environment for KKBox customer retention."""

from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from constants import ACTION_DEFINITIONS, N_OBSERVATION, DO_NOTHING_ACTION, MAX_STEPS_PER_EPISODE, TURN_PENALTY
from actions_manager import ActionsManager
from supervised_model.customer_selector import CustomerSelector
from supervised_model.preprocessor_customer_data import PreprocessorCustomerData
from supervised_model.supervised_model import SupervisedModel
from acceptance import StaticAcceptance, CustomerAgentAcceptance
from customer_utility import CustomerUtility


class KKBoxRetentionEnv(gym.Env):
    """Gymnasium environment for KKBox customer retention.

    Each episode corresponds to up to 3 turns with one churned customer.
    The agent selects retention actions; the reward per turn is the reduction in
    predicted churn probability, discounted by action cost.

    Action space:  Discrete(7)
        0 — Send auto-renewal reminder
        1 — Offer loyalty renewal with discount
        2 — Send content re-engagement
        3 — Offer downgrade to 99 NTD plan
        4 — Recover cancelled subscription
        5 — Win-back inactive listener
        6 — Do nothing

    Observation space: Box(shape=(60,), dtype=float32)
        59 preprocessed XGBoost features + 1 churn probability
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        render_delay: float = 2.0,
        customer_mode: str = "static",
        customer_checkpoint: str | Path = "checkpoints/customer_agent_ppo.pt",
    ) -> None:
        super().__init__()
        self.customer_selector = CustomerSelector()
        self.preprocessor_customer_data = PreprocessorCustomerData()
        self.supervised_model = SupervisedModel()
        self.action_manager = ActionsManager()

        self.customer_data: dict | None = None
        self.probability_churn_before: float = 0.0
        self.probability_churn_after: float = 0.0
        self.turn: int = 0
        self.accepted_actions: set[int] = set()
        self.n_rejections: int = 0

        # Observation cache: list of (observation, churn_probability) per pool index.
        # Auto-rebuilt whenever the pool size changes (e.g. after train/val split).
        self._obs_cache: list[tuple[np.ndarray, float]] | None = None

        self._rng = np.random.default_rng()
        self._customer_utility = CustomerUtility()

        if customer_mode == "static":
            self._acceptance_policy = StaticAcceptance(self.action_manager)
        elif customer_mode == "agent":
            self._acceptance_policy = CustomerAgentAcceptance(customer_checkpoint)
        else:
            raise ValueError(f"customer_mode must be 'static' or 'agent', got '{customer_mode}'.")

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
        self.turn = 0
        self.accepted_actions = set()
        self.n_rejections = 0

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
        all_features: list[pd.DataFrame] = []
        for customer in pool:
            with redirect_stdout(io.StringIO()):
                features = self.preprocessor_customer_data.preprocess_customer(customer)
                prob = float(self.supervised_model.predict_probability(features))
            obs_df = features.copy()
            obs_df["churn_probability"] = prob
            cache.append((obs_df.to_numpy(dtype=np.float32).flatten(), prob))
            all_features.append(features)
        self._obs_cache = cache
        if all_features:
            pool_df = pd.concat(all_features, ignore_index=True)
            self._customer_utility.fit(pool_df)

    def get_valid_actions(self) -> list[int]:
        """Return valid actions for the current state, excluding already-accepted ones."""
        all_valid = self.action_manager.get_valid_actions(self.customer_data)
        return [a for a in all_valid if a == DO_NOTHING_ACTION or a not in self.accepted_actions]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if action == DO_NOTHING_ACTION:
            accepted = False
        else:
            with redirect_stdout(io.StringIO()):
                features_df = self.preprocessor_customer_data.preprocess_customer(self.customer_data)
            features_row = features_df.iloc[0]
            accepted = self._acceptance_policy.decide(
                self.customer_data,
                features_row,
                action,
                self.turn,
                self.n_rejections,
                self._rng,
            )
            if not accepted:
                self.n_rejections += 1

        return self.step_with_decision(action, accepted)

    def step_with_decision(self, action: int, accepted: bool) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute a step given a pre-determined accept/reject decision."""
        customer_before = self.customer_data

        if action == DO_NOTHING_ACTION or not accepted:
            action_cost = 0.0
        else:
            self.customer_data, action_cost = self.action_manager.apply_effect(
                self.customer_data, action
            )

        customer_features = self.preprocessor_customer_data.preprocess_customer(self.customer_data)
        self.probability_churn_after = float(self.supervised_model.predict_probability(customer_features))

        if self.customer_data is customer_before:
            self.probability_churn_after = self.probability_churn_before

        observation_df = customer_features.copy()
        observation_df["churn_probability"] = self.probability_churn_after
        observation = observation_df.to_numpy(dtype=np.float32).flatten()

        if accepted and action != DO_NOTHING_ACTION:
            self.accepted_actions.add(action)

        prob_before_this_step = self.probability_churn_before
        churn_disminution = self.probability_churn_before - self.probability_churn_after
        reward = churn_disminution * (1 - action_cost) - TURN_PENALTY[self.turn]
        self.probability_churn_before = self.probability_churn_after

        # Customer reward: utility if accepted, else 0.0
        if accepted and action != DO_NOTHING_ACTION:
            with redirect_stdout(io.StringIO()):
                features_df = self.preprocessor_customer_data.preprocess_customer(customer_before)
            features_row = features_df.iloc[0]
            customer_reward = self._customer_utility.utility(action, features_row, self.turn + 1)
        else:
            customer_reward = 0.0

        self.turn += 1
        valid_actions_next = self.get_valid_actions()
        terminated = (
            action == DO_NOTHING_ACTION
            or self.turn >= MAX_STEPS_PER_EPISODE
            or valid_actions_next == [DO_NOTHING_ACTION]
        )
        truncated = False

        info = {
            "probability_before": prob_before_this_step,
            "probability_after": self.probability_churn_after,
            "action": action,
            "action_cost": action_cost,
            "accepted": accepted,
            "turn": self.turn,
            "valid_actions": [] if terminated else valid_actions_next,
            "customer_reward": customer_reward,
        }

        if self._bridge is not None:
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
