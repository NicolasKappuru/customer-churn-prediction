"""Customer-side Gym environment for training the customer acceptance agent."""
from __future__ import annotations
import datetime
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _p in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import (
    ACTION_DEFINITIONS, DO_NOTHING_ACTION, N_MODEL_FEATURES,
    CUSTOMER_N_ACTIONS, CUSTOMER_OBS_SIZE,
    REJECT_PENALTY, REJECT_BONUS,
    CUTOFF_DATE, CUTOFF_MONTH, PREV_MONTH,
)
from env import KKBoxRetentionEnv
from customer_utility import CustomerUtility


_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 5: all except Do nothing

# Normalization caps for context features
_DAYS_SINCE_CAP = 90.0    # cap days_since_last_log at 90 for normalization
_TENURE_CAP     = 1095.0  # cap tenure at ~3 years for normalization

_SONG_KEYS = ("num_25", "num_50", "num_75", "num_985", "num_100")


def _date_from_int(d: int) -> datetime.date:
    d = int(d)
    return datetime.date(d // 10000, (d // 100) % 100, d % 100)


def build_customer_observation(action_manager, features_row, offer: int, customer_data=None) -> np.ndarray:
    """Build the 58-dim customer observation (48 XGB features + 5 one-hot + 5 context).

    Module-level function so both CustomerEnv and acceptance.py can call it
    without needing a full CustomerEnv instance.
    """
    feat = np.array(list(features_row.values), dtype=np.float32)  # 48

    one_hot = np.zeros(_N_OFFERS, dtype=np.float32)
    if 0 <= offer < _N_OFFERS:
        one_hot[offer] = 1.0

    if customer_data is not None:
        logs = customer_data.get("user_logs") or []

        is_fav = float(action_manager.is_favorable(offer, customer_data))

        if logs:
            last_date = max(_date_from_int(int(log["date"])) for log in logs)
            days_since = (CUTOFF_DATE - last_date).days
        else:
            days_since = int(_DAYS_SINCE_CAP)
        days_norm = min(days_since / _DAYS_SINCE_CAP, 1.0)

        has_jan = float(
            any(str(int(log["date"]))[:6] == CUTOFF_MONTH for log in logs)
        )

        def _month_songs(prefix: str) -> int:
            return sum(
                sum(int(log.get(k, 0)) for k in _SONG_KEYS)
                for log in logs if str(int(log["date"]))[:6] == prefix
            )
        is_declining = float(_month_songs(CUTOFF_MONTH) < _month_songs(PREV_MONTH)
                             and _month_songs(PREV_MONTH) > 0)

        rit = customer_data.get("registration_init_time")
        if rit is not None:
            tenure_days = (CUTOFF_DATE - _date_from_int(int(rit))).days
            tenure_norm = min(tenure_days / _TENURE_CAP, 1.0)
        else:
            tenure_norm = 0.0
    else:
        is_fav = 0.0
        days_norm = 0.5
        has_jan = 0.0
        is_declining = 0.0
        tenure_norm = 0.0

    context = np.array([is_fav, days_norm, has_jan, is_declining, tenure_norm], dtype=np.float32)
    return np.concatenate([feat, one_hot, context])


class CustomerEnv(gym.Env):
    """Gym environment where the agent plays the customer role.

    Each episode = one customer, one offer, one accept/reject decision.

    Observation (58 floats):
        48 preprocessed XGBoost features
        + 5 one-hot offer encoding  (actions 0-4)
        + 5 engagement context features:
              is_favorable, days_since_last_log_norm,
              has_jan_activity, is_declining, tenure_norm
    Action space: Discrete(2)  -- 0=reject, 1=accept
    """

    metadata = {"render_modes": []}

    def __init__(self, pool_size: int = 10000) -> None:
        super().__init__()
        self._inner = KKBoxRetentionEnv(customer_mode="static", pool_size=pool_size)
        self._utility = CustomerUtility(self._inner.action_manager)
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
            self._features_row, offer,
            customer_data=self._inner.customer_data,
        )
        return obs.astype(np.float32), {"offer": offer}

    def step(self, customer_action: int):
        offer = self._current_offer
        accepted = (customer_action == 1)
        with redirect_stdout(io.StringIO()):
            _, _, terminated, truncated, info = self._inner.step_with_decision(offer, accepted)

        is_fav = self._utility._am.is_favorable(offer, self._inner.customer_data)
        if accepted:
            reward = self._utility.sample_utility(offer, self._inner.customer_data, self._rng)
        else:
            reward = -REJECT_PENALTY if is_fav else REJECT_BONUS

        return np.zeros(CUSTOMER_OBS_SIZE, dtype=np.float32), float(reward), True, False, {
            "offer": offer,
            "utility_no_noise": self._utility.utility(offer, self._inner.customer_data),
            "accepted": accepted,
        }

    def build_customer_observation(
        self,
        features_row,
        offer: int,
        customer_data=None,
    ) -> np.ndarray:
        return build_customer_observation(
            self._utility._am, features_row, offer, customer_data
        )

    def _get_features(self):
        inner = self._inner
        if inner._features_series_cache is not None:
            return inner._features_series_cache[inner._current_pool_idx]
        with redirect_stdout(io.StringIO()):
            df = inner.preprocessor_customer_data.preprocess_customer(inner.customer_data)
        return df.iloc[0]

    def _pick_offer(self):
        """Pick a balanced offer: 50% favorable, 50% unfavorable when both exist."""
        valid = self._inner.get_valid_actions()
        offers = [a for a in valid if a != DO_NOTHING_ACTION]
        if not offers:
            return None
        customer = self._inner.customer_data
        favorable   = [a for a in offers if self._utility._am.is_favorable(a, customer)]
        unfavorable = [a for a in offers if not self._utility._am.is_favorable(a, customer)]
        if favorable and unfavorable:
            pool = favorable if self._rng.random() < 0.5 else unfavorable
        else:
            pool = offers
        return int(self._rng.choice(pool))

    def close(self):
        self._inner.close()
        super().close()
