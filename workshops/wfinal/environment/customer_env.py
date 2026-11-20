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
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _p in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constants import (
    ACTION_DEFINITIONS, DO_NOTHING_ACTION, N_MODEL_FEATURES,
    CUSTOMER_N_ACTIONS, CUSTOMER_OBS_SIZE,
    REJECT_PENALTY, REJECT_BONUS,
)
from env import KKBoxRetentionEnv
from customer_utility import CustomerUtility


_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 6: all except Do nothing

# Price reference for normalization (NTD)
_MAX_PRICE = 149.0

# Price mapping per action: what the customer would pay if accepting
# Actions 0, 2, 4, 5 keep current price; action 1 = 129; action 3 = 99
_OFFER_PRICE_MAP = {
    0: None,       # non-commercial: price_offer_ratio = price_current_ratio
    1: 129.0,      # loyalty discount
    2: None,       # non-commercial
    3: 99.0,       # downgrade
    4: None,       # recover cancellation: commercial but same price
    5: None,       # winback engage: non-commercial
}

# Flags per action
_IS_COMMERCIAL  = {0: False, 1: True,  2: False, 3: True,  4: True,  5: False}
_IS_ENGAGEMENT  = {0: True,  1: False, 2: True,  3: False, 4: False, 5: True}
_EXTENDS_MEMBERSHIP = {0: False, 1: True, 2: False, 3: True, 4: False, 5: False}


class CustomerEnv(gym.Env):
    """Gym environment where the agent plays the customer role.

    Each episode = one customer, one offer, one accept/reject decision.

    Observation (71 floats):
        59 preprocessed features
        + 6 one-hot offer encoding
        + 6 offer extras (price_current_ratio, price_offer_ratio, saving_ratio,
                          extends_membership, is_commercial, is_engagement)
    Action space: Discrete(2)  -- 0=reject, 1=accept
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self._inner = KKBoxRetentionEnv(customer_mode="static")
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

    @staticmethod
    def build_customer_observation(
        features_row,   # pd.Series with 59 features
        offer: int,
        customer_data=None,
    ) -> np.ndarray:
        """Build the 71-dim customer observation vector (59 features + 6 one-hot + 6 offer extras)."""
        feat = np.array(list(features_row.values), dtype=np.float32)  # 59

        # --- offer one-hot (6) ---
        one_hot = np.zeros(_N_OFFERS, dtype=np.float32)
        if 0 <= offer < _N_OFFERS:
            one_hot[offer] = 1.0

        # --- offer extras (6) ---
        # price_current_ratio: last plan_list_price / 149
        price_current = 149.0
        if customer_data is not None:
            try:
                txns = customer_data.get("transactions", [])
                if txns:
                    last_txn = txns[-1] if isinstance(txns, list) else None
                    if last_txn is not None and "plan_list_price" in last_txn:
                        price_current = float(last_txn["plan_list_price"])
            except Exception:
                pass
        price_current_ratio = price_current / _MAX_PRICE

        # price_offer_ratio
        mapped_price = _OFFER_PRICE_MAP.get(offer, None)
        if mapped_price is not None:
            price_offer_ratio = mapped_price / _MAX_PRICE
        else:
            price_offer_ratio = price_current_ratio

        saving_ratio = (price_current_ratio - price_offer_ratio)
        extends_membership = float(_EXTENDS_MEMBERSHIP.get(offer, False))
        is_commercial = float(_IS_COMMERCIAL.get(offer, False))
        is_engagement = float(_IS_ENGAGEMENT.get(offer, False))

        offer_extras = np.array(
            [price_current_ratio, price_offer_ratio, saving_ratio,
             extends_membership, is_commercial, is_engagement],
            dtype=np.float32,
        )

        return np.concatenate([feat, one_hot, offer_extras])

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
