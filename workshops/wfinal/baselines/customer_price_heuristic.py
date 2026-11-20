"""Run a price-based heuristic acceptance baseline for the customer agent."""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from environment.customer_env import CustomerEnv
from environment.constants import ACTION_DEFINITIONS

_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 6

# Actions 0,1,2,3,5 → accept (1); Action 4 (recover cancellation) → reject (0)
_PRICE_ACCEPT = {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1}


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class CustomerPriceHeuristic:
    """Accept based solely on offer type: actions 0,1,2,3,5 → accept; action 4 → reject."""

    def decide(self, obs, offer: int, env: CustomerEnv) -> tuple[int, float]:
        action = _PRICE_ACCEPT.get(offer, 1)
        p = float(action)
        return action, p


def run_customer_price_heuristic(episodes: int = 100, seed: int = 42) -> dict[str, Any]:
    """Accept based on offer type (action 4 rejected, all others accepted)."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    env = CustomerEnv()
    policy = CustomerPriceHeuristic()
    episode_returns: list[float] = []
    accept_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()

    try:
        print(f"Customer price heuristic | episodes={episodes} | seed={seed}")
        for ep in range(1, episodes + 1):
            obs, info = _quiet(env.reset, seed=seed + ep - 1)
            ep_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                offer = env._current_offer
                total_counts[offer] += 1

                action, _ = policy.decide(obs, offer, env)
                if action == 1:
                    accept_counts[offer] += 1

                obs, reward, terminated, truncated, _ = _quiet(env.step, action)
                ep_return += float(reward)
            episode_returns.append(ep_return)
            print(
                f"Episode {ep:03d}/{episodes} | return={ep_return:+.4f}"
            )

        mean_return = float(np.mean(episode_returns))
        std_return  = float(np.std(episode_returns))
        total_accepts = sum(accept_counts.values())
        total_steps   = sum(total_counts.values())
        acceptance_rate = total_accepts / max(total_steps, 1)

        print("\nCustomer Price Heuristic summary")
        print(f"Mean customer reward:   {mean_return:+.4f}")
        print(f"Std customer reward:    {std_return:.4f}")
        print(f"Overall acceptance rate:{acceptance_rate:.1%}  ({total_accepts}/{total_steps})")
        print("Acceptance rate per offer:")
        for i in range(_N_OFFERS):
            rate = accept_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
            print(f"  {i}: {ACTION_DEFINITIONS[i].name}: {rate:.1%}  ({accept_counts[i]}/{total_counts[i]})")

        return {
            "mean_return": mean_return,
            "std_return": std_return,
            "acceptance_rate": acceptance_rate,
            "acceptance_per_action": {
                i: accept_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
                for i in range(_N_OFFERS)
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the price heuristic acceptance baseline for the customer agent."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    run_customer_price_heuristic(episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
