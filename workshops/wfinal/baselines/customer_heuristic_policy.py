"""Run a heuristic acceptance baseline for the customer agent."""

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
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from environment.customer_env import CustomerEnv
from environment.constants import ACTION_DEFINITIONS, DO_NOTHING_ACTION

_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 6


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def run_customer_heuristic_policy(episodes: int = 100, seed: int = 42) -> dict[str, Any]:
    """Accept an offer if its deterministic utility > 0, else reject."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    env = CustomerEnv()
    utility = env._utility
    episode_returns: list[float] = []
    accept_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()

    try:
        print(f"Customer heuristic policy | episodes={episodes} | seed={seed}")
        for ep in range(1, episodes + 1):
            obs, info = _quiet(env.reset, seed=seed + ep - 1)
            ep_return = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                offer = env._current_offer
                features_row = env._features_row
                turn = env._inner.turn + 1   # 1-indexed
                total_counts[offer] += 1

                u = utility.utility(offer, features_row, turn)
                action = 1 if u > 0 else 0   # accept if utility positive
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

        print("\nCustomer Heuristic summary")
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
        description="Run the heuristic acceptance baseline for the customer agent."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    run_customer_heuristic_policy(episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
