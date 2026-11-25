"""Retention heuristic baseline: always picks the cheapest valid action."""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv

_DO_NOTHING_IDX: int = next(
    i for i, a in enumerate(ACTION_DEFINITIONS) if a.name == "Do nothing"
)


class RetentionHeuristicPolicy:
    """Picks the valid action with the lowest cost (ties broken by smallest index).

    Falls back to Do nothing if no other valid action is available.
    """

    def select_action(
        self,
        valid_actions: Sequence[int],
        customer_data: dict,
        action_manager: Any,
        **kwargs: Any,
    ) -> int:
        candidates = [a for a in valid_actions if a != _DO_NOTHING_IDX]
        if not candidates:
            return _DO_NOTHING_IDX
        return min(candidates, key=lambda a: (ACTION_DEFINITIONS[a].cost, a))


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def run_heuristic_policy(
    episodes: int = 100,
    seed: int = 42,
    customer_mode: str = "static",
    customer_checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run the cheapest-action heuristic and report episode diagnostics."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    import random
    random.seed(seed)
    np.random.seed(seed)

    env = KKBoxRetentionEnv(customer_mode=customer_mode, customer_checkpoint=customer_checkpoint)
    policy = RetentionHeuristicPolicy()

    results: list[dict[str, Any]] = []
    action_frequencies: Counter[int] = Counter()
    accepted_actions = 0
    total_action_cost = 0.0
    probability_reductions: list[float] = []
    success_count = 0
    interventions = 0
    accepted_interventions = 0

    try:
        print(
            f"Heuristic policy [cheapest] | episodes={episodes} | seed={seed} | "
            f"customers={len(env.customer_selector._pool)}"
        )

        for episode in range(1, episodes + 1):
            observation, _ = _quiet_env_call(env.reset, seed=seed + episode - 1)
            episode_return = 0.0
            episode_length = 0
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False

            while not (terminated or truncated):
                customer_before = env.customer_data
                valid_actions = env.get_valid_actions()
                action = policy.select_action(
                    valid_actions, env.customer_data, env.action_manager
                )
                observation, reward, terminated, truncated, info = _quiet_env_call(
                    env.step, action
                )

                accepted = env.customer_data is not customer_before
                accepted_actions += int(accepted)
                episode_length += 1
                episode_return += float(reward)
                action_frequencies[action] += 1
                total_action_cost += float(info["action_cost"])
                final_info = info

                if action != _DO_NOTHING_IDX:
                    interventions += 1
                    if accepted:
                        accepted_interventions += 1

            prob_before = float(final_info["probability_before"])
            prob_after  = float(final_info["probability_after"])
            probability_reductions.append(prob_before - prob_after)
            if prob_after < prob_before:
                success_count += 1

            results.append({
                "episode_return":     episode_return,
                "probability_before": prob_before,
                "probability_after":  prob_after,
                "action_cost":        float(final_info["action_cost"]),
            })
            print(
                f"Episode {episode:03d}/{episodes} | "
                f"return={episode_return:+.4f} | "
                f"churn_prob={prob_after:.3f}"
            )

        all_action_indices = range(len(ACTION_DEFINITIONS))
        missing_actions = [i for i in all_action_indices if action_frequencies[i] == 0]
        success_rate    = success_count / episodes
        acceptance_rate = accepted_interventions / interventions if interventions > 0 else 0.0
        episode_returns = [r["episode_return"] for r in results]

        print("\nEvaluation summary")
        print(f"Mean episode return:      {np.mean(episode_returns):+.4f}")
        print(f"Std episode return:       {np.std(episode_returns):.4f}")
        print(f"Success rate:             {success_rate:.1%}  ({success_count}/{episodes} improved)")
        print(
            f"Acceptance rate:          {acceptance_rate:.1%}  "
            f"({accepted_interventions}/{interventions} interventions accepted)"
        )
        print("Action frequency (index: name: count):")
        for index, definition in enumerate(ACTION_DEFINITIONS):
            print(f"  {index}: {definition.name}: {action_frequencies[index]}")
        if missing_actions:
            print(f"Actions not selected:     {missing_actions}")
        print(f"Accepted actions:         {accepted_actions}/{sum(action_frequencies.values())}")
        print(f"Total action cost:        {total_action_cost:.2f}")
        print(f"Mean churn-prob reduction:{np.mean(probability_reductions):+.4f}")

        return results
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cheapest-action heuristic on KKBoxRetentionEnv."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--customer", choices=["static", "agent"], default="static",
                        help="Acceptance mode: 'static' uses fixed probabilities, 'agent' uses trained CustomerPPOAgent.")
    parser.add_argument("--customer-checkpoint", type=str,
                        default=str(Path(__file__).resolve().parents[2] / "checkpoints" / "customer_agent_ppo.pt"))
    args = parser.parse_args()
    run_heuristic_policy(episodes=args.episodes, seed=args.seed,
                         customer_mode=args.customer,
                         customer_checkpoint=args.customer_checkpoint)


if __name__ == "__main__":
    main()
