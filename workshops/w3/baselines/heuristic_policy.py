"""Greedy heuristic policy: pick the valid action maximising P(accept) × (1 − cost)."""

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
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv

_DO_NOTHING_IDX: int = next(
    i for i, a in enumerate(ACTION_DEFINITIONS) if a.name == "Do nothing"
)


class HeuristicPolicy:
    """Selects the valid action that maximises P(accept) × (1 − cost).

    P(accept) is  acceptance_probability_high  when is_favorable(action, customer)
    is True, and  acceptance_probability  otherwise. "Do nothing" is excluded
    unless it is the only valid action. Ties are resolved by the smallest
    action index, making the policy deterministic.
    """

    def select_action(
        self,
        valid_actions: Sequence[int],
        customer_data: dict,
        action_manager: Any,
    ) -> int:
        candidates = [a for a in valid_actions if a != _DO_NOTHING_IDX]
        if not candidates:
            return _DO_NOTHING_IDX

        best_action: int = candidates[0]
        best_score: float = -1.0

        for action in sorted(candidates):
            defn = ACTION_DEFINITIONS[action]
            if action_manager.is_favorable(action, customer_data):
                p = (
                    defn.acceptance_probability_high
                    if defn.acceptance_probability_high is not None
                    else defn.acceptance_probability
                )
            else:
                p = defn.acceptance_probability
            score = p * (1.0 - defn.cost)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def run_heuristic_policy(episodes: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    """Run the heuristic policy and report episode diagnostics."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    import random
    random.seed(seed)
    np.random.seed(seed)

    policy = HeuristicPolicy()
    env = KKBoxRetentionEnv()
    results: list[dict[str, Any]] = []
    action_frequencies: Counter[int] = Counter()
    favorable_counts: dict[int, Counter] = {i: Counter() for i in range(len(ACTION_DEFINITIONS))}
    episode_lengths: Counter[int] = Counter()
    valid_action_sizes: Counter[int] = Counter()
    accepted_actions = 0
    total_action_cost = 0.0
    probability_reductions: list[float] = []
    success_count = 0
    interventions = 0
    accepted_interventions = 0

    try:
        print(
            f"Heuristic policy | episodes={episodes} | seed={seed} | "
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
                valid_actions = env.action_manager.get_valid_actions(env.customer_data)
                valid_action_sizes[len(valid_actions)] += 1

                action = policy.select_action(
                    valid_actions, env.customer_data, env.action_manager
                )
                scenario = (
                    "favorable"
                    if env.action_manager.is_favorable(action, env.customer_data)
                    else "unfavorable"
                )
                favorable_counts[action][scenario] += 1

                observation, reward, terminated, truncated, info = _quiet_env_call(env.step, action)

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

            episode_lengths[episode_length] += 1
            prob_before = float(final_info["probability_before"])
            prob_after  = float(final_info["probability_after"])
            probability_reductions.append(prob_before - prob_after)
            if prob_after < prob_before:
                success_count += 1

            results.append({
                "episode_return":     episode_return,
                "episode_length":     episode_length,
                "probability_before": prob_before,
                "probability_after":  prob_after,
                "action_cost":        float(final_info["action_cost"]),
            })
            print(
                f"Episode {episode:03d}/{episodes} | "
                f"return={episode_return:+.4f} | length={episode_length} | "
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
        print(f"Success rate:             {success_rate:.1%}  ({success_count}/{episodes} episodes improved)")
        print(
            f"Acceptance rate:          {acceptance_rate:.1%}  "
            f"({accepted_interventions}/{interventions} interventions accepted)"
        )
        print("Episode length (steps: episodes):")
        for length, count in sorted(episode_lengths.items()):
            print(f"  {length}: {count}")
        print("Action frequency (index: name: count):")
        for index, definition in enumerate(ACTION_DEFINITIONS):
            print(f"  {index}: {definition.name}: {action_frequencies[index]}")
        print(f"All actions occurred:     {'YES' if not missing_actions else 'NO'}")
        if missing_actions:
            print(f"Actions not selected:     {missing_actions}")
        print(f"Accepted actions:         {accepted_actions}/{sum(episode_lengths.values())}")
        print(f"Total action cost:        {total_action_cost:.2f}")
        print(f"Mean churn-prob reduction:{np.mean(probability_reductions):+.4f}")
        print("Action scenario breakdown (favorable / unfavorable):")
        for index, definition in enumerate(ACTION_DEFINITIONS):
            fav   = favorable_counts[index]["favorable"]
            unfav = favorable_counts[index]["unfavorable"]
            print(f"  {index}: {definition.name}: {fav} favorable / {unfav} unfavorable")
        print("Valid-action count distribution:")
        for size in sorted(valid_action_sizes):
            print(f"  {valid_action_sizes[size]}/{episodes} episodes with {size} action(s) available")

        return results
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the greedy heuristic policy on KKBoxRetentionEnv."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    run_heuristic_policy(episodes=args.episodes, seed=args.seed)



if __name__ == "__main__":
    main()
