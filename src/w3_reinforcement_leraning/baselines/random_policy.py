"""Run a random valid-action baseline against the Telco retention environment."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from w3_reinforcement_leraning.environment.constants import ACTION_DEFINITIONS
from w3_reinforcement_leraning.environment.env import TelcoRetentionEnv


def run_random_policy(episodes: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    """Run random valid actions and report episode and action diagnostics."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    action_rng = np.random.default_rng(seed)
    env = TelcoRetentionEnv()
    results: list[dict[str, Any]] = []
    action_frequencies: Counter[int] = Counter()
    episode_lengths: Counter[int] = Counter()
    accepted_actions = 0

    try:
        print(
            f"Random policy | episodes={episodes} | seed={seed} | "
            f"sampled customers={len(env.customer_selector.df)}"
        )

        for episode in range(1, episodes + 1):
            observation, _ = env.reset(seed=seed + episode - 1)
            if not env.observation_space.contains(observation):
                raise ValueError(
                    f"Episode {episode}: reset observation does not match "
                    "the environment observation space."
                )

            episode_return = 0.0
            episode_length = 0
            episode_actions: list[int] = []
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False

            while not (terminated or truncated):
                customer_before = env.customer_data_df.copy(deep=True)
                valid_actions = env.action_manager.get_valid_actions(
                    env.customer_data_df
                )
                if not valid_actions:
                    raise ValueError(
                        f"Episode {episode}: no valid actions available."
                    )

                action = int(action_rng.choice(valid_actions))
                action_name = ACTION_DEFINITIONS[action].name
                observation, reward, terminated, truncated, info = env.step(action)

                if not env.observation_space.contains(observation):
                    raise ValueError(
                        f"Episode {episode}: step observation does not match "
                        "the environment observation space."
                    )

                customer_after = env.customer_data_df
                accepted = not customer_before.equals(customer_after)
                accepted_actions += int(accepted)
                episode_length += 1
                episode_return += float(reward)
                episode_actions.append(action)
                action_frequencies[action] += 1
                final_info = info

                probability_before = float(info["probability_before"])
                probability_after = float(info["probability_after"])
                probability_change = probability_after - probability_before
                print(
                    f"Episode {episode:03d}/{episodes} | "
                    f"step={episode_length} | "
                    f"valid={valid_actions} | "
                    f"action={action} ({action_name}) | "
                    f"accepted={accepted} | "
                    f"p_churn={probability_before:.3f}"
                    f"->{probability_after:.3f} "
                    f"(delta={probability_change:+.3f}) | "
                    f"cost={float(info['action_cost']):.2f} | "
                    f"reward={float(reward):+.4f} | "
                    f"terminated={terminated} | truncated={truncated}"
                )

            episode_lengths[episode_length] += 1
            results.append(
                {
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "actions": episode_actions,
                    "probability_before": float(final_info["probability_before"]),
                    "probability_after": float(final_info["probability_after"]),
                    "action_cost": float(final_info["action_cost"]),
                }
            )

        returns = np.asarray([result["episode_return"] for result in results])
        probabilities_before = np.asarray(
            [result["probability_before"] for result in results]
        )
        probabilities_after = np.asarray(
            [result["probability_after"] for result in results]
        )
        costs = np.asarray([result["action_cost"] for result in results])

        print("\nSummary")
        print(f"Episodes:                 {len(results)}")
        print(
            f"Accepted actions:         "
            f"{accepted_actions}/{sum(episode_lengths.values())}"
        )
        print(f"Mean episode return:      {returns.mean():.4f}")
        print(f"Total episode return:     {returns.sum():.4f}")
        print("Episode length distribution (steps: episodes):")
        for length, count in sorted(episode_lengths.items()):
            print(f"  {length}: {count}")
        print("Action frequency (index: name: count):")
        for action_index, definition in enumerate(ACTION_DEFINITIONS):
            print(
                f"  {action_index}: {definition.name}: "
                f"{action_frequencies[action_index]}"
            )
        missing_actions = [
            action_index
            for action_index in range(len(ACTION_DEFINITIONS))
            if action_frequencies[action_index] == 0
        ]
        print(
            "All actions occurred: "
            f"{'YES' if not missing_actions else 'NO'}"
        )
        if missing_actions:
            print(f"Actions not selected: {missing_actions}")
        print(f"Mean churn probability:   {probabilities_before.mean():.4f}")
        print(f"Mean final probability:   {probabilities_after.mean():.4f}")
        print(
            "Mean probability change:  "
            f"{(probabilities_after - probabilities_before).mean():+.4f}"
        )
        print(f"Total action cost:        {costs.sum():.2f}")
        return results
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a random valid-action policy on TelcoRetentionEnv."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of one-step episodes to run (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random action selection (default: 42).",
    )
    arguments = parser.parse_args()
    run_random_policy(episodes=arguments.episodes, seed=arguments.seed)


if __name__ == "__main__":
    main()