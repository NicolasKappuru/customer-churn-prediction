"""Run a random valid-action baseline against the Telco retention environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from w3_reinforcement_leraning.environment.constants import ACTION_DEFINITIONS
from w3_reinforcement_leraning.environment.env import TelcoRetentionEnv


def run_random_policy(episodes: int = 100, seed: int = 42) -> list[dict[str, float]]:
    """Run one random valid action per episode and log each transition."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    action_rng = np.random.default_rng(seed)
    env = TelcoRetentionEnv()
    results: list[dict[str, float]] = []

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

            customer_before = env.customer_data_df.copy(deep=True)
            valid_actions = env.action_manager.get_valid_actions(
                env.customer_data_df
            )
            if not valid_actions:
                raise ValueError(f"Episode {episode}: no valid actions available.")

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
            result = {
                "reward": float(reward),
                "probability_before": float(info["probability_before"]),
                "probability_after": float(info["probability_after"]),
                "action_cost": float(info["action_cost"]),
                "accepted": float(accepted),
            }
            results.append(result)

            probability_change = (
                result["probability_after"] - result["probability_before"]
            )
            print(
                f"Episode {episode:03d}/{episodes} | "
                f"valid={valid_actions} | "
                f"action={action} ({action_name}) | "
                f"accepted={accepted} | "
                f"p_churn={result['probability_before']:.3f}"
                f"->{result['probability_after']:.3f} "
                f"(delta={probability_change:+.3f}) | "
                f"cost={result['action_cost']:.2f} | "
                f"reward={result['reward']:+.4f} | "
                f"terminated={terminated} | truncated={truncated}"
            )

        rewards = np.asarray([result["reward"] for result in results])
        probabilities_before = np.asarray(
            [result["probability_before"] for result in results]
        )
        probabilities_after = np.asarray(
            [result["probability_after"] for result in results]
        )
        costs = np.asarray([result["action_cost"] for result in results])
        accepted_count = int(sum(result["accepted"] for result in results))

        print("\nSummary")
        print(f"Episodes:                 {len(results)}")
        print(f"Accepted actions:         {accepted_count}/{len(results)}")
        print(f"Mean reward:              {rewards.mean():.4f}")
        print(f"Total reward:             {rewards.sum():.4f}")
        print(f"Mean churn probability:   {probabilities_before.mean():.4f}")
        print(f"Mean final probability:   {probabilities_after.mean():.4f}")
        print(f"Mean probability change:  {(probabilities_after - probabilities_before).mean():+.4f}")
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