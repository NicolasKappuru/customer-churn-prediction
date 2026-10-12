"""Evaluate a saved PPO checkpoint using the DQN evaluation protocol."""

from __future__ import annotations

import argparse
import io
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from w3_reinforcement_leraning.agent.agent_ppo import PPOAgent
from w3_reinforcement_leraning.environment.constants import ACTION_DEFINITIONS
from w3_reinforcement_leraning.environment.env import TelcoRetentionEnv


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "src"
    / "w3_reinforcement_leraning"
    / "checkpoints"
    / "agent_ppo_best.pt"
)


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Suppress preprocessing progress output during evaluation."""
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def evaluate_ppo(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    episodes: int = 100,
    seed: int = 2025,
) -> dict[str, Any]:
    """Evaluate the PPO policy deterministically using valid-action masks."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    agent, checkpoint_metadata = PPOAgent.load_checkpoint(checkpoint_path)
    env = TelcoRetentionEnv()
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    action_counts: Counter[int] = Counter()
    accepted_actions = 0
    total_action_cost = 0.0
    probability_reductions: list[float] = []

    try:
        print(
            f"PPO evaluation | episodes={episodes} | seed={seed} | "
            f"checkpoint={Path(checkpoint_path)} | device={agent.device}"
        )
        print(f"Checkpoint metadata: {checkpoint_metadata}")

        for episode in range(1, episodes + 1):
            state, _ = _quiet_env_call(env.reset, seed=seed + episode - 1)
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}

            while not (terminated or truncated):
                if env.customer_data_df is None:
                    raise RuntimeError("Environment did not initialize a customer.")
                customer_before = env.customer_data_df.copy(deep=True)
                valid_actions = env.action_manager.get_valid_actions(
                    env.customer_data_df
                )
                action, _, _ = agent.select_action(
                    state,
                    valid_actions,
                    deterministic=True,
                )
                state, reward, terminated, truncated, info = _quiet_env_call(
                    env.step,
                    action,
                )
                if not env.observation_space.contains(state):
                    raise ValueError(
                        f"Episode {episode}: observation does not match "
                        "the environment observation space."
                    )

                accepted_actions += int(
                    not customer_before.equals(env.customer_data_df)
                )
                action_counts[action] += 1
                total_action_cost += float(info["action_cost"])
                episode_return += float(reward)
                episode_length += 1
                final_info = info

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            probability_reductions.append(
                float(final_info["probability_before"])
                - float(final_info["probability_after"])
            )
            print(
                f"Episode {episode:03d}/{episodes} | "
                f"return={episode_return:+.4f} | length={episode_length} | "
                f"actions={episode_length} | "
                f"final_churn_probability="
                f"{float(final_info['probability_after']):.3f}"
            )

        length_distribution = Counter(episode_lengths)
        missing_actions = [
            index
            for index in range(len(ACTION_DEFINITIONS))
            if action_counts[index] == 0
        ]

        print("\nEvaluation summary")
        print(f"Mean episode return:      {np.mean(episode_returns):+.4f}")
        print(f"Std episode return:       {np.std(episode_returns):.4f}")
        print("Episode length (steps: episodes):")
        for length, count in sorted(length_distribution.items()):
            print(f"  {length}: {count}")
        print("Action frequency (index: name: count):")
        for index, definition in enumerate(ACTION_DEFINITIONS):
            print(f"  {index}: {definition.name}: {action_counts[index]}")
        print(f"All actions occurred:     {'YES' if not missing_actions else 'NO'}")
        if missing_actions:
            print(f"Actions not selected:     {missing_actions}")
        print(
            f"Accepted actions:         {accepted_actions}/"
            f"{sum(episode_lengths)}"
        )
        print(f"Total action cost:        {total_action_cost:.2f}")
        print(
            "Mean churn-probability reduction: "
            f"{np.mean(probability_reductions):+.4f}"
        )

        return {
            "mean_episode_return": float(np.mean(episode_returns)),
            "std_episode_return": float(np.std(episode_returns)),
            "episode_lengths": dict(length_distribution),
            "action_frequencies": {
                index: action_counts[index]
                for index in range(len(ACTION_DEFINITIONS))
            },
            "all_actions_occurred": not missing_actions,
            "accepted_actions": accepted_actions,
            "total_action_cost": total_action_cost,
            "mean_churn_probability_reduction": float(
                np.mean(probability_reductions)
            ),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint to evaluate (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2025)
    arguments = parser.parse_args()
    evaluate_ppo(
        checkpoint_path=arguments.checkpoint,
        episodes=arguments.episodes,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()