"""Train and compare DQN hyperparameter configurations for TelcoRetentionEnv."""

from __future__ import annotations

import argparse
import io
import random
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from w3_reinforcement_leraning.agent.agent_dqn import DQNAgent, DQNConfig
from w3_reinforcement_leraning.environment.env import TelcoRetentionEnv


# Proposed starter settings. Adjust these values or pass CLI overrides.
EPISODES_PER_CONFIG = 3_000
VALIDATION_INTERVAL = 250
VALIDATION_EPISODES = 100
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
LOG_INTERVAL = 100
TRAINING_CUSTOMER_FRACTION = 0.80
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "src"
    / "w3_reinforcement_leraning"
    / "checkpoints"
    / "agent_best.pt"
)

BASE_CONFIG = DQNConfig()
CONFIGURATIONS = (
    ("lr_1e-3", replace(BASE_CONFIG, learning_rate=1e-3)),
    ("lr_5e-4", replace(BASE_CONFIG, learning_rate=5e-4)),
    ("lr_1e-4", replace(BASE_CONFIG, learning_rate=1e-4)),
)


@dataclass
class ValidationResult:
    mean_return: float
    mean_episode_length: float
    action_counts: dict[int, int]


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Hide preprocessing progress prints while keeping training logs readable."""
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _create_train_validation_envs(
    seed: int,
) -> tuple[TelcoRetentionEnv, TelcoRetentionEnv]:
    """Split one sampled customer pool so validation customers are held out."""
    training_env = TelcoRetentionEnv()
    validation_env = TelcoRetentionEnv()
    customer_pool = training_env.customer_selector.df
    if len(customer_pool) < 2:
        training_env.close()
        validation_env.close()
        raise ValueError("At least two customers are required for a train/validation split.")

    shuffled_customers = customer_pool.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)
    split_index = int(len(shuffled_customers) * TRAINING_CUSTOMER_FRACTION)
    split_index = min(max(split_index, 1), len(shuffled_customers) - 1)
    training_env.customer_selector.df = shuffled_customers.iloc[:split_index].reset_index(
        drop=True
    )
    validation_env.customer_selector.df = shuffled_customers.iloc[split_index:].reset_index(
        drop=True
    )
    return training_env, validation_env


def _evaluate_agent(
    agent: DQNAgent,
    env: TelcoRetentionEnv,
    episodes: int,
    seed: int,
) -> ValidationResult:
    returns: list[float] = []
    lengths: list[int] = []
    action_counts = {action: 0 for action in range(agent.config.action_size)}

    for episode in range(episodes):
        state, _ = _quiet_env_call(env.reset, seed=seed + episode)
        episode_return = 0.0
        episode_length = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if env.customer_data_df is None:
                raise RuntimeError("Environment reset did not initialize customer data.")
            valid_actions = env.action_manager.get_valid_actions(
                env.customer_data_df
            )
            action = agent.select_action(state, valid_actions, epsilon=0.0)
            state, reward, terminated, truncated, _ = _quiet_env_call(
                env.step, action
            )
            action_counts[action] += 1
            episode_return += float(reward)
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)

    return ValidationResult(
        mean_return=float(np.mean(returns)),
        mean_episode_length=float(np.mean(lengths)),
        action_counts=action_counts,
    )


def train_dqn(
    episodes_per_config: int = EPISODES_PER_CONFIG,
    validation_interval: int = VALIDATION_INTERVAL,
    validation_episodes: int = VALIDATION_EPISODES,
    seed: int = 42,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> list[dict[str, float | str]]:
    """Train each configured agent and save the highest-validation-return model."""
    if episodes_per_config <= 0:
        raise ValueError("episodes_per_config must be greater than zero.")
    if validation_interval <= 0 or validation_episodes <= 0:
        raise ValueError("Validation interval and episode count must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training_env, validation_env = _create_train_validation_envs(seed)
    summary: list[dict[str, float | str]] = []
    best_validation_return = float("-inf")
    checkpoint = Path(checkpoint_path)

    try:
        print(
            "DQN training | "
            f"train customers={len(training_env.customer_selector.df)} | "
            f"validation customers={len(validation_env.customer_selector.df)} | "
            f"episodes/config={episodes_per_config} | seed={seed}"
        )
        print(f"Best checkpoint: {checkpoint}")

        for config_index, (config_name, config) in enumerate(CONFIGURATIONS):
            config_seed = seed + config_index * 10_000
            random.seed(config_seed)
            np.random.seed(config_seed)
            torch.manual_seed(config_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config_seed)

            agent = DQNAgent(config=config)
            epsilon = EPSILON_START
            rolling_returns: list[float] = []
            config_best_return = float("-inf")
            config_best_episode = 0
            validation_episodes_completed = 0

            print(
                f"\nConfiguration {config_name}: "
                f"{asdict(config)} | device={agent.device}"
            )

            for episode in range(1, episodes_per_config + 1):
                state, _ = _quiet_env_call(
                    training_env.reset,
                    seed=config_seed + episode,
                )
                episode_return = 0.0
                episode_length = 0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    if training_env.customer_data_df is None:
                        raise RuntimeError(
                            "Environment reset did not initialize customer data."
                        )
                    valid_actions = training_env.action_manager.get_valid_actions(
                        training_env.customer_data_df
                    )
                    action = agent.select_action(state, valid_actions, epsilon)
                    next_state, reward, terminated, truncated, _ = _quiet_env_call(
                        training_env.step,
                        action,
                    )
                    if training_env.customer_data_df is None:
                        raise RuntimeError(
                            "Environment step left customer data uninitialized."
                        )
                    next_valid_actions = (
                        training_env.action_manager.get_valid_actions(
                            training_env.customer_data_df
                        )
                        if not (terminated or truncated)
                        else ()
                    )
                    agent.remember(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated or truncated,
                        next_valid_actions,
                    )
                    loss = agent.train_step()
                    state = next_state
                    episode_return += float(reward)
                    episode_length += 1
                    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

                rolling_returns.append(episode_return)
                if episode % LOG_INTERVAL == 0 or episode == 1:
                    recent_returns = rolling_returns[-LOG_INTERVAL:]
                    loss_text = f"{loss:.5f}" if loss is not None else "warming up"
                    print(
                        f"{config_name} | episode={episode}/{episodes_per_config} | "
                        f"return={episode_return:+.4f} | "
                        f"mean_return_{len(recent_returns)}="
                        f"{np.mean(recent_returns):+.4f} | "
                        f"length={episode_length} | epsilon={epsilon:.3f} | "
                        f"loss={loss_text}"
                    )

                should_validate = (
                    episode % validation_interval == 0
                    or episode == episodes_per_config
                )
                if should_validate:
                    validation = _evaluate_agent(
                        agent,
                        validation_env,
                        episodes=validation_episodes,
                        seed=config_seed + 1_000_000 + episode * validation_episodes,
                    )
                    validation_episodes_completed += validation_episodes
                    print(
                        f"Validation | {config_name} | episode={episode} | "
                        f"mean_return={validation.mean_return:+.4f} | "
                        f"mean_length={validation.mean_episode_length:.2f} | "
                        f"actions={validation.action_counts}"
                    )

                    if validation.mean_return > config_best_return:
                        config_best_return = validation.mean_return
                        config_best_episode = episode
                    if validation.mean_return > best_validation_return:
                        best_validation_return = validation.mean_return
                        agent.save_checkpoint(
                            checkpoint,
                            configuration_name=config_name,
                            validation_mean_return=validation.mean_return,
                            validation_episode=episode,
                            training_seed=config_seed,
                        )
                        print(
                            f"Saved new best checkpoint "
                            f"(validation return={best_validation_return:+.4f})."
                        )

            summary.append(
                {
                    "configuration": config_name,
                    "best_validation_return": config_best_return,
                    "best_validation_episode": float(config_best_episode),
                    "validation_episodes": float(validation_episodes_completed),
                }
            )

        print("\nConfiguration comparison (best validation return)")
        for result in sorted(
            summary,
            key=lambda item: float(item["best_validation_return"]),
            reverse=True,
        ):
            print(
                f"  {result['configuration']}: "
                f"{float(result['best_validation_return']):+.4f} "
                f"at episode {int(float(result['best_validation_episode']))}"
            )
        print(f"Selected checkpoint: {checkpoint}")
        return summary
    finally:
        training_env.close()
        validation_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN agents and save the best validation checkpoint."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES_PER_CONFIG,
        help=f"Training episodes per configuration (default: {EPISODES_PER_CONFIG}).",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=VALIDATION_INTERVAL,
        help=f"Validate every N episodes (default: {VALIDATION_INTERVAL}).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=VALIDATION_EPISODES,
        help=f"Validation episodes per check (default: {VALIDATION_EPISODES}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Output checkpoint path (default: {CHECKPOINT_PATH}).",
    )
    arguments = parser.parse_args()
    train_dqn(
        episodes_per_config=arguments.episodes,
        validation_interval=arguments.eval_every,
        validation_episodes=arguments.eval_episodes,
        seed=arguments.seed,
        checkpoint_path=arguments.checkpoint,
    )


if __name__ == "__main__":
    main()