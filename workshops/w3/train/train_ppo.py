"""Train and compare PPO configurations for KKBoxRetentionEnv."""

from __future__ import annotations

import argparse
import io
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.agent_ppo import PPOAgent, PPOConfig, PPOTransition
from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv


EPISODES_PER_CONFIG = 3_000
VALIDATION_INTERVAL = 250
VALIDATION_EPISODES = 100
ROLLOUT_EPISODES = 128
LOG_INTERVAL = 100
TRAINING_CUSTOMER_FRACTION = 0.80
CHECKPOINT_PATH = W3_ROOT / "checkpoints" / "agent_ppo.pt"

BASE_CONFIG = PPOConfig()
CONFIGURATIONS = (
    ("lr_1e-3", replace(BASE_CONFIG, learning_rate=1e-3)),
    ("lr_3e-4", replace(BASE_CONFIG, learning_rate=3e-4)),
    ("lr_1e-4", replace(BASE_CONFIG, learning_rate=1e-4)),
)


@dataclass
class ValidationResult:
    mean_return: float
    std_return: float
    mean_episode_length: float
    action_counts: dict[int, int]


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Suppress preprocessing progress output during agent training."""
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _create_train_validation_envs(
    seed: int,
) -> tuple[KKBoxRetentionEnv, KKBoxRetentionEnv]:
    """Split a single sampled customer pool into disjoint train and validation sets."""
    training_env = KKBoxRetentionEnv()
    validation_env = KKBoxRetentionEnv()
    pool = list(training_env.customer_selector._pool)
    if len(pool) < 2:
        training_env.close()
        validation_env.close()
        raise ValueError(
            "At least two customers are required for train/validation."
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    split_index = int(len(pool) * TRAINING_CUSTOMER_FRACTION)
    split_index = min(max(split_index, 1), len(pool) - 1)

    training_env.customer_selector._pool = pool[:split_index]
    validation_env.customer_selector._pool = pool[split_index:]
    return training_env, validation_env


def _evaluate_agent(
    agent: PPOAgent,
    env: KKBoxRetentionEnv,
    episodes: int,
    seed: int,
) -> ValidationResult:
    np.random.seed(seed)
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
            if env.customer_data is None:
                raise RuntimeError("Environment did not initialize a customer.")
            valid_actions = env.action_manager.get_valid_actions(env.customer_data)
            action, _, _ = agent.select_action(
                state, valid_actions, deterministic=True,
            )
            state, reward, terminated, truncated, _ = _quiet_env_call(env.step, action)
            action_counts[action] += 1
            episode_return += float(reward)
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)

    return ValidationResult(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_episode_length=float(np.mean(lengths)),
        action_counts=action_counts,
    )


def train_ppo(
    episodes_per_config: int = EPISODES_PER_CONFIG,
    validation_interval: int = VALIDATION_INTERVAL,
    validation_episodes: int = VALIDATION_EPISODES,
    rollout_episodes: int = ROLLOUT_EPISODES,
    seed: int = 42,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> list[dict[str, float | str]]:
    """Train PPO configurations and save the checkpoint with best validation return."""
    if episodes_per_config <= 0:
        raise ValueError("episodes_per_config must be greater than zero.")
    if validation_interval <= 0 or validation_episodes <= 0:
        raise ValueError("Validation interval and episode count must be positive.")
    if rollout_episodes <= 0:
        raise ValueError("rollout_episodes must be greater than zero.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training_env, validation_env = _create_train_validation_envs(seed)
    checkpoint = Path(checkpoint_path)
    results: list[dict[str, float | str]] = []
    best_validation_return = float("-inf")

    try:
        print(
            "PPO training | "
            f"train customers={len(training_env.customer_selector._pool)} | "
            f"validation customers={len(validation_env.customer_selector._pool)} | "
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

            agent = PPOAgent(config=config)
            rollout: list[PPOTransition] = []
            config_best_return = float("-inf")
            config_best_episode = 0
            validation_checks = 0
            episode_returns: list[float] = []
            action_frequencies: Counter[int] = Counter()
            accepted_actions = 0
            total_action_cost = 0.0

            print(
                f"\nConfiguration {config_name}: {asdict(config)} | "
                f"device={agent.device}"
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
                    if training_env.customer_data is None:
                        raise RuntimeError("Environment did not initialize a customer.")
                    customer_before = training_env.customer_data
                    valid_actions = training_env.action_manager.get_valid_actions(
                        training_env.customer_data
                    )
                    action, old_log_probability, value = agent.select_action(
                        state, valid_actions,
                    )
                    next_state, reward, terminated, truncated, info = _quiet_env_call(
                        training_env.step, action,
                    )
                    if not training_env.observation_space.contains(next_state):
                        raise ValueError(
                            f"Episode {episode}: step observation does not match "
                            "the environment observation space."
                        )

                    rollout.append(
                        PPOTransition(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            terminated=terminated or truncated,
                            valid_actions=valid_actions,
                            old_log_probability=old_log_probability,
                            value=value,
                        )
                    )
                    action_frequencies[action] += 1
                    accepted_actions += int(
                        training_env.customer_data is not customer_before
                    )
                    total_action_cost += float(info["action_cost"])
                    state = next_state
                    episode_return += float(reward)
                    episode_length += 1

                episode_returns.append(episode_return)
                if len(rollout) >= rollout_episodes:
                    losses = agent.update(rollout)
                    rollout.clear()
                else:
                    losses = None

                if episode % LOG_INTERVAL == 0 or episode == 1:
                    recent_returns = episode_returns[-LOG_INTERVAL:]
                    loss_text = (
                        f"policy={losses['policy_loss']:.5f}, "
                        f"value={losses['value_loss']:.5f}, "
                        f"entropy={losses['entropy']:.5f}"
                        if losses is not None
                        else "rollout collecting"
                    )
                    print(
                        f"{config_name} | episode={episode}/{episodes_per_config} | "
                        f"return={episode_return:+.4f} | "
                        f"mean_return_{len(recent_returns)}="
                        f"{np.mean(recent_returns):+.4f} | "
                        f"length={episode_length} | "
                        f"accepted={accepted_actions} | "
                        f"cost={total_action_cost:.2f} | {loss_text}"
                    )

                should_validate = (
                    episode % validation_interval == 0
                    or episode == episodes_per_config
                )
                if should_validate:
                    if rollout:
                        losses = agent.update(rollout)
                        rollout.clear()
                    validation = _evaluate_agent(
                        agent, validation_env, validation_episodes,
                        seed=(
                            config_seed
                            + 1_000_000
                            + episode * validation_episodes
                        ),
                    )
                    validation_checks += 1
                    print(
                        f"Validation | {config_name} | episode={episode} | "
                        f"mean_return={validation.mean_return:+.4f} | "
                        f"std_return={validation.std_return:.4f} | "
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
                            "Saved new best checkpoint "
                            f"(validation return={best_validation_return:+.4f})."
                        )

            results.append(
                {
                    "configuration": config_name,
                    "best_validation_return": config_best_return,
                    "best_validation_episode": float(config_best_episode),
                    "validation_checks": float(validation_checks),
                }
            )

        print("\nConfiguration comparison (best validation return)")
        for result in sorted(
            results,
            key=lambda item: float(item["best_validation_return"]),
            reverse=True,
        ):
            print(
                f"  {result['configuration']}: "
                f"{float(result['best_validation_return']):+.4f} "
                f"at episode {int(float(result['best_validation_episode']))}"
            )
        print(f"Selected checkpoint: {checkpoint}")
        return results
    finally:
        training_env.close()
        validation_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO configurations on KKBoxRetentionEnv and save the best checkpoint."
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
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=ROLLOUT_EPISODES,
        help=f"On-policy rollout size (default: {ROLLOUT_EPISODES}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Output checkpoint path (default: {CHECKPOINT_PATH}).",
    )
    arguments = parser.parse_args()
    train_ppo(
        episodes_per_config=arguments.episodes,
        validation_interval=arguments.eval_every,
        validation_episodes=arguments.eval_episodes,
        rollout_episodes=arguments.rollout_episodes,
        seed=arguments.seed,
        checkpoint_path=arguments.checkpoint,
    )


if __name__ == "__main__":
    main()
