"""Train and compare DQN hyperparameter configurations for KKBoxRetentionEnv."""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.agent_dqn import DQNAgent, DQNConfig
from environment.env import KKBoxRetentionEnv


EPISODES_PER_CONFIG = 3_000
VALIDATION_INTERVAL = 250
VALIDATION_EPISODES = 100
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
LOG_INTERVAL  = 100
VALIDATION_SEED = 0
TRAINING_CUSTOMER_FRACTION = 0.80
CHECKPOINT_PATH = W3_ROOT / "checkpoints" / "agent_dqn.pt"
RUNS_DIR        = W3_ROOT / "runs"
CSV_DIR         = RUNS_DIR / "csv"

BASE_CONFIG = DQNConfig()
CONFIGURATIONS = (
    # lr=1e-3 | epsilon_decay=0.999 | weight_decay=0.0  → learns fast, stays exploratory long
    ("exploratory",  replace(BASE_CONFIG, learning_rate=1e-3,  epsilon_decay=0.999, weight_decay=0.0)),
    # lr=5e-4 | epsilon_decay=0.995 | weight_decay=1e-5 → default speed + light regularization
    ("balanced",     replace(BASE_CONFIG, learning_rate=5e-4,  epsilon_decay=0.995, weight_decay=1e-5)),
    # lr=1e-4 | epsilon_decay=0.990 | weight_decay=1e-4 → precise updates, greedy sooner, strong reg
    ("conservative", replace(BASE_CONFIG, learning_rate=1e-4,  epsilon_decay=0.990, weight_decay=1e-4)),
)


@dataclass
class ValidationResult:
    mean_return: float
    mean_episode_length: float
    action_counts: dict[int, int]


@dataclass
class _EpisodeResult:
    episode_return: float
    episode_length: int
    epsilon: float


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Hide preprocessing progress prints while keeping training logs readable."""
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_train_validation_envs(
    seed: int,
) -> tuple[KKBoxRetentionEnv, KKBoxRetentionEnv]:
    """Split one sampled customer pool so validation customers are held out."""
    training_env = KKBoxRetentionEnv()
    validation_env = KKBoxRetentionEnv()
    pool = list(training_env.customer_selector._pool)
    if len(pool) < 2:
        training_env.close()
        validation_env.close()
        raise ValueError("At least two customers are required for a train/validation split.")

    rng = random.Random(seed)
    rng.shuffle(pool)
    split_index = int(len(pool) * TRAINING_CUSTOMER_FRACTION)
    split_index = min(max(split_index, 1), len(pool) - 1)

    training_env.customer_selector._pool = pool[:split_index]
    validation_env.customer_selector._pool = pool[split_index:]
    return training_env, validation_env


def _evaluate_agent(
    agent: DQNAgent,
    env: KKBoxRetentionEnv,
    episodes: int,
    seed: int,
) -> ValidationResult:
    random.seed(seed)
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
                raise RuntimeError("Environment reset did not initialize customer data.")
            valid_actions = env.action_manager.get_valid_actions(env.customer_data)
            action = agent.select_action(state, valid_actions, epsilon=0.0)
            state, reward, terminated, truncated, _ = _quiet_env_call(env.step, action)
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


def _run_dqn_episode(
    agent: DQNAgent,
    env: KKBoxRetentionEnv,
    config_seed: int,
    episode: int,
    epsilon: float,
) -> tuple[_EpisodeResult, float | None]:
    """Run one training episode. Returns (result, last_loss)."""
    state, _ = _quiet_env_call(env.reset, seed=config_seed + episode)
    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    last_loss: float | None = None

    while not (terminated or truncated):
        if env.customer_data is None:
            raise RuntimeError("Environment reset did not initialize customer data.")
        valid_actions = env.action_manager.get_valid_actions(env.customer_data)
        action = agent.select_action(state, valid_actions, epsilon)
        next_state, reward, terminated, truncated, _ = _quiet_env_call(
            env.step, action,
        )
        next_valid_actions = (
            env.action_manager.get_valid_actions(env.customer_data)
            if not (terminated or truncated)
            else ()
        )
        agent.remember(
            state, action, reward, next_state,
            terminated or truncated, next_valid_actions,
        )
        last_loss = agent.train_step()
        state = next_state
        episode_return += float(reward)
        episode_length += 1
        epsilon = max(EPSILON_MIN, epsilon * agent.config.epsilon_decay)

    return _EpisodeResult(episode_return, episode_length, epsilon), last_loss


def _log_dqn_progress(
    config_name: str,
    episode: int,
    episodes_per_config: int,
    episode_return: float,
    recent_returns: list[float],
    episode_length: int,
    epsilon: float,
    loss: float | None,
) -> None:
    loss_text = f"{loss:.5f}" if loss is not None else "warming up"
    print(
        f"{config_name} | episode={episode}/{episodes_per_config} | "
        f"return={episode_return:+.4f} | "
        f"mean_return_{len(recent_returns)}={np.mean(recent_returns):+.4f} | "
        f"length={episode_length} | epsilon={epsilon:.3f} | loss={loss_text}"
    )


def _run_dqn_validation(
    agent: DQNAgent,
    validation_env: KKBoxRetentionEnv,
    config_name: str,
    episode: int,
    config_seed: int,
    validation_episodes: int,
    best_validation_return: float,
    checkpoint: Path,
) -> tuple[float, float]:
    """Run validation and save checkpoint if improved. Returns (val_return, new_best)."""
    validation = _evaluate_agent(
        agent, validation_env,
        episodes=validation_episodes,
        seed=VALIDATION_SEED,
    )
    print(
        f"Validation | {config_name} | episode={episode} | "
        f"mean_return={validation.mean_return:+.4f} | "
        f"mean_length={validation.mean_episode_length:.2f} | "
        f"actions={validation.action_counts}"
    )
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
    return validation.mean_return, best_validation_return


def _train_dqn_config(
    config_name: str,
    config: DQNConfig,
    training_env: KKBoxRetentionEnv,
    validation_env: KKBoxRetentionEnv,
    episodes_per_config: int,
    validation_interval: int,
    validation_episodes: int,
    config_seed: int,
    best_validation_return: float,
    checkpoint: Path,
) -> tuple[dict[str, float | str], float]:
    """Train one configuration. Returns (result_dict, updated_best_validation_return)."""
    agent = DQNAgent(config=config)
    epsilon = EPSILON_START
    rolling_returns: list[float] = []
    config_best_return = float("-inf")
    config_best_episode = 0
    validation_episodes_completed = 0

    print(f"\nConfiguration {config_name} | device={agent.device}")
    print(f"  lr={config.learning_rate} | epsilon_decay={config.epsilon_decay} | weight_decay={config.weight_decay}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"dqn_{config_name}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "return", "mean_return_100", "epsilon", "validation_return"])

        for episode in range(1, episodes_per_config + 1):
            result, last_loss = _run_dqn_episode(
                agent, training_env, config_seed, episode, epsilon,
            )
            epsilon = result.epsilon
            rolling_returns.append(result.episode_return)
            mean_100 = float(np.mean(rolling_returns[-100:]))

            if episode % LOG_INTERVAL == 0 or episode == 1:
                _log_dqn_progress(
                    config_name, episode, episodes_per_config,
                    result.episode_return, rolling_returns[-LOG_INTERVAL:],
                    result.episode_length, epsilon, last_loss,
                )

            val_return_csv = ""
            if episode % validation_interval == 0 or episode == episodes_per_config:
                val_return, best_validation_return = _run_dqn_validation(
                    agent, validation_env, config_name, episode, config_seed,
                    validation_episodes, best_validation_return, checkpoint,
                )
                validation_episodes_completed += validation_episodes
                val_return_csv = f"{val_return:.6f}"
                if val_return > config_best_return:
                    config_best_return = val_return
                    config_best_episode = episode

            writer.writerow([episode, f"{result.episode_return:.6f}",
                             f"{mean_100:.6f}", f"{epsilon:.6f}", val_return_csv])
            csv_file.flush()

    return (
        {
            "configuration": config_name,
            "best_validation_return": config_best_return,
            "best_validation_episode": float(config_best_episode),
            "validation_episodes": float(validation_episodes_completed),
        },
        best_validation_return,
    )


def _plot_best_config(best_config_name: str) -> None:
    csv_path = CSV_DIR / f"dqn_{best_config_name}.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["mean_return_100"], color="steelblue",
            linewidth=1.5, label="Mean return (100 ep)")
    val = df.dropna(subset=["validation_return"])
    ax.scatter(val["episode"], val["validation_return"], color="orange",
               zorder=5, s=30, label="Validation return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(f"DQN — Best configuration: {best_config_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RUNS_DIR / "dqn_best.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved: {out}")


def _print_summary(
    summary: list[dict[str, float | str]], checkpoint: Path
) -> None:
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

    _seed_all(seed)
    training_env, validation_env = _create_train_validation_envs(seed)
    summary: list[dict[str, float | str]] = []
    best_validation_return = float("-inf")
    checkpoint = Path(checkpoint_path)

    try:
        _grad_steps = episodes_per_config - BASE_CONFIG.batch_size
        print(
            "DQN training | "
            f"train customers={len(training_env.customer_selector._pool)} | "
            f"validation customers={len(validation_env.customer_selector._pool)} | "
            f"episodes/config={episodes_per_config} | "
            f"grad_steps/config≈{_grad_steps} | validation_seed={VALIDATION_SEED} | seed={seed}"
        )
        print(f"Best checkpoint: {checkpoint}")

        best_config_name = CONFIGURATIONS[0][0]
        for config_index, (config_name, config) in enumerate(CONFIGURATIONS):
            config_seed = seed + config_index * 10_000
            _seed_all(config_seed)
            result, best_validation_return = _train_dqn_config(
                config_name, config, training_env, validation_env,
                episodes_per_config, validation_interval, validation_episodes,
                config_seed, best_validation_return, checkpoint,
            )
            if float(result["best_validation_return"]) >= best_validation_return:
                best_config_name = config_name
            summary.append(result)

        _print_summary(summary, checkpoint)
        _plot_best_config(best_config_name)
        return summary
    finally:
        training_env.close()
        validation_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN agents on KKBoxRetentionEnv and save the best checkpoint."
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
