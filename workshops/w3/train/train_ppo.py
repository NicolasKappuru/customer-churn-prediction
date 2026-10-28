"""Train and compare PPO configurations for KKBoxRetentionEnv."""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from collections import Counter
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
RUNS_DIR        = W3_ROOT / "runs"
CSV_DIR         = RUNS_DIR / "csv"

BASE_CONFIG = PPOConfig()
CONFIGURATIONS = (
    # lr=1e-3 | weight_decay=0.0  → fast policy updates, no regularization
    ("exploratory",  replace(BASE_CONFIG, learning_rate=1e-3,  weight_decay=0.0)),
    # lr=3e-4 | weight_decay=1e-5 → default speed + light regularization
    ("balanced",     replace(BASE_CONFIG, learning_rate=3e-4,  weight_decay=1e-5)),
    # lr=1e-4 | weight_decay=1e-4 → conservative updates + strong regularization
    ("conservative", replace(BASE_CONFIG, learning_rate=1e-4,  weight_decay=1e-4)),
)


@dataclass
class ValidationResult:
    mean_return: float
    std_return: float
    mean_episode_length: float
    action_counts: dict[int, int]


@dataclass
class _EpisodeResult:
    transitions: list[PPOTransition]
    episode_return: float
    episode_length: int
    accepted_count: int
    action_cost: float
    action_frequency_update: Counter


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Suppress preprocessing progress output during agent training."""
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


def _run_ppo_episode(
    agent: PPOAgent,
    env: KKBoxRetentionEnv,
    config_seed: int,
    episode: int,
) -> _EpisodeResult:
    state, _ = _quiet_env_call(env.reset, seed=config_seed + episode)
    transitions: list[PPOTransition] = []
    episode_return = 0.0
    episode_length = 0
    accepted_count = 0
    action_cost = 0.0
    action_frequency_update: Counter[int] = Counter()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        if env.customer_data is None:
            raise RuntimeError("Environment did not initialize a customer.")
        customer_before = env.customer_data
        valid_actions = env.action_manager.get_valid_actions(env.customer_data)
        action, old_log_probability, value = agent.select_action(state, valid_actions)
        next_state, reward, terminated, truncated, info = _quiet_env_call(
            env.step, action,
        )
        if not env.observation_space.contains(next_state):
            raise ValueError(
                f"Episode {episode}: step observation does not match "
                "the environment observation space."
            )
        transitions.append(PPOTransition(
            state=state, action=action, reward=reward, next_state=next_state,
            terminated=terminated or truncated, valid_actions=valid_actions,
            old_log_probability=old_log_probability, value=value,
        ))
        action_frequency_update[action] += 1
        accepted_count += int(env.customer_data is not customer_before)
        action_cost += float(info["action_cost"])
        state = next_state
        episode_return += float(reward)
        episode_length += 1

    return _EpisodeResult(
        transitions=transitions,
        episode_return=episode_return,
        episode_length=episode_length,
        accepted_count=accepted_count,
        action_cost=action_cost,
        action_frequency_update=action_frequency_update,
    )


def _log_ppo_progress(
    config_name: str,
    episode: int,
    episodes_per_config: int,
    episode_return: float,
    recent_returns: list[float],
    episode_length: int,
    accepted_actions: int,
    total_action_cost: float,
    losses: dict[str, float] | None,
) -> None:
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
        f"mean_return_{len(recent_returns)}={np.mean(recent_returns):+.4f} | "
        f"length={episode_length} | "
        f"accepted={accepted_actions} | "
        f"cost={total_action_cost:.2f} | {loss_text}"
    )


def _run_ppo_validation(
    agent: PPOAgent,
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
        agent, validation_env, validation_episodes,
        seed=config_seed + 1_000_000 + episode * validation_episodes,
    )
    print(
        f"Validation | {config_name} | episode={episode} | "
        f"mean_return={validation.mean_return:+.4f} | "
        f"std_return={validation.std_return:.4f} | "
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


def _train_ppo_config(
    config_name: str,
    config: PPOConfig,
    training_env: KKBoxRetentionEnv,
    validation_env: KKBoxRetentionEnv,
    episodes_per_config: int,
    validation_interval: int,
    validation_episodes: int,
    rollout_episodes: int,
    config_seed: int,
    best_validation_return: float,
    checkpoint: Path,
) -> tuple[dict[str, float | str], float]:
    """Train one configuration. Returns (result_dict, updated_best_validation_return)."""
    agent = PPOAgent(config=config)
    rollout: list[PPOTransition] = []
    config_best_return = float("-inf")
    config_best_episode = 0
    validation_checks = 0
    episode_returns: list[float] = []
    accepted_actions = 0
    total_action_cost = 0.0
    losses: dict[str, float] | None = None

    print(f"\nConfiguration {config_name} | device={agent.device}")
    print(f"  lr={config.learning_rate} | weight_decay={config.weight_decay}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"ppo_{config_name}.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "return", "mean_return_100",
                         "policy_loss", "value_loss", "entropy", "validation_return"])

        for episode in range(1, episodes_per_config + 1):
            result = _run_ppo_episode(agent, training_env, config_seed, episode)
            rollout.extend(result.transitions)
            accepted_actions += result.accepted_count
            total_action_cost += result.action_cost
            episode_returns.append(result.episode_return)
            mean_100 = float(np.mean(episode_returns[-100:]))

            if len(rollout) >= rollout_episodes:
                losses = agent.update(rollout)
                rollout.clear()

            if episode % LOG_INTERVAL == 0 or episode == 1:
                _log_ppo_progress(
                    config_name, episode, episodes_per_config,
                    result.episode_return, episode_returns[-LOG_INTERVAL:],
                    result.episode_length, accepted_actions, total_action_cost, losses,
                )

            val_return_csv = ""
            should_validate = (
                episode % validation_interval == 0 or episode == episodes_per_config
            )
            if should_validate:
                if rollout:
                    losses = agent.update(rollout)
                    rollout.clear()
                val_return, best_validation_return = _run_ppo_validation(
                    agent, validation_env, config_name, episode, config_seed,
                    validation_episodes, best_validation_return, checkpoint,
                )
                validation_checks += 1
                val_return_csv = f"{val_return:.6f}"
                if val_return > config_best_return:
                    config_best_return = val_return
                    config_best_episode = episode

            policy_csv  = f"{losses['policy_loss']:.6f}"  if losses else ""
            value_csv   = f"{losses['value_loss']:.6f}"   if losses else ""
            entropy_csv = f"{losses['entropy']:.6f}"      if losses else ""
            writer.writerow([episode, f"{result.episode_return:.6f}", f"{mean_100:.6f}",
                             policy_csv, value_csv, entropy_csv, val_return_csv])
            csv_file.flush()

    return (
        {
            "configuration": config_name,
            "best_validation_return": config_best_return,
            "best_validation_episode": float(config_best_episode),
            "validation_checks": float(validation_checks),
        },
        best_validation_return,
    )


def _plot_best_config(best_config_name: str) -> None:
    csv_path = CSV_DIR / f"ppo_{best_config_name}.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)

    # ── Image 1: learning curve ───────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["episode"], df["mean_return_100"], color="steelblue",
             linewidth=1.5, label="Mean return (100 ep)")
    val = df.dropna(subset=["validation_return"])
    ax1.scatter(val["episode"], val["validation_return"], color="orange",
                zorder=5, s=30, label="Validation return")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title(f"PPO — Learning curve: {best_config_name}")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    out1 = RUNS_DIR / "ppo_best_return.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"Learning curve saved: {out1}")

    # ── Image 2: policy losses ────────────────────────────────────────────────
    losses = df.dropna(subset=["policy_loss"])
    if losses.empty:
        return
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(losses["episode"], losses["policy_loss"], color="tomato",
             linewidth=1.2, label="Policy loss")
    ax2.plot(losses["episode"], losses["value_loss"], color="mediumseagreen",
             linewidth=1.2, label="Value loss")
    ax2.plot(losses["episode"], losses["entropy"], color="mediumpurple",
             linewidth=1.2, label="Entropy", linestyle="--")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss / Entropy")
    ax2.set_title(f"PPO — Policy losses: {best_config_name}")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = RUNS_DIR / "ppo_best_losses.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"Policy losses saved: {out2}")


def _print_summary(
    results: list[dict[str, float | str]], checkpoint: Path
) -> None:
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

    _seed_all(seed)
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

        best_config_name = CONFIGURATIONS[0][0]
        for config_index, (config_name, config) in enumerate(CONFIGURATIONS):
            config_seed = seed + config_index * 10_000
            _seed_all(config_seed)
            result, best_validation_return = _train_ppo_config(
                config_name, config, training_env, validation_env,
                episodes_per_config, validation_interval, validation_episodes,
                rollout_episodes, config_seed, best_validation_return, checkpoint,
            )
            if float(result["best_validation_return"]) >= best_validation_return:
                best_config_name = config_name
            results.append(result)

        _print_summary(results, checkpoint)
        _plot_best_config(best_config_name)
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
