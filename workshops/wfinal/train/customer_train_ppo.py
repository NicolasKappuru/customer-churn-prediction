"""Train the CustomerPPOAgent on CustomerEnv."""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.customer_agent_ppo import CustomerPPOAgent, CustomerPPOConfig, CustomerTransition
from environment.customer_env import CustomerEnv
from environment.constants import ACTION_DEFINITIONS, DO_NOTHING_ACTION


EPISODES_PER_CONFIG = 20_000
VALIDATION_INTERVAL = 1_000
VALIDATION_EPISODES = 200
ROLLOUT_EPISODES = 8
LOG_INTERVAL = 500
VALIDATION_SEED = 0
CHECKPOINT_PATH = W3_ROOT / "checkpoints" / "customer_agent_ppo.pt"
RUNS_DIR        = W3_ROOT / "runs"
CSV_DIR         = RUNS_DIR / "csv"

_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 6 non-do-nothing actions


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate_agent(
    agent: CustomerPPOAgent,
    env: CustomerEnv,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    _seed_all(seed)
    returns: list[float] = []
    accept_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()

    for ep in range(episodes):
        obs, info = _quiet(env.reset, seed=seed + ep)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _, _ = agent.select_action(obs, deterministic=True)
            offer = env._current_offer
            total_counts[offer] += 1
            if action == 1:
                accept_counts[offer] += 1
            obs, reward, terminated, truncated, _ = _quiet(env.step, action)
            episode_return += float(reward)
        returns.append(episode_return)

    acceptance_per_action = {
        i: (accept_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0)
        for i in range(_N_OFFERS)
    }
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "acceptance_per_action": acceptance_per_action,
        "total_acceptance": sum(accept_counts.values()) / max(sum(total_counts.values()), 1),
    }


def train_customer_ppo(
    episodes: int = EPISODES_PER_CONFIG,
    validation_interval: int = VALIDATION_INTERVAL,
    validation_episodes: int = VALIDATION_EPISODES,
    rollout_episodes: int = ROLLOUT_EPISODES,
    seed: int = 42,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> None:
    """Train CustomerPPOAgent and save the best checkpoint by validation return."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    _seed_all(seed)
    agent = CustomerPPOAgent()
    env = CustomerEnv()
    val_env = CustomerEnv()
    checkpoint = Path(checkpoint_path)

    rollout: list[CustomerTransition] = []
    episode_returns: list[float] = []
    best_val_return = float("-inf")
    losses: dict[str, float] | None = None

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / "customer_ppo.csv"

    print(
        f"Customer PPO training | episodes={episodes} | rollout={rollout_episodes} | "
        f"seed={seed} | device={agent.device}"
    )
    print(f"Checkpoint: {checkpoint}")

    try:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["episode", "return", "mean_return_100",
                             "policy_loss", "value_loss", "entropy", "validation_return"])

            for episode in range(1, episodes + 1):
                obs, _ = _quiet(env.reset, seed=seed + episode)
                ep_return = 0.0
                terminated = truncated = False

                while not (terminated or truncated):
                    action, log_prob, value = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, _ = _quiet(env.step, action)
                    rollout.append(CustomerTransition(
                        state=obs, action=action, reward=reward, next_state=next_obs,
                        terminated=terminated or truncated,
                        old_log_probability=log_prob, value=value,
                    ))
                    obs = next_obs
                    ep_return += float(reward)

                episode_returns.append(ep_return)
                mean_100 = float(np.mean(episode_returns[-100:]))

                if len(rollout) >= rollout_episodes:
                    losses = agent.update(rollout)
                    rollout.clear()

                if episode % LOG_INTERVAL == 0 or episode == 1:
                    loss_text = (
                        f"policy={losses['policy_loss']:.5f}, "
                        f"value={losses['value_loss']:.5f}, "
                        f"entropy={losses['entropy']:.5f}"
                        if losses is not None else "collecting rollout"
                    )
                    print(
                        f"episode={episode}/{episodes} | "
                        f"return={ep_return:+.4f} | "
                        f"mean_100={mean_100:+.4f} | {loss_text}"
                    )

                val_return_csv = ""
                if episode % validation_interval == 0 or episode == episodes:
                    if rollout:
                        losses = agent.update(rollout)
                        rollout.clear()
                    val_stats = _evaluate_agent(agent, val_env, validation_episodes, VALIDATION_SEED)
                    val_return = val_stats["mean_return"]
                    acc = val_stats["total_acceptance"]
                    acc_per = val_stats["acceptance_per_action"]
                    print(
                        f"Validation | episode={episode} | "
                        f"mean_return={val_return:+.4f} | "
                        f"acceptance={acc:.1%} | "
                        f"per_action={[f'{acc_per[i]:.2f}' for i in range(_N_OFFERS)]}"
                    )
                    if val_return > best_val_return:
                        best_val_return = val_return
                        agent.save_checkpoint(
                            checkpoint,
                            validation_mean_return=val_return,
                            validation_episode=episode,
                            training_seed=seed,
                        )
                        print(f"Saved new best checkpoint (val_return={best_val_return:+.4f}).")
                    val_return_csv = f"{val_return:.6f}"

                policy_csv  = f"{losses['policy_loss']:.6f}"  if losses else ""
                value_csv   = f"{losses['value_loss']:.6f}"   if losses else ""
                entropy_csv = f"{losses['entropy']:.6f}"      if losses else ""
                writer.writerow([episode, f"{ep_return:.6f}", f"{mean_100:.6f}",
                                 policy_csv, value_csv, entropy_csv, val_return_csv])
                csv_file.flush()

        print(f"\nTraining complete. Best validation return: {best_val_return:+.4f}")
        print(f"Checkpoint saved: {checkpoint}")
        _plot_learning_curve(csv_path)
    finally:
        env.close()
        val_env.close()


def _plot_learning_curve(csv_path: Path) -> None:
    import pandas as pd
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["episode"], df["mean_return_100"], color="steelblue",
            linewidth=1.5, label="Mean return (100 ep)")
    val = df.dropna(subset=["validation_return"])
    if not val.empty:
        ax.scatter(val["episode"], val["validation_return"], color="orange",
                   zorder=5, s=30, label="Validation return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Customer utility return")
    ax.set_title("Customer PPO — Learning curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RUNS_DIR / "customer_ppo_learning_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Learning curve saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CustomerPPOAgent on CustomerEnv."
    )
    parser.add_argument("--episodes",           type=int,  default=EPISODES_PER_CONFIG)
    parser.add_argument("--eval-every",         type=int,  default=VALIDATION_INTERVAL)
    parser.add_argument("--eval-episodes",      type=int,  default=VALIDATION_EPISODES)
    parser.add_argument("--rollout-episodes",   type=int,  default=ROLLOUT_EPISODES)
    parser.add_argument("--seed",               type=int,  default=42)
    parser.add_argument("--checkpoint",         type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()
    train_customer_ppo(
        episodes=args.episodes,
        validation_interval=args.eval_every,
        validation_episodes=args.eval_episodes,
        rollout_episodes=args.rollout_episodes,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
