"""Evaluate baselines on the validation split and plot DQN/PPO learning curves."""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _p in (
    str(W3_ROOT),
    str(W3_ROOT / "environment"),
    str(W3_ROOT / "baselines"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.env import KKBoxRetentionEnv
from heuristic_policy import HeuristicPolicy

VALIDATION_SEED            = 0
EVAL_EPISODES              = 100
TRAINING_CUSTOMER_FRACTION = 0.80

CSV_DIR      = W3_ROOT / "runs" / "csv"
BASELINE_CSV = CSV_DIR / "baselines_validation.csv"
RUNS_DIR     = W3_ROOT / "runs"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _create_validation_env(seed: int) -> KKBoxRetentionEnv:
    """Reproduce the 80/20 train/val split from train_dqn and return the val env."""
    training_env = KKBoxRetentionEnv()
    validation_env = KKBoxRetentionEnv()
    pool = list(training_env.customer_selector._pool)
    if len(pool) < 2:
        training_env.close()
        validation_env.close()
        raise ValueError("Not enough customers for a train/validation split.")
    rng = random.Random(seed)
    rng.shuffle(pool)
    split = int(len(pool) * TRAINING_CUSTOMER_FRACTION)
    split = min(max(split, 1), len(pool) - 1)
    training_env.customer_selector._pool = pool[:split]
    validation_env.customer_selector._pool = pool[split:]
    training_env.close()
    return validation_env


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

ActionFn = Callable[[np.ndarray, list[int], KKBoxRetentionEnv], int]


def _evaluate(
    action_fn: ActionFn,
    env: KKBoxRetentionEnv,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Run action_fn for N episodes; return mean and std return."""
    _seed_all(seed)
    returns: list[float] = []
    for ep in range(episodes):
        state, _ = _quiet(env.reset, seed=seed + ep)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            if env.customer_data is None:
                raise RuntimeError("env.customer_data is None after reset.")
            valid_actions = env.action_manager.get_valid_actions(env.customer_data)
            action = action_fn(state, valid_actions, env)
            state, reward, terminated, truncated, _ = _quiet(env.step, action)
            episode_return += float(reward)
        returns.append(episode_return)
    arr = np.asarray(returns)
    return {"mean_return": float(arr.mean()), "std_return": float(arr.std())}


def evaluate_baselines(seed: int = 42) -> dict[str, dict[str, float]]:
    """Evaluate Random and Heuristic baselines on the validation split."""
    print(f"Building validation env (seed={seed})...")
    env = _create_validation_env(seed)
    print(f"  Validation pool size: {len(env.customer_selector._pool)}")

    heuristic = HeuristicPolicy()
    rng = np.random.default_rng(VALIDATION_SEED)

    def random_fn(obs: np.ndarray, va: list[int], e: KKBoxRetentionEnv) -> int:
        return int(rng.choice(va))

    def heuristic_fn(obs: np.ndarray, va: list[int], e: KKBoxRetentionEnv) -> int:
        return heuristic.select_action(va, e.customer_data, e.action_manager)

    results: dict[str, dict[str, float]] = {}
    for name, fn in (("Random", random_fn), ("Heuristic", heuristic_fn)):
        print(f"Evaluating {name} ({EVAL_EPISODES} episodes, seed={VALIDATION_SEED})...")
        stats = _evaluate(fn, env, EVAL_EPISODES, VALIDATION_SEED)
        results[name] = stats
        print(f"  mean_return={stats['mean_return']:+.4f}  std={stats['std_return']:.4f}")

    env.close()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "mean_return", "std_return"])
        for name, stats in results.items():
            writer.writerow([name, f"{stats['mean_return']:.6f}", f"{stats['std_return']:.6f}"])
    print(f"Saved: {BASELINE_CSV}")
    return results


# ---------------------------------------------------------------------------
# Learning curve plot
# ---------------------------------------------------------------------------

def plot_learning_curves(dqn_config: str, ppo_config: str) -> None:
    """Plot validation returns for DQN and PPO with baseline reference lines."""
    dqn_df = pd.read_csv(CSV_DIR / f"dqn_{dqn_config}.csv")
    ppo_df = pd.read_csv(CSV_DIR / f"ppo_{ppo_config}.csv")

    dqn_val = dqn_df.dropna(subset=["validation_return"])
    ppo_val = ppo_df.dropna(subset=["validation_return"])

    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found at {BASELINE_CSV}. "
            "Run without --skip-baselines first."
        )
    bl = pd.read_csv(BASELINE_CSV)
    baselines = dict(zip(bl["policy"], bl["mean_return"].astype(float)))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        dqn_val["episode"], dqn_val["validation_return"],
        color="steelblue", marker="o", markersize=4, linewidth=1.5,
        label=f"DQN — {dqn_config}",
    )
    ax.plot(
        ppo_val["episode"], ppo_val["validation_return"],
        color="darkorange", marker="s", markersize=4, linewidth=1.5,
        label=f"PPO — {ppo_config}",
    )

    palette = {"Heuristic": "seagreen", "Random": "crimson"}
    for policy, mean_ret in baselines.items():
        ax.axhline(
            mean_ret, linestyle="--", linewidth=1.2,
            color=palette.get(policy, "gray"),
            label=f"{policy} baseline ({mean_ret:+.4f})",
        )

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Validation return")
    ax.set_title("Sample efficiency: DQN and PPO vs baselines")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = RUNS_DIR / "learning_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baselines on the validation split and plot learning curves."
        )
    )
    parser.add_argument(
        "--dqn-config", default="exploratory",
        help="DQN configuration name matching runs/csv/dqn_<name>.csv (default: exploratory).",
    )
    parser.add_argument(
        "--ppo-config", default="balanced",
        help="PPO configuration name matching runs/csv/ppo_<name>.csv (default: balanced).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed used to recreate the train/validation split (default: 42).",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip baseline evaluation and use existing baselines_validation.csv.",
    )
    args = parser.parse_args()

    if args.skip_baselines:
        if not BASELINE_CSV.exists():
            raise FileNotFoundError(
                f"--skip-baselines given but {BASELINE_CSV} does not exist."
            )
        print(f"Using existing baselines: {BASELINE_CSV}")
    else:
        evaluate_baselines(seed=args.seed)

    plot_learning_curves(args.dqn_config, args.ppo_config)


if __name__ == "__main__":
    main()
