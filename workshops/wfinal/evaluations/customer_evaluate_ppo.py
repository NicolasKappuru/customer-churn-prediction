"""Evaluate CustomerPPOAgent vs random and heuristic baselines."""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT   = PROJECT_ROOT / "workshops" / "w3"
_EVAL_DIR = Path(__file__).resolve().parent
for _path in (str(W3_ROOT), str(W3_ROOT / "environment"), str(W3_ROOT / "baselines"), str(_EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.customer_agent_ppo import CustomerPPOAgent
from environment.customer_env import CustomerEnv
from environment.constants import ACTION_DEFINITIONS

DEFAULT_CHECKPOINT = W3_ROOT / "checkpoints" / "customer_agent_ppo.pt"
_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 6


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _eval_policy(
    action_fn: Callable,
    env: CustomerEnv,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run action_fn on CustomerEnv for episodes, return aggregated stats."""
    episode_returns: list[float] = []
    accept_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()
    accept_probs: dict[int, list[float]] = {i: [] for i in range(_N_OFFERS)}

    for ep in range(episodes):
        obs, _ = _quiet(env.reset, seed=seed + ep)
        ep_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            offer = env._current_offer
            total_counts[offer] += 1
            action, p_accept = action_fn(obs, offer, env)
            accept_probs[offer].append(p_accept)
            if action == 1:
                accept_counts[offer] += 1
            obs, reward, terminated, truncated, _ = _quiet(env.step, action)
            ep_return += float(reward)
        episode_returns.append(ep_return)

    total_accepts = sum(accept_counts.values())
    total_steps   = sum(total_counts.values())
    return {
        "mean_return": float(np.mean(episode_returns)),
        "std_return":  float(np.std(episode_returns)),
        "acceptance_rate": total_accepts / max(total_steps, 1),
        "acceptance_per_action": {
            i: accept_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
            for i in range(_N_OFFERS)
        },
        "mean_pi_accept_per_action": {
            i: float(np.mean(accept_probs[i])) if accept_probs[i] else 0.0
            for i in range(_N_OFFERS)
        },
    }


def evaluate_customer_ppo(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    episodes: int = 100,
    seed: int = 2025,
) -> None:
    """Compare CustomerPPO, Random and Heuristic on the customer task."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent, meta = CustomerPPOAgent.load_checkpoint(checkpoint_path)
    print(f"Customer PPO checkpoint: {meta}")

    env = CustomerEnv()
    rng = np.random.default_rng(seed)

    def _ppo_fn(obs, offer, env_):
        action, _, _ = agent.select_action(obs, deterministic=True)
        p = agent.accept_probability(obs)
        return action, p

    def _rand_fn(obs, offer, env_):
        action = int(rng.integers(0, 2))
        return action, 0.5

    def _heur_fn(obs, offer, env_):
        u = env_._utility.utility(offer, env_._features_row, env_._inner.turn + 1)
        action = 1 if u > 0 else 0
        p = 1.0 if action == 1 else 0.0
        return action, p

    policies = [
        ("CustomerPPO", _ppo_fn),
        ("Random",      _rand_fn),
        ("Heuristic",   _heur_fn),
    ]

    all_stats: dict[str, dict[str, Any]] = {}

    try:
        print(f"\nEvaluating customer policies | episodes={episodes} | seed={seed}")
        for name, fn in policies:
            print(f"\n--- {name} ---")
            stats = _eval_policy(fn, env, episodes, seed)
            all_stats[name] = stats
            print(f"  Mean customer reward: {stats['mean_return']:+.4f} ± {stats['std_return']:.4f}")
            print(f"  Overall acceptance:   {stats['acceptance_rate']:.1%}")
            print("  Per-offer acceptance rate:")
            for i in range(_N_OFFERS):
                ar = stats["acceptance_per_action"][i]
                pi = stats["mean_pi_accept_per_action"][i]
                print(f"    {i}: {ACTION_DEFINITIONS[i].name}: accept={ar:.1%}  mean_π={pi:.3f}")

        # Comparison table
        col_w = 14
        label_w = 32
        agents = list(all_stats.keys())
        W = label_w + col_w * len(agents)

        print(f"\n{'='*W}")
        print("CUSTOMER AGENT COMPARISON")
        print(f"{'='*W}")
        header = " " * (label_w + 2)
        for name in agents:
            header += f"{name:>{col_w}}"
        print(header)
        print(f"  {'-'*(label_w + col_w * len(agents))}")

        def _row(label: str, key: str, fmt: str, higher_is_better: bool = True) -> None:
            vals = [all_stats[n][key] for n in agents]
            best_idx = vals.index(max(vals) if higher_is_better else min(vals))
            line = f"  {label:<{label_w}}"
            for j, name in enumerate(agents):
                cell = fmt.format(vals[j])
                mark = "◀" if j == best_idx else " "
                line += f"{cell:>{col_w - 1}}{mark}"
            print(line)

        _row("Mean customer reward",    "mean_return",     "{:+.4f}")
        _row("Std customer reward",     "std_return",      "{:.4f}", False)
        _row("Overall acceptance rate", "acceptance_rate", "{:.1%}")

        print(f"\n  {'Offer':<30}" + "".join(f"{n:>{col_w}}" for n in agents))
        print(f"  {'-'*(28 + col_w * len(agents))}")
        for i in range(_N_OFFERS):
            line = f"  {i}: {ACTION_DEFINITIONS[i].name:<26}"
            for name in agents:
                rate = all_stats[name]["acceptance_per_action"][i]
                line += f"{rate:>{col_w}.1%}"
            print(line)

        print(f"{'='*W}")

    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CustomerPPOAgent vs baselines."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes",   type=int,  default=100)
    parser.add_argument("--seed",       type=int,  default=2025)
    args = parser.parse_args()
    evaluate_customer_ppo(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
