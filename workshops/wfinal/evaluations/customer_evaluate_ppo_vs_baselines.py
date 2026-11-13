"""Compare CustomerPPO (sampled) against Random, Price and Activity baselines."""

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
W3_ROOT   = PROJECT_ROOT / "workshops" / "wfinal"
_EVAL_DIR = Path(__file__).resolve().parent
for _path in (str(W3_ROOT), str(W3_ROOT / "environment"), str(W3_ROOT / "baselines"), str(_EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.customer_agent_ppo import CustomerPPOAgent
from environment.customer_env import CustomerEnv
from environment.constants import ACTION_DEFINITIONS
from customer_price_heuristic import CustomerPriceHeuristic
from customer_activity_heuristic import CustomerActivityHeuristic

DEFAULT_CHECKPOINT = W3_ROOT / "checkpoints" / "customer_agent_ppo.pt"
_N_OFFERS = len(ACTION_DEFINITIONS) - 1


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _run_policy(
    action_fn: Callable,
    env: CustomerEnv,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run action_fn for episodes with fixed seeds. Returns aggregated stats."""
    episode_returns: list[float] = []
    accept_counts: Counter[int] = Counter()
    total_counts:  Counter[int] = Counter()

    for ep in range(episodes):
        obs, _ = _quiet(env.reset, seed=seed + ep)
        ep_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            offer = env._current_offer
            total_counts[offer] += 1
            action, _ = action_fn(obs, offer, env)
            if action == 1:
                accept_counts[offer] += 1
            obs, reward, terminated, truncated, _ = _quiet(env.step, action)
            ep_return += float(reward)
        episode_returns.append(ep_return)

    total_accepts = sum(accept_counts.values())
    total_steps   = sum(total_counts.values())
    return {
        "mean_return":     float(np.mean(episode_returns)),
        "std_return":      float(np.std(episode_returns)),
        "acceptance_rate": total_accepts / max(total_steps, 1),
        "acceptance_per_action": {
            i: accept_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
            for i in range(_N_OFFERS)
        },
    }


def compare_customer_policies(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    episodes: int = 100,
    seed: int = 2025,
) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent, meta = CustomerPPOAgent.load_checkpoint(checkpoint_path)
    print(f"Customer policy comparison | episodes={episodes} | seed={seed}")
    print(f"PPO checkpoint: {meta}")

    env = CustomerEnv()
    rng = np.random.default_rng(seed)
    price_policy    = CustomerPriceHeuristic()
    activity_policy = CustomerActivityHeuristic()

    policies: list[tuple[str, Callable]] = [
        ("Random",             lambda obs, offer, e: (int(rng.integers(0, 2)), 0.5)),
        ("Price heuristic",    price_policy.decide),
        ("Activity heuristic", activity_policy.decide),
        ("PPO (sampled)",      lambda obs, offer, e: (
            agent.select_action(obs, deterministic=False)[0],
            agent.accept_probability(obs),
        )),
    ]

    all_stats: dict[str, dict[str, Any]] = {}

    try:
        for name, fn in policies:
            print(f"  Running {name}...", end=" ", flush=True)
            stats = _run_policy(fn, env, episodes, seed)
            all_stats[name] = stats
            print(f"mean_return={stats['mean_return']:+.4f}  acceptance={stats['acceptance_rate']:.1%}")
    finally:
        env.close()

    # ── Comparison table ──────────────────────────────────────────────────────
    names   = list(all_stats.keys())
    col_w   = 20
    label_w = 28
    width   = label_w + col_w * len(names)

    print(f"\n{'=' * width}")
    print("CUSTOMER POLICY COMPARISON")
    print(f"{'=' * width}")
    header = " " * (label_w + 2)
    for n in names:
        header += f"{n:>{col_w}}"
    print(header)
    print(f"  {'-' * (label_w + col_w * len(names))}")

    def _row(label: str, values: list[float], fmt: str, higher_is_better: bool = True) -> None:
        best = max(values) if higher_is_better else min(values)
        line = f"  {label:<{label_w}}"
        for v in values:
            cell = fmt.format(v)
            mark = "<" if v == best else " "
            line += f"{cell:>{col_w - 1}}{mark}"
        print(line)

    _row("Mean customer reward",    [all_stats[n]["mean_return"]     for n in names], "{:+.4f}")
    _row("Std customer reward",     [all_stats[n]["std_return"]      for n in names], "{:.4f}", False)
    _row("Overall acceptance rate", [all_stats[n]["acceptance_rate"] for n in names], "{:.1%}")

    print(f"\n  {'Offer':<28}" + "".join(f"{n:>{col_w}}" for n in names))
    print(f"  {'-' * (26 + col_w * len(names))}")
    for i in range(_N_OFFERS):
        line = f"  {i}: {ACTION_DEFINITIONS[i].name:<24}"
        for n in names:
            rate = all_stats[n]["acceptance_per_action"][i]
            line += f"{rate:>{col_w}.1%}"
        print(line)

    print(f"{'=' * width}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CustomerPPO vs baselines under identical episodes."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes",   type=int,  default=100)
    parser.add_argument("--seed",       type=int,  default=2025)
    args = parser.parse_args()
    compare_customer_policies(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
