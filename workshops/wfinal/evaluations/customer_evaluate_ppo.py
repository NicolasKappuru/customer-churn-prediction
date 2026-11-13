"""Evaluate CustomerPPOAgent (sampled mode)."""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_CHECKPOINT = W3_ROOT / "checkpoints" / "customer_agent_ppo.pt"
_RESULTS_DIR = _EVAL_DIR / "results"
_N_OFFERS = len(ACTION_DEFINITIONS) - 1   # 6


def _quiet(fn: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _oracle_action(env: CustomerEnv, offer: int) -> int:
    u = env._utility.utility(offer, env._inner.customer_data)
    return 1 if u > 0 else 0


def _eval_ppo(
    action_fn: Callable,
    env: CustomerEnv,
    episodes: int,
    seed: int,
    verbose: bool = False,
    csv_writer=None,
) -> dict[str, Any]:
    episode_returns: list[float] = []
    accept_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()
    accept_probs: dict[int, list[float]] = {i: [] for i in range(_N_OFFERS)}
    oracle_agree_total = 0
    oracle_total = 0
    extreme_count = 0
    extreme_total = 0
    per_ep_pi_std: list[float] = []
    fav_agree:   dict[int, list[int]] = {i: [] for i in range(_N_OFFERS)}
    unfav_agree: dict[int, list[int]] = {i: [] for i in range(_N_OFFERS)}

    for ep in range(episodes):
        obs, _ = _quiet(env.reset, seed=seed + ep)
        ep_return = 0.0
        step = 0
        terminated = truncated = False
        episode_pi: list[float] = []
        if verbose:
            print(f"  Episode {ep + 1:03d}/{episodes}")
        while not (terminated or truncated):
            offer = env._current_offer
            total_counts[offer] += 1
            step += 1
            action, p_accept = action_fn(obs, offer, env)
            accept_probs[offer].append(p_accept)
            episode_pi.append(p_accept)

            oracle = _oracle_action(env, offer)
            agreed = int(action == oracle)
            oracle_agree_total += agreed
            oracle_total += 1
            is_fav = env._inner.action_manager.is_favorable(offer, env._inner.customer_data)
            (fav_agree if is_fav else unfav_agree)[offer].append(agreed)

            if p_accept < 0.10 or p_accept > 0.90:
                extreme_count += 1
            extreme_total += 1

            if action == 1:
                accept_counts[offer] += 1
            obs, reward, terminated, truncated, _ = _quiet(env.step, action)
            ep_return += float(reward)

            if verbose:
                decision = "ACCEPT" if action == 1 else "REJECT"
                fav_tag = "FAV" if is_fav else "unfav"
                print(
                    f"    Step {step} | {ACTION_DEFINITIONS[offer].name:<30} "
                    f"p_accept={p_accept:.3f}  [{fav_tag}]  -->  {decision}"
                )

            if csv_writer is not None:
                csv_writer.writerow([
                    ep + 1, step, ACTION_DEFINITIONS[offer].name,
                    f"{p_accept:.6f}", action, f"{float(reward):.6f}",
                ])

        episode_returns.append(ep_return)
        if len(episode_pi) > 1:
            per_ep_pi_std.append(float(np.std(episode_pi)))

    total_accepts = sum(accept_counts.values())
    total_steps   = sum(total_counts.values())
    return {
        "mean_return":        float(np.mean(episode_returns)),
        "std_return":         float(np.std(episode_returns)),
        "acceptance_rate":    total_accepts / max(total_steps, 1),
        "acceptance_per_action": {
            i: accept_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
            for i in range(_N_OFFERS)
        },
        "mean_pi_per_action": {
            i: float(np.mean(accept_probs[i])) if accept_probs[i] else 0.0
            for i in range(_N_OFFERS)
        },
        "oracle_agreement":   oracle_agree_total / max(oracle_total, 1),
        "extreme_decisions":  extreme_count / max(extreme_total, 1),
        "pi_std_per_ep":      float(np.mean(per_ep_pi_std)) if per_ep_pi_std else 0.0,
        "fav_agree":          {i: float(np.mean(fav_agree[i]))   if fav_agree[i]   else float("nan") for i in range(_N_OFFERS)},
        "unfav_agree":        {i: float(np.mean(unfav_agree[i])) if unfav_agree[i] else float("nan") for i in range(_N_OFFERS)},
    }


def _print_stats(name: str, stats: dict[str, Any]) -> None:
    print(f"\n--- {name} ---")
    print(f"  Mean reward:          {stats['mean_return']:+.4f} +/- {stats['std_return']:.4f}")
    print(f"  Acceptance rate:      {stats['acceptance_rate']:.1%}")
    print(f"  Oracle agreement:     {stats['oracle_agreement']:.1%}")
    print(f"  Extreme decisions:    {stats['extreme_decisions']:.1%}  (p<0.10 or p>0.90)")
    print(f"  pi std between offers:{stats['pi_std_per_ep']:.3f}")
    print("  Per-offer  accept / mean_pi / fav_agree / unfav_agree:")
    for i in range(_N_OFFERS):
        ar  = stats["acceptance_per_action"][i]
        pi  = stats["mean_pi_per_action"][i]
        fa  = stats["fav_agree"][i]
        ua  = stats["unfav_agree"][i]
        fa_s = f"{fa:.1%}" if fa == fa else "n/a"
        ua_s = f"{ua:.1%}" if ua == ua else "n/a"
        print(f"    {i}: {ACTION_DEFINITIONS[i].name:<30} {ar:.1%}  pi={pi:.3f}  fav={fa_s}  unfav={ua_s}")


def evaluate_customer_ppo(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    episodes: int = 200,
    seed: int = 2025,
) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent, meta = CustomerPPOAgent.load_checkpoint(checkpoint_path)
    print(f"Customer PPO evaluation | episodes={episodes} | seed={seed}")
    print(f"Checkpoint: {meta}")

    env = CustomerEnv()

    def _sampled_fn(obs, offer, env_):
        action, _, _ = agent.select_action(obs, deterministic=False)
        return action, agent.accept_probability(obs)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RESULTS_DIR / "customer_ppo_steps.csv"

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["episode", "step", "offer_name", "p_accept", "action", "reward"])

            sampled_stats = _eval_ppo(
                _sampled_fn, env, episodes, seed,
                verbose=(episodes <= 20),
                csv_writer=writer,
            )

        _print_stats("PPO (sampled)", sampled_stats)
        print(f"\nPer-step CSV saved: {csv_path}")

    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CustomerPPOAgent (sampled)."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes",   type=int,  default=200)
    parser.add_argument("--seed",       type=int,  default=2025)
    args = parser.parse_args()
    evaluate_customer_ppo(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
