"""Evaluate a saved PPO checkpoint on the held-out test set."""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
W3_ROOT   = PROJECT_ROOT / "workshops" / "w3"
_EVAL_DIR = Path(__file__).resolve().parent
for _path in (str(W3_ROOT), str(W3_ROOT / "environment"), str(_EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _test_pools import load_or_create_test_pools, use_pool
from agent.agent_ppo import PPOAgent
from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv

DEFAULT_CHECKPOINT = W3_ROOT / "checkpoints" / "agent_ppo.pt"
_DO_NOTHING_IDX = next(
    i for i, a in enumerate(ACTION_DEFINITIONS) if a.name == "Do nothing"
)


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _render_config() -> tuple[str | None, float]:
    render = os.getenv("RENDER", "false").strip().lower() == "true"
    delay  = float(os.getenv("RENDER_DELAY", "2.0"))
    return ("human" if render else None), delay


def evaluate_ppo(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    episodes: int = 100,
    seed: int = 2025,
) -> dict[str, Any]:
    """Evaluate deterministically on the held-out test set (fingerprint-excluded from training)."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Load test pools and flatten into one evaluation pool ───────────────
    test_pools = load_or_create_test_pools()
    flat_pool  = [c for pool in test_pools for c in pool]

    render_mode, render_delay = _render_config()
    env   = KKBoxRetentionEnv(render_mode=render_mode, render_delay=render_delay)
    use_pool(env, flat_pool)
    agent, checkpoint_metadata = PPOAgent.load_checkpoint(checkpoint_path)

    episode_returns: list[float] = []
    episode_lengths: list[int]   = []
    action_counts: Counter[int]  = Counter()
    favorable_counts: dict[int, Counter] = {i: Counter() for i in range(len(ACTION_DEFINITIONS))}
    accepted_actions       = 0
    total_action_cost      = 0.0
    probability_reductions: list[float] = []
    valid_action_sizes: Counter[int]    = Counter()
    success_count          = 0
    interventions          = 0
    accepted_interventions = 0

    try:
        print(
            f"PPO evaluation | episodes={episodes} | seed={seed} | "
            f"checkpoint={Path(checkpoint_path)} | device={agent.device}"
        )
        print(f"Checkpoint metadata: {checkpoint_metadata}")
        print(
            f"Test pool: {len(flat_pool)} customers "
            f"({len(test_pools)} pools × {len(test_pools[0])} each, fingerprint-excluded)"
        )

        print("Building observation cache (first reset may take a few minutes)…")
        _quiet_env_call(env.reset, seed=seed)
        print("Cache ready. Starting evaluation…")

        for episode in range(1, episodes + 1):
            state, _ = _quiet_env_call(env.reset, seed=seed + episode - 1)
            episode_return = 0.0
            episode_length = 0
            terminated     = False
            truncated      = False
            final_info: dict[str, Any] = {}

            while not (terminated or truncated):
                if env.customer_data is None:
                    raise RuntimeError("Environment reset did not initialise customer data.")
                customer_before = env.customer_data
                valid_actions   = env.action_manager.get_valid_actions(env.customer_data)
                valid_action_sizes[len(valid_actions)] += 1
                action, _, _ = agent.select_action(state, valid_actions, deterministic=True)
                scenario = (
                    "favorable"
                    if env.action_manager.is_favorable(action, env.customer_data)
                    else "unfavorable"
                )
                favorable_counts[action][scenario] += 1
                state, reward, terminated, truncated, info = _quiet_env_call(env.step, action)
                if not env.observation_space.contains(state):
                    raise ValueError(
                        f"Episode {episode}: observation does not match the space."
                    )
                if not env.render():
                    return {}

                accepted = env.customer_data is not customer_before
                accepted_actions += int(accepted)
                action_counts[action] += 1
                total_action_cost += float(info["action_cost"])
                episode_return += float(reward)
                episode_length += 1
                final_info = info

                # ── New metrics ────────────────────────────────────────────
                if action != _DO_NOTHING_IDX:
                    interventions += 1
                    if accepted:
                        accepted_interventions += 1

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            prob_before = float(final_info["probability_before"])
            prob_after  = float(final_info["probability_after"])
            probability_reductions.append(prob_before - prob_after)
            if prob_after < prob_before:
                success_count += 1

            print(
                f"Episode {episode:03d}/{episodes} | "
                f"return={episode_return:+.4f} | length={episode_length} | "
                f"churn_prob={prob_after:.3f}"
            )

        length_distribution = Counter(episode_lengths)
        all_action_indices  = range(len(ACTION_DEFINITIONS))
        missing_actions     = [i for i in all_action_indices if action_counts[i] == 0]
        success_rate        = success_count / episodes
        acceptance_rate     = (
            accepted_interventions / interventions if interventions > 0 else 0.0
        )

        print("\nEvaluation summary")
        print(f"Mean episode return:      {np.mean(episode_returns):+.4f}")
        print(f"Std episode return:       {np.std(episode_returns):.4f}")
        print(f"Success rate:             {success_rate:.1%}  ({success_count}/{episodes} episodes improved)")
        print(
            f"Acceptance rate:          {acceptance_rate:.1%}  "
            f"({accepted_interventions}/{interventions} interventions accepted)"
        )
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
            f"Accepted actions:         {accepted_actions}/{sum(episode_lengths)}"
        )
        print(f"Total action cost:        {total_action_cost:.2f}")
        print(f"Mean churn-prob reduction:{np.mean(probability_reductions):+.4f}")
        print("Action scenario breakdown (favorable / unfavorable):")
        for index, definition in enumerate(ACTION_DEFINITIONS):
            fav   = favorable_counts[index]["favorable"]
            unfav = favorable_counts[index]["unfavorable"]
            print(f"  {index}: {definition.name}: {fav} favorable / {unfav} unfavorable")
        print("Valid-action count distribution:")
        for size in sorted(valid_action_sizes):
            print(f"  {valid_action_sizes[size]}/{episodes} episodes with {size} action(s) available")

        return {
            "mean_episode_return":            float(np.mean(episode_returns)),
            "std_episode_return":             float(np.std(episode_returns)),
            "success_rate":                   success_rate,
            "acceptance_rate":                acceptance_rate,
            "episode_lengths":                dict(length_distribution),
            "action_frequencies":             {i: action_counts[i] for i in all_action_indices},
            "all_actions_occurred":           not missing_actions,
            "accepted_actions":               accepted_actions,
            "total_action_cost":              total_action_cost,
            "mean_churn_probability_reduction": float(np.mean(probability_reductions)),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO checkpoint on the held-out test set."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes",   type=int,  default=100)
    parser.add_argument("--seed",       type=int,  default=2025)
    args = parser.parse_args()
    evaluate_ppo(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
