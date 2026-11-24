"""Diagnose per-action reward distributions by running 100 random episodes.

Saves a CSV with one row per episode showing which action was chosen,
churn probability before/after, reward, and whether it was accepted.
Run this before training to understand which actions actually help.
"""

from __future__ import annotations

import csv
import io
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv

EPISODES = 100
SEED = 42
OUT_PATH = W3_ROOT / "runs" / "csv" / "action_diagnostic.csv"


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def run_diagnostic(episodes: int = EPISODES, seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = KKBoxRetentionEnv(customer_mode="static")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    action_stats: dict[int, dict] = {
        i: {"total": 0, "accepted": 0, "reward_sum": 0.0, "positive": 0}
        for i in range(len(ACTION_DEFINITIONS))
    }

    print(f"Action diagnostic | episodes={episodes} | seed={seed}")

    for ep in range(1, episodes + 1):
        obs, _ = _quiet(env.reset, seed=seed + ep - 1)
        valid = env.get_valid_actions()
        # Pick a random valid action (excluding Do nothing when alternatives exist)
        candidates = [a for a in valid if a != len(ACTION_DEFINITIONS) - 1] or valid
        action = int(rng.choice(candidates))

        prob_before = env.probability_churn_before
        obs, reward, _, _, info = _quiet(env.step, action)
        prob_after = info["probability_after"]
        accepted = info["accepted"]
        cost = info["action_cost"]

        rows.append({
            "episode":      ep,
            "action_index": action,
            "action_name":  ACTION_DEFINITIONS[action].name,
            "churn_before": round(prob_before, 4),
            "churn_after":  round(prob_after, 4),
            "churn_delta":  round(prob_before - prob_after, 4),
            "reward":       round(float(reward), 4),
            "accepted":     "Yes" if accepted else "No",
            "action_cost":  cost,
            "valid_actions": len(valid),
        })

        s = action_stats[action]
        s["total"] += 1
        if accepted:
            s["accepted"] += 1
        s["reward_sum"] += float(reward)
        if float(reward) > 0:
            s["positive"] += 1

    env.close()

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {episodes} episodes -> {OUT_PATH}\n")
    print(f"{'Action':<40} {'Tried':>5} {'Accepted':>8} {'Avg reward':>10} {'Positive%':>10}")
    print("-" * 76)
    for idx, name in enumerate(a.name for a in ACTION_DEFINITIONS):
        s = action_stats[idx]
        if s["total"] == 0:
            continue
        avg_r = s["reward_sum"] / s["total"]
        acc_r = s["accepted"] / s["total"] * 100
        pos_r = s["positive"] / s["total"] * 100
        print(f"{name:<40} {s['total']:>5} {acc_r:>7.1f}% {avg_r:>+10.4f} {pos_r:>9.1f}%")


if __name__ == "__main__":
    run_diagnostic()
