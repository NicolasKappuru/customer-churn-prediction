"""Evaluate DQN and PPO checkpoints across 5 independent customer pools."""

from __future__ import annotations

import argparse
import io
import pickle
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
POOL_DIR = PROJECT_ROOT / "workshops" / "data" / "pool_users"
DATA_DIR = PROJECT_ROOT / "workshops" / "data" / "parquet"

for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent.agent_dqn import DQNAgent
from agent.agent_ppo import PPOAgent
from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv

DQN_CHECKPOINT = W3_ROOT / "checkpoints" / "agent_dqn.pt"
PPO_CHECKPOINT = W3_ROOT / "checkpoints" / "agent_ppo.pt"
N_POOLS = 5
POOL_SIZE = 1500
EPISODES_PER_POOL = 100
_CUTOFF_INT = 20170131
_TXN_KEYS = [
    "payment_method_id", "payment_plan_days", "plan_list_price",
    "actual_amount_paid", "is_auto_renew", "transaction_date",
    "membership_expire_date", "is_cancel",
]


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _load_or_create_pools(
    n_pools: int = N_POOLS,
    pool_size: int = POOL_SIZE,
    base_seed: int = 0,
) -> list[list[dict]]:
    """Load pools from disk or build them from parquet (reads parquet only once)."""
    POOL_DIR.mkdir(parents=True, exist_ok=True)

    paths = [POOL_DIR / f"pool_{i}.pkl" for i in range(n_pools)]
    if all(p.exists() for p in paths):
        pools = []
        for path in paths:
            with open(path, "rb") as f:
                pools.append(pickle.load(f))
        print(f"Loaded {n_pools} pools from {POOL_DIR}")
        return pools

    print("Building pools from parquet files (one-time)...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    churned_msno = train.loc[train["is_churn"] == 1, "msno"].tolist()

    members = pd.read_parquet(
        DATA_DIR / "members_v3.parquet",
        columns=["msno", "city", "registered_via", "bd", "registration_init_time"],
    )
    member_map = members.set_index("msno").to_dict("index")
    available = [m for m in churned_msno if m in member_map]

    txn = pd.concat([
        pd.read_parquet(DATA_DIR / "transactions.parquet"),
        pd.read_parquet(DATA_DIR / "transactions_v2.parquet"),
    ], ignore_index=True)
    txn = txn[txn["transaction_date"] <= _CUTOFF_INT]
    txn = txn.drop_duplicates(subset=["msno", *_TXN_KEYS], keep="first")
    txn_map = txn.groupby("msno").apply(
        lambda g: g.to_dict("records"), include_groups=False
    ).to_dict()

    pools = []
    for i in range(n_pools):
        sample = (
            pd.Series(available)
            .sample(min(pool_size, len(available)), random_state=base_seed + i, replace=False)
            .tolist()
        )
        pool = []
        for msno in sample:
            row = dict(member_map[msno])
            row["transactions"] = txn_map.get(msno, [])
            row["user_logs"] = []
            pool.append(row)

        with open(paths[i], "wb") as f:
            pickle.dump(pool, f)
        pools.append(pool)
        print(f"  Pool {i}: {len(pool)} customers saved to {paths[i].name}")

    return pools


def _evaluate_on_pool(
    agent: Union[DQNAgent, PPOAgent],
    env: KKBoxRetentionEnv,
    pool: list[dict],
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run greedy/deterministic evaluation of one agent on one pool."""
    env.customer_selector._pool = pool
    env._obs_cache = None

    episode_returns: list[float] = []
    action_counts: Counter[int] = Counter()
    accepted_actions = 0
    total_action_cost = 0.0
    probability_reductions: list[float] = []

    for episode in range(1, episodes + 1):
        state, _ = _quiet_env_call(env.reset, seed=seed + episode - 1)
        final_info: dict[str, Any] = {}
        customer_before = env.customer_data

        valid_actions = env.action_manager.get_valid_actions(env.customer_data)
        if isinstance(agent, DQNAgent):
            action = agent.select_action(state, valid_actions, epsilon=0.0)
        else:
            action, _, _ = agent.select_action(state, valid_actions, deterministic=True)

        state, reward, _, _, info = _quiet_env_call(env.step, action)

        accepted_actions += int(env.customer_data is not customer_before)
        action_counts[action] += 1
        total_action_cost += float(info["action_cost"])
        episode_returns.append(float(reward))
        final_info = info

        probability_reductions.append(
            float(final_info["probability_before"]) - float(final_info["probability_after"])
        )

    return {
        "mean_episode_return": float(np.mean(episode_returns)),
        "std_episode_return": float(np.std(episode_returns)),
        "action_frequencies": {i: action_counts[i] for i in range(len(ACTION_DEFINITIONS))},
        "accepted_actions": accepted_actions,
        "total_episode": episodes,
        "total_action_cost": total_action_cost,
        "mean_churn_probability_reduction": float(np.mean(probability_reductions)),
    }


def _print_pool_result(pool_index: int, result: dict[str, Any]) -> None:
    freq = result["action_frequencies"]
    print(
        f"  Pool {pool_index} | "
        f"return={result['mean_episode_return']:+.4f} ± {result['std_episode_return']:.4f} | "
        f"churn_reduction={result['mean_churn_probability_reduction']:+.4f} | "
        f"accepted={result['accepted_actions']}/{result['total_episode']} | "
        f"cost={result['total_action_cost']:.2f}"
    )
    for i, action in enumerate(ACTION_DEFINITIONS):
        print(f"    {i}: {action.name}: {freq[i]}")


def _print_agent_summary(agent_name: str, results: list[dict[str, Any]]) -> None:
    returns = [r["mean_episode_return"] for r in results]
    reductions = [r["mean_churn_probability_reduction"] for r in results]
    print(f"  {agent_name} overall | "
          f"mean_return={np.mean(returns):+.4f} ± {np.std(returns):.4f} | "
          f"mean_churn_reduction={np.mean(reductions):+.4f}")


def compare_agents(
    n_pools: int = N_POOLS,
    pool_size: int = POOL_SIZE,
    episodes: int = EPISODES_PER_POOL,
    base_seed: int = 0,
    eval_seed: int = 2025,
    dqn_checkpoint: Path = DQN_CHECKPOINT,
    ppo_checkpoint: Path = PPO_CHECKPOINT,
) -> None:
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    pools = _load_or_create_pools(n_pools, pool_size, base_seed)

    dqn_agent, dqn_meta = DQNAgent.load_checkpoint(dqn_checkpoint)
    ppo_agent, ppo_meta = PPOAgent.load_checkpoint(ppo_checkpoint)
    print(f"\nDQN checkpoint: {dqn_meta}")
    print(f"PPO checkpoint: {ppo_meta}")

    env = KKBoxRetentionEnv()

    print(f"\n{'='*60}")
    print(f"DQN — {n_pools} pools × {episodes} episodes each")
    print(f"{'='*60}")
    dqn_results = []
    for i, pool in enumerate(pools):
        result = _evaluate_on_pool(dqn_agent, env, pool, episodes, seed=eval_seed + i * episodes)
        _print_pool_result(i, result)
        dqn_results.append(result)
    _print_agent_summary("DQN", dqn_results)

    print(f"\n{'='*60}")
    print(f"PPO — {n_pools} pools × {episodes} episodes each")
    print(f"{'='*60}")
    ppo_results = []
    for i, pool in enumerate(pools):
        result = _evaluate_on_pool(ppo_agent, env, pool, episodes, seed=eval_seed + i * episodes)
        _print_pool_result(i, result)
        ppo_results.append(result)
    _print_agent_summary("PPO", ppo_results)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN and PPO across multiple customer pools."
    )
    parser.add_argument("--pools", type=int, default=N_POOLS)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_POOL)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=2025)
    parser.add_argument("--dqn-checkpoint", type=Path, default=DQN_CHECKPOINT)
    parser.add_argument("--ppo-checkpoint", type=Path, default=PPO_CHECKPOINT)
    args = parser.parse_args()
    compare_agents(
        n_pools=args.pools,
        pool_size=args.pool_size,
        episodes=args.episodes,
        base_seed=args.base_seed,
        eval_seed=args.eval_seed,
        dqn_checkpoint=args.dqn_checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
    )


if __name__ == "__main__":
    main()
