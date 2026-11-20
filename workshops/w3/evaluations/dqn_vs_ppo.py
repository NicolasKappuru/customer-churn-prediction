"""Evaluate DQN, PPO, Heuristic and Random across 5 held-out test pools."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from collections import Counter
from contextlib import redirect_stdout
from datetime import date as _date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT   = PROJECT_ROOT / "workshops" / "w3"
_EVAL_DIR = Path(__file__).resolve().parent

for _path in (
    str(W3_ROOT),
    str(W3_ROOT / "environment"),
    str(W3_ROOT / "baselines"),
    str(_EVAL_DIR),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _test_pools import load_or_create_test_pools, use_pool
from agent.agent_dqn import DQNAgent
from agent.agent_ppo import PPOAgent
from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv
from heuristic_policy import HeuristicPolicy

DQN_CHECKPOINT  = W3_ROOT / "checkpoints" / "agent_dqn.pt"
PPO_CHECKPOINT  = W3_ROOT / "checkpoints" / "agent_ppo.pt"
N_POOLS              = 5
N_CHURN_PER_POOL     = 90
N_NON_CHURN_PER_POOL = 10
EPISODES_PER_POOL    = 100

_RESULTS_DIR = _EVAL_DIR / "results"
_CUTOFF_DATE = _date(2017, 1, 31)

_DO_NOTHING_IDX: int = next(
    i for i, a in enumerate(ACTION_DEFINITIONS) if a.name == "Do nothing"
)

# ActionFn: callable(observation, valid_actions, env) -> action_index
ActionFn = Callable[[np.ndarray, list, KKBoxRetentionEnv], int]


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _parse_yyyymmdd(d: Any) -> _date | None:
    try:
        d = int(d)
        return _date(d // 10000, (d % 10000) // 100, d % 100)
    except (ValueError, TypeError):
        return None


def _extract_features(customer: dict | None) -> dict:
    """Extract 4 human-readable features from a customer dict."""
    if customer is None:
        return dict(auto_renew=None, plan_price=None, days_to_expire=None, log_secs=None)
    txns = customer.get("transactions", [])
    if txns:
        last = max(txns, key=lambda t: t.get("transaction_date", 0))
        auto_renew     = int(last.get("is_auto_renew", 0))
        plan_price     = float(last.get("plan_list_price", 0))
        expire         = _parse_yyyymmdd(last.get("membership_expire_date", 0))
        days_to_expire = (expire - _CUTOFF_DATE).days if expire else None
    else:
        auto_renew = plan_price = days_to_expire = None
    logs = customer.get("user_logs", [])
    if logs:
        last_log = max(logs, key=lambda l: l.get("date", 0))
        log_secs = float(last_log.get("total_secs", 0))
    else:
        log_secs = None
    return dict(auto_renew=auto_renew, plan_price=plan_price,
                days_to_expire=days_to_expire, log_secs=log_secs)


def _evaluate_on_pool(
    action_fn: ActionFn,
    env: KKBoxRetentionEnv,
    pool: list[dict],
    episodes: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one agent on one pool; returns (aggregate_stats, episode_records)."""
    use_pool(env, pool)

    episode_returns: list[float]          = []
    action_counts: Counter[int]           = Counter()
    favorable_counts: dict[int, Counter]  = {i: Counter() for i in range(len(ACTION_DEFINITIONS))}
    accepted_actions       = 0
    total_action_cost      = 0.0
    probability_reductions: list[float]   = []
    valid_action_sizes: Counter[int]      = Counter()
    success_count          = 0
    interventions          = 0
    accepted_interventions = 0
    episode_records: list[dict[str, Any]] = []

    for episode in range(1, episodes + 1):
        state, _ = _quiet_env_call(env.reset, seed=seed + episode - 1)
        customer_before = env.customer_data
        feat_before = _extract_features(customer_before)

        valid_actions = env.action_manager.get_valid_actions(env.customer_data)
        valid_action_sizes[len(valid_actions)] += 1

        action = action_fn(state, valid_actions, env)

        scenario = (
            "favorable"
            if env.action_manager.is_favorable(action, env.customer_data)
            else "unfavorable"
        )
        favorable_counts[action][scenario] += 1

        state, reward, _, _, info = _quiet_env_call(env.step, action)

        accepted    = env.customer_data is not customer_before
        feat_after  = _extract_features(env.customer_data)
        accepted_actions   += int(accepted)
        action_counts[action] += 1
        total_action_cost  += float(info["action_cost"])
        episode_returns.append(float(reward))

        prob_before = float(info["probability_before"])
        prob_after  = float(info["probability_after"])
        probability_reductions.append(prob_before - prob_after)
        if prob_after < prob_before:
            success_count += 1

        if action != _DO_NOTHING_IDX:
            interventions += 1
            if accepted:
                accepted_interventions += 1

        episode_records.append({
            "episode":              episode,
            "action":               action,
            "action_name":          ACTION_DEFINITIONS[action].name,
            "valid_actions":        json.dumps(valid_actions),
            "valid_actions_count":  len(valid_actions),
            "accepted":             int(accepted),
            "p_churn_before":       round(prob_before, 6),
            "p_churn_after":        round(prob_after, 6),
            "p_churn_reduction":    round(prob_before - prob_after, 6),
            "cost":                 float(info["action_cost"]),
            "reward":               float(reward),
            "auto_renew_before":    feat_before["auto_renew"],
            "plan_price_before":    feat_before["plan_price"],
            "days_to_expire_before": feat_before["days_to_expire"],
            "log_secs_before":      feat_before["log_secs"],
            "auto_renew_after":     feat_after["auto_renew"],
            "plan_price_after":     feat_after["plan_price"],
            "days_to_expire_after": feat_after["days_to_expire"],
            "log_secs_after":       feat_after["log_secs"],
        })

    aggregate = {
        "mean_episode_return":              float(np.mean(episode_returns)),
        "std_episode_return":               float(np.std(episode_returns)),
        "action_frequencies":               {i: action_counts[i] for i in range(len(ACTION_DEFINITIONS))},
        "favorable_counts":                 {i: dict(favorable_counts[i]) for i in range(len(ACTION_DEFINITIONS))},
        "accepted_actions":                 accepted_actions,
        "total_episode":                    episodes,
        "total_action_cost":                total_action_cost,
        "mean_churn_probability_reduction": float(np.mean(probability_reductions)),
        "valid_action_sizes":               dict(valid_action_sizes),
        "success_count":                    success_count,
        "success_rate":                     success_count / episodes,
        "interventions":                    interventions,
        "accepted_interventions":           accepted_interventions,
        "acceptance_rate":                  accepted_interventions / interventions if interventions > 0 else 0.0,
    }
    return aggregate, episode_records


def _print_pool_result(pool_index: int, result: dict[str, Any]) -> None:
    freq = result["action_frequencies"]
    print(
        f"  Pool {pool_index} | "
        f"return={result['mean_episode_return']:+.4f} ± {result['std_episode_return']:.4f} | "
        f"success={result['success_rate']:.1%} | "
        f"acceptance={result['acceptance_rate']:.1%} | "
        f"churn_red={result['mean_churn_probability_reduction']:+.4f} | "
        f"cost={result['total_action_cost']:.2f}"
    )
    fav = result["favorable_counts"]
    for i, action in enumerate(ACTION_DEFINITIONS):
        f = fav[i].get("favorable", 0)
        u = fav[i].get("unfavorable", 0)
        print(f"    {i}: {action.name}: {freq[i]}  ({f} fav / {u} unfav)")


def _print_agent_summary(agent_name: str, results: list[dict[str, Any]]) -> None:
    returns    = [r["mean_episode_return"] for r in results]
    reductions = [r["mean_churn_probability_reduction"] for r in results]
    n_eps      = sum(r["total_episode"] for r in results)
    successes  = sum(r["success_count"] for r in results)
    intv       = sum(r["interventions"] for r in results)
    acc_intv   = sum(r["accepted_interventions"] for r in results)
    print(
        f"  {agent_name} overall | "
        f"mean_return={np.mean(returns):+.4f} ± {np.std(returns):.4f} | "
        f"success={successes/n_eps:.1%} | "
        f"acceptance={acc_intv/intv:.1%} | "
        f"churn_red={np.mean(reductions):+.4f}"
    )


def _print_comparison(all_results: dict[str, list[dict[str, Any]]]) -> None:
    agents = list(all_results.keys())

    def _agg(results: list[dict]) -> dict:
        n_eps      = sum(r["total_episode"] for r in results)
        returns    = [r["mean_episode_return"] for r in results]
        reductions = [r["mean_churn_probability_reduction"] for r in results]
        successes  = sum(r["success_count"] for r in results)
        intv       = sum(r["interventions"] for r in results)
        acc_intv   = sum(r["accepted_interventions"] for r in results)
        freq: Counter[int] = Counter()
        for r in results:
            for i, c in r["action_frequencies"].items():
                freq[i] += c
        return {
            "mean_return":     float(np.mean(returns)),
            "std_return":      float(np.std(returns)),
            "mean_reduction":  float(np.mean(reductions)),
            "success_rate":    successes / n_eps,
            "acceptance_rate": acc_intv / intv if intv > 0 else 0.0,
            "total_cost":      sum(r["total_action_cost"] for r in results),
            "total_eps":       n_eps,
            "freq":            freq,
        }

    agg = {name: _agg(all_results[name]) for name in agents}

    col_w = 12
    label_w = 32
    W = label_w + col_w * len(agents)

    print(f"\n{'='*W}")
    print(f"FINAL COMPARISON — held-out test set  ({' | '.join(agents)})")
    print(f"{'='*W}")

    header = " " * (label_w + 2)
    for name in agents:
        header += f"{name:>{col_w}}"
    print(header)
    print(f"  {'-'*(label_w + col_w * len(agents))}")

    def _best_idx(key: str, higher_is_better: bool = True) -> str:
        vals = [agg[n][key] for n in agents]
        return agents[vals.index(max(vals) if higher_is_better else min(vals))]

    def _row(label: str, key: str, fmt: str, higher_is_better: bool = True) -> None:
        best = _best_idx(key, higher_is_better)
        line = f"  {label:<{label_w}}"
        for name in agents:
            cell = fmt.format(agg[name][key])
            mark = "◀" if name == best else " "
            line += f"{cell:>{col_w - 1}}{mark}"
        print(line)

    _row("Mean return (across pools)", "mean_return",    "{:+.4f}")
    _row("Std return (across pools)",  "std_return",     "{:.4f}",  False)
    _row("Success rate",               "success_rate",   "{:.1%}")
    _row("Acceptance rate",            "acceptance_rate","{:.1%}")
    _row("Mean churn reduction",       "mean_reduction", "{:+.4f}")
    _row("Total action cost",          "total_cost",     "{:.2f}",  False)

    print(f"\n  {'Action':<30}" + "".join(f"{n:>{col_w}}" for n in agents))
    print(f"  {'-'*(28 + col_w * len(agents))}")
    for i, action in enumerate(ACTION_DEFINITIONS):
        line = f"  {i}: {action.name:<26}"
        for name in agents:
            line += f"{agg[name]['freq'][i]:>{col_w}}"
        print(line)

    print(f"\n  Pool-by-pool mean return:")
    pool_header = f"  {'Pool':<6}" + "".join(f"{n:>{col_w}}" for n in agents) + "   Winner"
    print(pool_header)
    print(f"  {'-'*(4 + col_w * len(agents) + 10)}")
    for i in range(len(next(iter(all_results.values())))):
        pool_returns = {name: all_results[name][i]["mean_episode_return"] for name in agents}
        best = max(pool_returns, key=pool_returns.get)
        line = f"  {i:<6}"
        for name in agents:
            line += f"{pool_returns[name]:>+{col_w}.4f}"
        line += f"   {best}"
        print(line)

    print(f"{'='*W}")


def _save_results(
    results_dir: Path,
    all_results: dict[str, list[dict[str, Any]]],
    all_episodes: list[dict[str, Any]],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Summary CSV: one row per (agent, pool) ──────────────────────────────
    summary_rows: list[dict] = []
    for agent, results in all_results.items():
        for pool_i, r in enumerate(results):
            summary_rows.append({
                "agent":               agent,
                "pool":                pool_i,
                "mean_return":         round(r["mean_episode_return"], 6),
                "std_return":          round(r["std_episode_return"], 6),
                "success_rate":        round(r["success_rate"], 6),
                "acceptance_rate":     round(r["acceptance_rate"], 6),
                "mean_churn_reduction": round(r["mean_churn_probability_reduction"], 6),
                "total_cost":          round(r["total_action_cost"], 4),
                "episodes":            r["total_episode"],
                "success_count":       r["success_count"],
                "interventions":       r["interventions"],
                "accepted_interventions": r["accepted_interventions"],
            })

    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── Episodes CSV: one row per (policy, pool, episode) ───────────────────
    episodes_path = results_dir / "episodes.csv"
    first = all_episodes[0]
    leading  = ["policy", "pool", "episode"]
    trailing = [k for k in first.keys() if k not in leading]
    cols = leading + trailing
    with open(episodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(all_episodes)

    print(f"\nResults saved to {results_dir}")
    print(f"  {summary_path.name}: {len(summary_rows)} rows  "
          f"({len(all_results)} agents × {len(summary_rows) // len(all_results)} pools)")
    print(f"  {episodes_path.name}: {len(all_episodes)} rows  "
          f"({len(all_results)} agents × {len(all_episodes) // len(all_results)} episodes each)")


def compare_agents(
    n_pools: int     = N_POOLS,
    n_churn: int     = N_CHURN_PER_POOL,
    n_non_churn: int = N_NON_CHURN_PER_POOL,
    episodes: int    = EPISODES_PER_POOL,
    base_seed: int   = 0,
    eval_seed: int   = 2025,
    dqn_checkpoint: Path = DQN_CHECKPOINT,
    ppo_checkpoint: Path = PPO_CHECKPOINT,
) -> None:
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    pools = load_or_create_test_pools(n_pools, n_churn, n_non_churn, base_seed)

    dqn_agent, dqn_meta = DQNAgent.load_checkpoint(dqn_checkpoint)
    ppo_agent, ppo_meta = PPOAgent.load_checkpoint(ppo_checkpoint)
    heuristic           = HeuristicPolicy()
    random_rng          = np.random.default_rng(eval_seed)

    print(f"\nDQN checkpoint: {dqn_meta}")
    print(f"PPO checkpoint: {ppo_meta}")

    # ── Define action callables ─────────────────────────────────────────────
    def _dqn(state, va, env):
        return dqn_agent.select_action(state, va, epsilon=0.0)

    def _ppo(state, va, env):
        return ppo_agent.select_action(state, va, deterministic=True)[0]

    def _heur(state, va, env):
        return heuristic.select_action(va, env.customer_data, env.action_manager)

    def _rand(state, va, env):
        return int(random_rng.choice(va))

    policies: list[tuple[str, ActionFn]] = [
        ("DQN",       _dqn),
        ("PPO",       _ppo),
        ("Heuristic", _heur),
        ("Random",    _rand),
    ]

    env = KKBoxRetentionEnv()
    all_results: dict[str, list[dict[str, Any]]] = {}
    all_episodes: list[dict[str, Any]] = []

    try:
        for agent_name, action_fn in policies:
            print(f"\n{'='*60}")
            print(
                f"{agent_name} — {n_pools} pools × {episodes} episodes "
                f"({n_churn} churn + {n_non_churn} non-churn per pool)"
            )
            print(f"{'='*60}")
            results = []
            for i, pool in enumerate(pools):
                result, ep_records = _evaluate_on_pool(
                    action_fn, env, pool, episodes,
                    seed=eval_seed + i * episodes,
                )
                for rec in ep_records:
                    rec["policy"] = agent_name
                    rec["pool"]   = i
                    all_episodes.append(rec)
                _print_pool_result(i, result)
                results.append(result)
            _print_agent_summary(agent_name, results)
            all_results[agent_name] = results

        _print_comparison(all_results)
        _save_results(_RESULTS_DIR, all_results, all_episodes)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN, PPO, Heuristic and Random on held-out test pools."
    )
    parser.add_argument("--pools",          type=int,  default=N_POOLS)
    parser.add_argument("--churn",          type=int,  default=N_CHURN_PER_POOL)
    parser.add_argument("--non-churn",      type=int,  default=N_NON_CHURN_PER_POOL)
    parser.add_argument("--episodes",       type=int,  default=EPISODES_PER_POOL)
    parser.add_argument("--base-seed",      type=int,  default=0)
    parser.add_argument("--eval-seed",      type=int,  default=2025)
    parser.add_argument("--dqn-checkpoint", type=Path, default=DQN_CHECKPOINT)
    parser.add_argument("--ppo-checkpoint", type=Path, default=PPO_CHECKPOINT)
    args = parser.parse_args()
    compare_agents(
        n_pools=args.pools,
        n_churn=args.churn,
        n_non_churn=args.non_churn,
        episodes=args.episodes,
        base_seed=args.base_seed,
        eval_seed=args.eval_seed,
        dqn_checkpoint=args.dqn_checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
    )


if __name__ == "__main__":
    main()
