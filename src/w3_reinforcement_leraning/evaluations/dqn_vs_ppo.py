"""Compare DQN and PPO on the same customers with paired random seeds."""

from __future__ import annotations

import argparse
import io
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from w3_reinforcement_leraning.agent.agent_dqn import DQNAgent
from w3_reinforcement_leraning.agent.agent_ppo import PPOAgent
from w3_reinforcement_leraning.environment.constants import ACTION_DEFINITIONS
from w3_reinforcement_leraning.environment.env import TelcoRetentionEnv


DEFAULT_DQN_CHECKPOINT = (
    PROJECT_ROOT / "src" / "w3_reinforcement_leraning" / "checkpoints"
    / "agent_best.pt"
)
DEFAULT_PPO_CHECKPOINT = (
    PROJECT_ROOT / "src" / "w3_reinforcement_leraning" / "checkpoints"
    / "agent_ppo_best.pt"
)
DEFAULT_CUSTOMERS = 100
DEFAULT_REPETITIONS = 10
DEFAULT_SEED = 2025
CONFIDENCE_LEVEL = 0.95


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Suppress preprocessing progress output during evaluation."""
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def _mean_confidence_interval(
    values: list[float],
    confidence: float = CONFIDENCE_LEVEL,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float, int]:
    """Return mean, Student-t confidence limits, and sample size."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    sample_size = len(finite_values)
    if sample_size == 0:
        return float("nan"), float("nan"), float("nan"), 0

    mean = float(finite_values.mean())
    if sample_size == 1:
        return mean, float("nan"), float("nan"), sample_size

    standard_error = float(finite_values.std(ddof=1) / np.sqrt(sample_size))
    critical_value = float(t.ppf((1 + confidence) / 2, df=sample_size - 1))
    margin = critical_value * standard_error
    lower, upper = mean - margin, mean + margin
    if bounds is not None:
        lower = max(bounds[0], lower)
        upper = min(bounds[1], upper)
    return mean, lower, upper, sample_size


def _paired_seed(seed: int, customer_index: int) -> int:
    """Derive a repeatable per-customer seed shared by both policies."""
    return (seed * 1_000_003 + customer_index) % (2**32)


def _check_agent_action_sizes(
    dqn_agent: DQNAgent, ppo_agent: PPOAgent
) -> None:
    """Reject checkpoints trained with a different action mapping."""
    expected_size = len(ACTION_DEFINITIONS)
    for name, agent in (("DQN", dqn_agent), ("PPO", ppo_agent)):
        if agent.config.action_size != expected_size:
            raise ValueError(
                f"{name} checkpoint has action_size={agent.config.action_size}; "
                f"the current environment has {expected_size} actions. "
                "Retrain this agent with the current action definitions before "
                "comparing policies."
            )


def _evaluate_policy(
    env: TelcoRetentionEnv,
    agent: DQNAgent | PPOAgent,
    policy_name: str,
    customers: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    """Evaluate one policy once on every customer in the fixed cohort."""
    records: list[dict[str, float | int | str]] = []
    for customer_index in range(len(customers)):
        customer = customers.iloc[[customer_index]].copy(deep=True)
        env.customer_selector.df = customer
        episode_seed = _paired_seed(seed, customer_index)

        np.random.seed(episode_seed)
        state, _ = _quiet_env_call(env.reset, seed=episode_seed)
        if env.customer_data_df is None:
            raise RuntimeError("Environment reset did not initialize a customer.")

        valid_actions = env.action_manager.get_valid_actions(env.customer_data_df)
        if isinstance(agent, DQNAgent):
            action = agent.select_action(state, valid_actions, epsilon=0.0)
        else:
            action, _, _ = agent.select_action(
                state, valid_actions, deterministic=True
            )

        random.seed(episode_seed)
        next_state, reward, terminated, truncated, info = _quiet_env_call(
            env.step, action
        )
        if not (terminated or truncated):
            raise RuntimeError("Expected each customer evaluation to end in one step.")
        if not env.observation_space.contains(next_state):
            raise ValueError(
                f"{policy_name} produced an observation outside the environment "
                f"space for customer {customer_index}."
            )

        churn_reduction = float(info["probability_before"]) - float(
            info["probability_after"]
        )
        records.append(
            {
                "policy": policy_name,
                "customer_index": customer_index,
                "action": int(action),
                "accepted": int(not customer.equals(env.customer_data_df)),
                "net_reward": float(reward),
                "churn_reduction": churn_reduction,
                "action_cost": float(info["action_cost"]),
            }
        )

    rewards = np.asarray([row["net_reward"] for row in records], dtype=float)
    churn_reductions = np.asarray(
        [row["churn_reduction"] for row in records], dtype=float
    )
    summary: dict[str, float | int | str] = {
        "policy": policy_name,
        "mean_net_reward": float(rewards.mean()),
        "total_net_reward": float(rewards.sum()),
        "mean_churn_reduction": float(churn_reductions.mean()),
        "accepted_offers": int(sum(row["accepted"] for row in records)),
        "total_action_cost": float(sum(row["action_cost"] for row in records)),
    }
    return summary, records


def _format_interval(
    mean: float, lower: float, upper: float, sample_size: int, digits: int = 5
) -> str:
    if sample_size == 0:
        return "n/a"
    if sample_size == 1:
        return f"{mean:.{digits}f} (1 seed; CI n/a)"
    return f"{mean:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def _print_overall_results(
    summaries: list[dict[str, float | int | str]],
    customers_count: int,
) -> None:
    frame = pd.DataFrame(summaries)
    print(f"\nOverall results: mean and {CONFIDENCE_LEVEL:.0%} CI across seeds")
    for policy_name in ("DQN", "PPO"):
        policy_rows = frame.loc[frame["policy"] == policy_name]
        print(f"\n{policy_name} ({len(policy_rows)} repetitions)")
        for metric, label in (
            ("mean_net_reward", "Mean net reward / customer"),
            ("total_net_reward", "Total net reward / cohort"),
            ("mean_churn_reduction", "Mean churn reduction"),
            ("accepted_offers", "Accepted offers / cohort"),
            ("total_action_cost", "Total action cost / cohort"),
        ):
            metric_bounds = {
                "mean_net_reward": (-1.0, 1.0),
                "total_net_reward": (
                    -customers_count,
                    customers_count,
                ),
                "mean_churn_reduction": (-1.0, 1.0),
                "accepted_offers": (0.0, customers_count),
                "total_action_cost": (
                    0.0,
                    customers_count * max(
                        definition.cost for definition in ACTION_DEFINITIONS
                    ),
                ),
            }
            mean, lower, upper, sample_size = _mean_confidence_interval(
                policy_rows[metric].astype(float).tolist(),
                bounds=metric_bounds[metric],
            )
            print(
                f"  {label}: "
                f"{_format_interval(mean, lower, upper, sample_size)}"
            )

    print("\nPaired difference (DQN - PPO), by repetition")
    paired = frame.pivot(index="seed", columns="policy")
    for metric, label in (
        ("mean_net_reward", "Mean net reward / customer"),
        ("mean_churn_reduction", "Mean churn reduction"),
    ):
        differences = (
            paired[(metric, "DQN")] - paired[(metric, "PPO")]
        ).astype(float).tolist()
        mean, lower, upper, sample_size = _mean_confidence_interval(
            differences,
            bounds=(-2.0, 2.0),
        )
        print(
            f"  {label}: "
            f"{_format_interval(mean, lower, upper, sample_size)}"
        )


def _print_action_results(
    records_by_seed: list[tuple[int, list[dict[str, float | int | str]]]],
    customers_count: int,
) -> None:
    rows: list[dict[str, float | int | str]] = []
    for seed, records in records_by_seed:
        for policy_name in ("DQN", "PPO"):
            policy_records = [
                row for row in records if row["policy"] == policy_name
            ]
            for action_index, definition in enumerate(ACTION_DEFINITIONS):
                selected = [
                    row for row in policy_records if row["action"] == action_index
                ]
                rows.append(
                    {
                        "seed": seed,
                        "policy": policy_name,
                        "action_index": action_index,
                        "action_name": definition.name,
                        "selections": len(selected),
                        "acceptance_rate": (
                            float(np.mean([row["accepted"] for row in selected]))
                            if selected
                            else float("nan")
                        ),
                        "mean_net_reward": (
                            float(np.mean([row["net_reward"] for row in selected]))
                            if selected
                            else float("nan")
                        ),
                        "mean_churn_reduction": (
                            float(
                                np.mean(
                                    [row["churn_reduction"] for row in selected]
                                )
                            )
                            if selected
                            else float("nan")
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    print(
        f"\nPer-action results: mean and {CONFIDENCE_LEVEL:.0%} CI "
        "across seed-level means"
    )
    for policy_name in ("DQN", "PPO"):
        print(f"\n{policy_name}")
        policy_frame = frame.loc[frame["policy"] == policy_name]
        output_rows: list[dict[str, str]] = []
        for action_index, definition in enumerate(ACTION_DEFINITIONS):
            action_rows = policy_frame.loc[
                policy_frame["action_index"] == action_index
            ]
            result: dict[str, str] = {"Action": definition.name}
            for metric, label, digits in (
                ("selections", f"Selections / {customers_count}", 2),
                ("acceptance_rate", "Acceptance rate", 3),
                ("mean_net_reward", "Net reward / selection", 5),
                ("mean_churn_reduction", "Churn reduction / selection", 5),
            ):
                mean, lower, upper, sample_size = _mean_confidence_interval(
                    action_rows[metric].astype(float).tolist(),
                    bounds={
                        "selections": (0.0, customers_count),
                        "acceptance_rate": (0.0, 1.0),
                        "mean_net_reward": (-1.0, 1.0),
                        "mean_churn_reduction": (-1.0, 1.0),
                    }[metric],
                )
                result[label] = _format_interval(
                    mean, lower, upper, sample_size, digits
                )
            output_rows.append(result)
        print(pd.DataFrame(output_rows).to_string(index=False))


def compare_dqn_vs_ppo(
    dqn_checkpoint: str | Path = DEFAULT_DQN_CHECKPOINT,
    ppo_checkpoint: str | Path = DEFAULT_PPO_CHECKPOINT,
    customers_count: int = DEFAULT_CUSTOMERS,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compare both policies on a fixed customer cohort over paired seeds."""
    if customers_count <= 0:
        raise ValueError("customers_count must be greater than zero.")
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero.")

    dqn_agent, dqn_metadata = DQNAgent.load_checkpoint(dqn_checkpoint)
    ppo_agent, ppo_metadata = PPOAgent.load_checkpoint(ppo_checkpoint)
    _check_agent_action_sizes(dqn_agent, ppo_agent)
    np.random.seed(seed)
    env = TelcoRetentionEnv()

    try:
        available_customers = env.customer_selector.df
        if len(available_customers) < customers_count:
            raise ValueError(
                f"Requested {customers_count} customers, but the environment "
                f"sample contains only {len(available_customers)}."
            )
        customers = available_customers.sample(
            n=customers_count,
            replace=False,
            random_state=seed,
        ).reset_index(drop=True)

        print(
            f"DQN vs PPO | customers={customers_count} | repetitions={repetitions} "
            f"| base_seed={seed} | confidence={CONFIDENCE_LEVEL:.0%}"
        )
        print(f"DQN checkpoint: {Path(dqn_checkpoint)} | metadata={dqn_metadata}")
        print(f"PPO checkpoint: {Path(ppo_checkpoint)} | metadata={ppo_metadata}")
        print("The same fixed customer cohort and per-customer random draws are used.")

        summaries: list[dict[str, float | int | str]] = []
        records_by_seed: list[
            tuple[int, list[dict[str, float | int | str]]]
        ] = []
        for repetition in range(repetitions):
            repetition_seed = seed + repetition
            repetition_records: list[dict[str, float | int | str]] = []
            for policy_name, agent in (
                ("DQN", dqn_agent),
                ("PPO", ppo_agent),
            ):
                summary, records = _evaluate_policy(
                    env,
                    agent,
                    policy_name,
                    customers,
                    repetition_seed,
                )
                summary["seed"] = repetition_seed
                summaries.append(summary)
                repetition_records.extend(records)

            records_by_seed.append((repetition_seed, repetition_records))

        _print_overall_results(summaries, customers_count)
        _print_action_results(records_by_seed, customers_count)
        return {
            "summaries": summaries,
            "records_by_seed": records_by_seed,
            "customer_indices": list(range(customers_count)),
            "base_seed": seed,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DQN and PPO on the same Telco customer cohort."
    )
    parser.add_argument(
        "--dqn-checkpoint",
        type=Path,
        default=DEFAULT_DQN_CHECKPOINT,
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=DEFAULT_PPO_CHECKPOINT,
    )
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMERS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    arguments = parser.parse_args()
    compare_dqn_vs_ppo(
        dqn_checkpoint=arguments.dqn_checkpoint,
        ppo_checkpoint=arguments.ppo_checkpoint,
        customers_count=arguments.customers,
        repetitions=arguments.repetitions,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
