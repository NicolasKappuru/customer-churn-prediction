"""Retention heuristic baselines: cheapest, risk-based, and greedy-simulator."""

from __future__ import annotations

import argparse
import copy
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from environment.constants import ACTION_DEFINITIONS
from environment.env import KKBoxRetentionEnv

_DO_NOTHING_IDX: int = next(
    i for i, a in enumerate(ACTION_DEFINITIONS) if a.name == "Do nothing"
)

# Commercial vs engagement classification (matches customer_env.py)
_IS_COMMERCIAL = {0: False, 1: True, 2: False, 3: True, 4: True, 5: False}
_IS_ENGAGEMENT = {0: True,  1: False, 2: True,  3: False, 4: False, 5: True}


class RetentionHeuristicPolicy:
    """Always picks the valid action with the lowest cost (ties broken by smallest index).

    If no valid action other than Do nothing is available, returns Do nothing.
    """

    def select_action(
        self,
        valid_actions: Sequence[int],
        customer_data: dict,
        action_manager: Any,
        **kwargs: Any,
    ) -> int:
        candidates = [a for a in valid_actions if a != _DO_NOTHING_IDX]
        if not candidates:
            return _DO_NOTHING_IDX
        return min(candidates, key=lambda a: (ACTION_DEFINITIONS[a].cost, a))


class RetentionRiskPolicy:
    """Risk-based heuristic: high churn risk → cheapest commercial; low risk → cheapest engagement.

    If the required category has no valid candidates, falls back to the other category,
    then to Do nothing.
    """

    def select_action(
        self,
        valid_actions: Sequence[int],
        customer_data: dict,
        action_manager: Any,
        p_churn: float = 0.5,
        **kwargs: Any,
    ) -> int:
        candidates = [a for a in valid_actions if a != _DO_NOTHING_IDX]
        if not candidates:
            return _DO_NOTHING_IDX

        if p_churn > 0.5:
            primary   = [a for a in candidates if _IS_COMMERCIAL.get(a, False)]
            secondary = [a for a in candidates if _IS_ENGAGEMENT.get(a, False)]
        else:
            primary   = [a for a in candidates if _IS_ENGAGEMENT.get(a, False)]
            secondary = [a for a in candidates if _IS_COMMERCIAL.get(a, False)]

        pool = primary or secondary or candidates
        return min(pool, key=lambda a: (ACTION_DEFINITIONS[a].cost, a))


class RetentionGreedyPolicy:
    """Greedy simulator: scores each action by 0.5 * Δp * (1 − cost) and picks the best.

    Requires the supervised_model and preprocessor_customer_data from the env.
    If no action improves churn more than Do nothing, returns Do nothing.
    """

    def __init__(self, supervised_model, preprocessor) -> None:
        self._model = supervised_model
        self._preprocessor = preprocessor

    def select_action(
        self,
        valid_actions: Sequence[int],
        customer_data: dict,
        action_manager: Any,
        p_churn_before: float = 0.0,
        **kwargs: Any,
    ) -> int:
        candidates = [a for a in valid_actions if a != _DO_NOTHING_IDX]
        if not candidates:
            return _DO_NOTHING_IDX

        best_action = _DO_NOTHING_IDX
        best_score = 0.0

        for action in candidates:
            updated, cost = action_manager.apply_effect(copy.deepcopy(customer_data), action)
            with redirect_stdout(io.StringIO()):
                features = self._preprocessor.preprocess_customer(updated)
            p_after = float(self._model.predict_probability(features))
            delta_p = p_churn_before - p_after
            score = 0.5 * delta_p * (1.0 - cost)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action


def _quiet_env_call(function: Any, *args: Any, **kwargs: Any) -> Any:
    with redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def run_heuristic_policy(
    episodes: int = 100,
    seed: int = 42,
    customer_mode: str = "agent",
    customer_checkpoint: str | Path | None = None,
    strategy: str = "cheapest",
) -> list[dict[str, Any]]:
    """Run one of the retention heuristic strategies and report episode diagnostics."""
    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    import random
    random.seed(seed)
    np.random.seed(seed)

    env = KKBoxRetentionEnv(customer_mode=customer_mode, customer_checkpoint=customer_checkpoint)

    if strategy == "cheapest":
        policy: Any = RetentionHeuristicPolicy()
    elif strategy == "risk":
        policy = RetentionRiskPolicy()
    elif strategy == "greedy":
        policy = RetentionGreedyPolicy(env.supervised_model, env.preprocessor_customer_data)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: cheapest, risk, greedy.")

    results: list[dict[str, Any]] = []
    action_frequencies: Counter[int] = Counter()
    accepted_actions = 0
    total_action_cost = 0.0
    probability_reductions: list[float] = []
    success_count = 0
    interventions = 0
    accepted_interventions = 0

    try:
        print(
            f"Heuristic policy [{strategy}] | episodes={episodes} | seed={seed} | "
            f"customer={customer_mode} | customers={len(env.customer_selector._pool)}"
        )

        for episode in range(1, episodes + 1):
            observation, _ = _quiet_env_call(env.reset, seed=seed + episode - 1)
            episode_return = 0.0
            episode_length = 0
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False

            while not (terminated or truncated):
                customer_before = env.customer_data
                valid_actions = env.get_valid_actions()

                if strategy == "risk":
                    action = policy.select_action(
                        valid_actions, env.customer_data, env.action_manager,
                        p_churn=env.probability_churn_before,
                    )
                elif strategy == "greedy":
                    action = policy.select_action(
                        valid_actions, env.customer_data, env.action_manager,
                        p_churn_before=env.probability_churn_before,
                    )
                else:
                    action = policy.select_action(
                        valid_actions, env.customer_data, env.action_manager,
                    )

                observation, reward, terminated, truncated, info = _quiet_env_call(env.step, action)

                accepted = env.customer_data is not customer_before
                accepted_actions += int(accepted)
                episode_length += 1
                episode_return += float(reward)
                action_frequencies[action] += 1
                total_action_cost += float(info["action_cost"])
                final_info = info

                if action != _DO_NOTHING_IDX:
                    interventions += 1
                    if accepted:
                        accepted_interventions += 1

            prob_before = float(final_info["probability_before"])
            prob_after  = float(final_info["probability_after"])
            probability_reductions.append(prob_before - prob_after)
            if prob_after < prob_before:
                success_count += 1

            results.append({
                "episode_return":     episode_return,
                "probability_before": prob_before,
                "probability_after":  prob_after,
                "action_cost":        float(final_info["action_cost"]),
            })
            print(
                f"Episode {episode:03d}/{episodes} | "
                f"return={episode_return:+.4f} | "
                f"churn_prob={prob_after:.3f}"
            )

        all_action_indices = range(len(ACTION_DEFINITIONS))
        missing_actions = [i for i in all_action_indices if action_frequencies[i] == 0]
        success_rate    = success_count / episodes
        acceptance_rate = accepted_interventions / interventions if interventions > 0 else 0.0
        episode_returns = [r["episode_return"] for r in results]

        print("\nEvaluation summary")
        print(f"Mean episode return:      {np.mean(episode_returns):+.4f}")
        print(f"Std episode return:       {np.std(episode_returns):.4f}")
        print(f"Success rate:             {success_rate:.1%}  ({success_count}/{episodes} episodes improved)")
        print(
            f"Acceptance rate:          {acceptance_rate:.1%}  "
            f"({accepted_interventions}/{interventions} interventions accepted)"
        )
        print("Action frequency (index: name: count):")
        for index, definition in enumerate(ACTION_DEFINITIONS):
            print(f"  {index}: {definition.name}: {action_frequencies[index]}")
        print(f"All actions occurred:     {'YES' if not missing_actions else 'NO'}")
        if missing_actions:
            print(f"Actions not selected:     {missing_actions}")
        print(f"Accepted actions:         {accepted_actions}/{sum(action_frequencies.values())}")
        print(f"Total action cost:        {total_action_cost:.2f}")
        print(f"Mean churn-prob reduction:{np.mean(probability_reductions):+.4f}")

        return results
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a retention heuristic policy on KKBoxRetentionEnv."
    )
    parser.add_argument("--episodes",  type=int, default=100)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--customer",  choices=["static", "agent"], default="agent")
    parser.add_argument(
        "--customer-checkpoint", type=str,
        default=str(W3_ROOT / "checkpoints" / "customer_agent_ppo.pt"),
    )
    parser.add_argument(
        "--strategy", choices=["cheapest", "risk", "greedy"], default="cheapest",
        help="cheapest: lowest cost; risk: commercial vs engagement by p_churn; greedy: simulator-scored",
    )
    args = parser.parse_args()
    run_heuristic_policy(
        episodes=args.episodes,
        seed=args.seed,
        customer_mode=args.customer,
        customer_checkpoint=args.customer_checkpoint,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
