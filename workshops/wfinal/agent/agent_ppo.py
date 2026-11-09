"""Proximal Policy Optimization agent with valid-action masking."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT = PROJECT_ROOT / "workshops" / "w3"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from constants import ACTION_DEFINITIONS, N_OBSERVATION


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters, kept together for easy experiments."""

    state_size: int = N_OBSERVATION
    action_size: int = len(ACTION_DEFINITIONS)
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.95
    weight_decay: float = 0.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 4
    gradient_clip_value: float = 0.5


class PPOTransition:
    """One on-policy transition with the behavior policy's log probability."""

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        valid_actions: Sequence[int],
        old_log_probability: float,
        value: float,
    ) -> None:
        self.state = np.asarray(state, dtype=np.float32).copy()
        self.action = int(action)
        self.reward = float(reward)
        self.next_state = np.asarray(next_state, dtype=np.float32).copy()
        self.terminated = bool(terminated)
        self.valid_actions = tuple(int(item) for item in valid_actions)
        self.old_log_probability = float(old_log_probability)
        self.value = float(value)


class ActorCriticNetwork(nn.Module):
    """Shared multilayer perceptron with policy-logit and value heads."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(states)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class PPOAgent:
    """PPO agent that samples or selects only actions allowed by the environment."""

    def __init__(
        self,
        config: PPOConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.network = ActorCriticNetwork(
            self.config.state_size,
            self.config.action_size,
            self.config.hidden_size,
        ).to(self.device)
        self.optimizer = Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.training_updates = 0

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Return action, log probability and critic value using the valid-action mask."""
        valid_action_indices = self._validate_valid_actions(valid_actions)
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.config.state_size,):
            raise ValueError(
                f"Expected a state with shape ({self.config.state_size},), "
                f"received {state_array.shape}."
            )

        state_tensor = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action_mask = torch.zeros(
            (1, self.config.action_size), dtype=torch.bool, device=self.device
        )
        action_mask[0, list(valid_action_indices)] = True

        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            distribution = self._masked_distribution(logits, action_mask)
            if deterministic:
                action = torch.argmax(distribution.logits, dim=-1)
            else:
                action = distribution.sample()
            log_probability = distribution.log_prob(action)
        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    def update(self, rollout: Sequence[PPOTransition]) -> dict[str, float]:
        """Update policy and critic using a completed on-policy rollout."""
        if not rollout:
            raise ValueError("Cannot update PPO with an empty rollout.")

        tensors = self._rollout_to_tensors(rollout)
        advantages, returns = self._compute_gae(
            tensors["rewards"], tensors["old_values"],
            tensors["next_values"], tensors["terminated"],
        )
        total_policy_loss, total_value_loss, total_entropy, n = self._run_update_epochs(
            tensors["states"], tensors["actions"], tensors["old_log_probabilities"],
            returns, advantages, tensors["action_masks"],
        )
        self.training_updates += 1
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
        }

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        **metadata: object,
    ) -> None:
        """Save policy, optimizer, configuration and training metadata."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> tuple[PPOAgent, dict[str, object]]:
        """Load an agent checkpoint and its training metadata."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"PPO checkpoint not found: {path}")
        checkpoint = torch.load(
            path,
            map_location=device or "cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or not {
            "model_state_dict",
            "config",
            "metadata",
        }.issubset(checkpoint):
            raise ValueError(f"Invalid PPO checkpoint: {path}")

        agent = cls(config=PPOConfig(**checkpoint["config"]), device=device)
        agent.network.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid PPO checkpoint metadata: {path}")
        return agent, metadata

    def _rollout_to_tensors(
        self, rollout: Sequence[PPOTransition]
    ) -> dict[str, torch.Tensor]:
        states = torch.as_tensor(
            np.stack([item.state for item in rollout]),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in rollout], dtype=torch.long, device=self.device,
        )
        old_log_probabilities = torch.as_tensor(
            [item.old_log_probability for item in rollout],
            dtype=torch.float32, device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in rollout], dtype=torch.float32, device=self.device,
        )
        old_values = torch.as_tensor(
            [item.value for item in rollout], dtype=torch.float32, device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in rollout]),
            dtype=torch.float32, device=self.device,
        )
        terminated = torch.as_tensor(
            [item.terminated for item in rollout], dtype=torch.bool, device=self.device,
        )
        action_masks = torch.zeros(
            (len(rollout), self.config.action_size), dtype=torch.bool, device=self.device,
        )
        for row, item in enumerate(rollout):
            valid_actions = self._validate_valid_actions(item.valid_actions)
            action_masks[row, list(valid_actions)] = True

        with torch.no_grad():
            _, next_values = self.network(next_states)
            next_values = next_values.masked_fill(terminated, 0.0)

        return {
            "states": states,
            "actions": actions,
            "old_log_probabilities": old_log_probabilities,
            "rewards": rewards,
            "old_values": old_values,
            "next_values": next_values,
            "terminated": terminated,
            "action_masks": action_masks,
        }

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        old_values: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            deltas = rewards + self.config.gamma * next_values - old_values
            advantages = torch.zeros_like(rewards)
            last_advantage = 0.0
            for index in reversed(range(len(rewards))):
                continuation = 0.0 if terminated[index] else 1.0
                last_advantage = (
                    deltas[index]
                    + self.config.gamma
                    * self.config.gae_lambda
                    * continuation
                    * last_advantage
                )
                advantages[index] = last_advantage
            returns = advantages + old_values
            if len(rewards) > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )
        return advantages, returns

    def _run_update_epochs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[float, float, float, int]:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        minibatch_count = 0
        self.network.train()

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(len(states), device=self.device)
            for start in range(0, len(states), self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                logits, predicted_values = self.network(states[indices])
                distribution = self._masked_distribution(logits, action_masks[indices])
                new_log_probabilities = distribution.log_prob(actions[indices])
                entropy = distribution.entropy().mean()

                probability_ratios = torch.exp(
                    new_log_probabilities - old_log_probabilities[indices]
                )
                unclipped_objective = probability_ratios * advantages[indices]
                clipped_objective = torch.clamp(
                    probability_ratios,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages[indices]
                policy_loss = -torch.minimum(
                    unclipped_objective, clipped_objective
                ).mean()
                value_loss = torch.nn.functional.mse_loss(
                    predicted_values, returns[indices]
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.gradient_clip_value,
                )
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                minibatch_count += 1

        return total_policy_loss, total_value_loss, total_entropy, minibatch_count

    def _validate_valid_actions(
        self, valid_actions: Sequence[int]
    ) -> tuple[int, ...]:
        action_indices = tuple(dict.fromkeys(int(action) for action in valid_actions))
        if not action_indices:
            raise ValueError("valid_actions must contain at least one action.")
        if any(
            action < 0 or action >= self.config.action_size
            for action in action_indices
        ):
            raise ValueError("valid_actions contains an action outside the action space.")
        return action_indices

    @staticmethod
    def _masked_distribution(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Categorical:
        if not action_mask.any(dim=-1).all():
            raise ValueError("Every policy row must have at least one valid action.")
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits)
