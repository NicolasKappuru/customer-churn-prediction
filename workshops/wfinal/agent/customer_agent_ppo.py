"""Customer-side PPO agent for the KKBox retention environment."""

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
W3_ROOT = PROJECT_ROOT / "workshops" / "wfinal"
for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from constants import CUSTOMER_OBS_SIZE, CUSTOMER_N_ACTIONS

# Index constants for two-branch split
_CUSTOMER_FEAT_END = 59
_OFFER_FEAT_START  = 59

# Irrationality floor: bounds pi(accept) to [IRRATIONALITY/2, 1 - IRRATIONALITY/2]
IRRATIONALITY = 0.10


class FeatureNormalizer:
    """Z-score normalizer fitted on the 59-feature customer block."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        self.mean_ = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            return x
        return np.clip((x - self.mean_) / self.std_, -5.0, 5.0)

    def state_dict(self) -> dict:
        return {"mean_": self.mean_, "std_": self.std_}

    @classmethod
    def from_state_dict(cls, d: dict) -> "FeatureNormalizer":
        obj = cls()
        obj.mean_ = d["mean_"]
        obj.std_  = d["std_"]
        return obj


@dataclass(frozen=True)
class CustomerPPOConfig:
    """PPO hyperparameters for the customer agent."""

    state_size: int = CUSTOMER_OBS_SIZE
    action_size: int = CUSTOMER_N_ACTIONS
    hidden_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.9
    weight_decay: float = 0.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    entropy_coefficient: float = 0.01   # kept for backward compat
    value_coefficient: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 4
    gradient_clip_value: float = 0.5


class CustomerTransition:
    """One on-policy transition for the customer agent."""

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        old_log_probability: float,
        value: float,
    ) -> None:
        self.state = np.asarray(state, dtype=np.float32).copy()
        self.action = int(action)
        self.reward = float(reward)
        self.next_state = np.asarray(next_state, dtype=np.float32).copy()
        self.terminated = bool(terminated)
        self.old_log_probability = float(old_log_probability)
        self.value = float(value)


class ActorCriticNetwork(nn.Module):
    """Two-branch actor-critic with separate customer-feature and offer+episode branches.

    Branch 1 (customer): input[:59]  -> Linear(64) -> ReLU -> Linear(32) -> ReLU
    Branch 2 (offer+ep): input[59:]  -> Linear(16) -> ReLU
    Concat(32+16=48)                 -> Linear(32) -> ReLU
    Actor head: 32 -> Linear(2)
    Critic head: 32 -> Linear(1)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        offer_feat_size = state_size - _CUSTOMER_FEAT_END   # 74 - 59 = 15

        self.customer_branch = nn.Sequential(
            nn.Linear(_CUSTOMER_FEAT_END, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.offer_branch = nn.Sequential(
            nn.Linear(offer_feat_size, 16),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(32, action_size)
        self.value_head  = nn.Linear(32, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cust = self.customer_branch(states[:, :_CUSTOMER_FEAT_END])
        offer = self.offer_branch(states[:, _OFFER_FEAT_START:])
        combined = self.shared(torch.cat([cust, offer], dim=-1))
        return self.policy_head(combined), self.value_head(combined).squeeze(-1)


class CustomerPPOAgent:
    """PPO agent for the customer role (binary accept/reject decision)."""

    def __init__(
        self,
        config: CustomerPPOConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or CustomerPPOConfig()
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
        self.normalizer: FeatureNormalizer | None = None
        self.training_updates = 0

    def _apply_normalizer(self, state_array: np.ndarray) -> np.ndarray:
        """Normalize the first 59 features; leave offer/episode blocks untouched."""
        if self.normalizer is None:
            return state_array
        normed = state_array.copy()
        normed[:_CUSTOMER_FEAT_END] = self.normalizer.transform(
            state_array[:_CUSTOMER_FEAT_END]
        )
        return normed

    def _bounded_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply irrationality floor/ceiling: pi in [IRRATIONALITY/2, 1-IRRATIONALITY/2]."""
        raw = torch.softmax(logits, dim=-1)
        return IRRATIONALITY / 2.0 + (1.0 - IRRATIONALITY) * raw

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Return (action, log_probability, value). No valid_actions mask needed."""
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.config.state_size,):
            raise ValueError(
                f"Expected a state with shape ({self.config.state_size},), "
                f"received {state_array.shape}."
            )
        state_array = self._apply_normalizer(state_array)
        state_tensor = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            probs = self._bounded_probs(logits)
            distribution = Categorical(probs=probs)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = distribution.sample()
            log_probability = distribution.log_prob(action)
        return (
            int(action.item()),
            float(log_probability.item()),
            float(value.item()),
        )

    def accept_probability(self, state: np.ndarray) -> float:
        """Return P(action=1 | state), i.e. probability of accepting the offer."""
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.config.state_size,):
            raise ValueError(
                f"Expected a state with shape ({self.config.state_size},), "
                f"received {state_array.shape}."
            )
        state_array = self._apply_normalizer(state_array)
        state_tensor = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            logits, _ = self.network(state_tensor)
            probs = self._bounded_probs(logits)
        return float(probs[0, 1].item())

    def update(self, rollout: Sequence[CustomerTransition]) -> dict[str, float]:
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
            returns, advantages,
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
        """Save policy, optimizer, normalizer, configuration and training metadata."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "metadata": metadata,
                "normalizer": self.normalizer.state_dict() if self.normalizer is not None else None,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> tuple[CustomerPPOAgent, dict[str, object]]:
        """Load a customer agent checkpoint and its training metadata."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Customer PPO checkpoint not found: {path}")
        checkpoint = torch.load(
            path,
            map_location=device or "cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or not {
            "model_state_dict",
            "config",
            "metadata",
        }.issubset(checkpoint):
            raise ValueError(f"Invalid customer PPO checkpoint: {path}")

        cfg_dict = dict(checkpoint["config"])
        # Back-compat: old checkpoints only have entropy_coefficient; new ones have entropy_coef
        if "entropy_coef" not in cfg_dict and "entropy_coefficient" in cfg_dict:
            cfg_dict["entropy_coef"] = cfg_dict["entropy_coefficient"]
        agent = cls(config=CustomerPPOConfig(**cfg_dict), device=device)
        agent.network.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("normalizer") is not None:
            agent.normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid customer PPO checkpoint metadata: {path}")
        return agent, metadata

    def _rollout_to_tensors(
        self, rollout: Sequence[CustomerTransition]
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
                probs = self._bounded_probs(logits)
                distribution = Categorical(probs=probs)
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
                ent_coef = self.config.entropy_coef
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - ent_coef * entropy
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
