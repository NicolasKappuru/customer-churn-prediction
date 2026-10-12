"""Deep Q-Network agent with valid-action masking and replay memory."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, NamedTuple, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters that can be tuned from the training script."""

    state_size: int = 48
    action_size: int = 15
    hidden_size: int = 128
    learning_rate: float = 5e-4
    gamma: float = 0.95
    batch_size: int = 64
    replay_capacity: int = 50_000
    target_update_interval: int = 100
    gradient_clip_value: float = 10.0


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    next_valid_actions: tuple[int, ...]


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay-buffer capacity must be positive.")
        self._transitions: Deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        next_valid_actions: Sequence[int],
    ) -> None:
        self._transitions.append(
            Transition(
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                terminated=bool(terminated),
                next_valid_actions=tuple(int(item) for item in next_valid_actions),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._transitions):
            raise ValueError("Cannot sample more transitions than the buffer contains.")
        return random.sample(self._transitions, batch_size)

    def __len__(self) -> int:
        return len(self._transitions)


class QNetwork(nn.Module):
    """Small fully connected Q-value function approximator."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class DQNAgent:
    """DQN with epsilon-greedy choice restricted to currently valid actions."""

    def __init__(
        self,
        config: DQNConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.online_network = QNetwork(
            self.config.state_size,
            self.config.action_size,
            self.config.hidden_size,
        ).to(self.device)
        self.target_network = QNetwork(
            self.config.state_size,
            self.config.action_size,
            self.config.hidden_size,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(self.config.replay_capacity)
        self.training_steps = 0
        self.update_target_network()

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: Sequence[int],
        epsilon: float = 0.0,
    ) -> int:
        """Choose a random or greedy action from the supplied valid indices."""
        valid_action_indices = tuple(dict.fromkeys(int(a) for a in valid_actions))
        if not valid_action_indices:
            raise ValueError("valid_actions must contain at least one action.")
        if any(
            action < 0 or action >= self.config.action_size
            for action in valid_action_indices
        ):
            raise ValueError("valid_actions contains an action outside the action space.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.config.state_size,):
            raise ValueError(
                f"Expected a state with shape ({self.config.state_size},), "
                f"received {state_array.shape}."
            )

        if random.random() < epsilon:
            return random.choice(valid_action_indices)

        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state_array, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online_network(state_tensor).squeeze(0)
            valid_tensor = torch.as_tensor(
                valid_action_indices, dtype=torch.long, device=self.device
            )
            return int(valid_tensor[torch.argmax(q_values[valid_tensor])].item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        next_valid_actions: Sequence[int],
    ) -> None:
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            terminated,
            next_valid_actions,
        )

    def train_step(self) -> float | None:
        """Perform one Double-DQN update; return None until a batch is available."""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        terminated = torch.as_tensor(
            [item.terminated for item in batch],
            dtype=torch.bool,
            device=self.device,
        )

        predicted_q_values = self.online_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        targets = rewards.clone()
        nonterminal_indices = [
            index
            for index, item in enumerate(batch)
            if not item.terminated and item.next_valid_actions
        ]
        if nonterminal_indices:
            index_tensor = torch.as_tensor(
                nonterminal_indices, dtype=torch.long, device=self.device
            )
            next_valid_actions = [
                batch[index].next_valid_actions for index in nonterminal_indices
            ]
            with torch.no_grad():
                online_next_q = self.online_network(next_states[index_tensor])
                target_next_q = self.target_network(next_states[index_tensor])
                next_values = []
                for row, valid_actions in enumerate(next_valid_actions):
                    valid_tensor = torch.as_tensor(
                        valid_actions, dtype=torch.long, device=self.device
                    )
                    best_action = valid_tensor[
                        torch.argmax(online_next_q[row, valid_tensor])
                    ]
                    next_values.append(target_next_q[row, best_action])
                targets[index_tensor] += self.config.gamma * torch.stack(next_values)

        targets[terminated] = rewards[terminated]
        loss = self.loss_function(predicted_q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_value,
        )
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval == 0:
            self.update_target_network()
        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy online-network weights to the target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        **metadata: object,
    ) -> None:
        """Save model, optimizer, configuration, and optional training metadata."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.online_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
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
    ) -> tuple[DQNAgent, dict[str, object]]:
        """Restore an agent checkpoint and return its metadata."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"DQN checkpoint not found: {path}")
        checkpoint = torch.load(
            path,
            map_location=device or "cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or not {
            "model_state_dict",
            "target_state_dict",
            "config",
            "metadata",
        }.issubset(checkpoint):
            raise ValueError(f"Invalid DQN checkpoint: {path}")

        agent = cls(config=DQNConfig(**checkpoint["config"]), device=device)
        agent.online_network.load_state_dict(checkpoint["model_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_state_dict"])
        agent.target_network.eval()
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        metadata = checkpoint["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid DQN checkpoint metadata: {path}")
        return agent, metadata