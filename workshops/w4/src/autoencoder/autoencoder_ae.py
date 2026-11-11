"""Autoencoder model, configuration, and reconstruction loss for W4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor


@dataclass(frozen=True)
class AEConfig:
    """All hyperparameters that fully describe one AE experiment."""
    numeric_dim:      int
    cat_cardinalities: tuple[int, ...]
    hidden_dims:      tuple[int, ...] = (64, 32)
    z_dim:            int   = 16
    dropout:          float = 0.0
    learning_rate:    float = 1e-3
    weight_decay:     float = 0.0
    batch_size:       int   = 512
    max_epochs:       int   = 50
    patience:         int   = 5
    cat_loss_weight:  float = 1.0
    seed:             int   = 110


class Encoder(nn.Module):
    """Dense encoder: concat(x_num, one_hot(x_cat)) → hidden layers → z."""

    def __init__(self, config: AEConfig) -> None:
        super().__init__()
        self._cat_cardinalities = config.cat_cardinalities
        input_dim = config.numeric_dim + sum(config.cat_cardinalities)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            prev = h
        layers.append(nn.Linear(prev, config.z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        cat_feats = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=card).float()
             for i, card in enumerate(self._cat_cardinalities)],
            dim=-1,
        )
        return self.net(torch.cat([x_num, cat_feats], dim=-1))


class Decoder(nn.Module):
    """Mirror decoder: z → reversed hidden layers → numeric head + categorical heads."""

    def __init__(self, config: AEConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = config.z_dim
        for h in reversed(config.hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.numeric_head = nn.Linear(prev, config.numeric_dim)  # linear activation
        self.cat_heads = nn.ModuleList(
            [nn.Linear(prev, card) for card in config.cat_cardinalities]
        )

    def forward(self, z: Tensor) -> tuple[Tensor, list[Tensor]]:
        h = self.trunk(z)
        num_out    = self.numeric_head(h)
        cat_logits = [head(h) for head in self.cat_heads]
        return num_out, cat_logits


class Autoencoder(nn.Module):
    """Encoder + Decoder with encode / decode / forward convenience methods."""

    def __init__(self, config: AEConfig) -> None:
        super().__init__()
        self.config  = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.encoder(x_num, x_cat)

    def decode(self, z: Tensor) -> tuple[Tensor, list[Tensor]]:
        return self.decoder(z)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        z = self.encode(x_num, x_cat)
        num_out, cat_logits = self.decode(z)
        return z, num_out, cat_logits


class ReconstructionLoss(nn.Module):
    """MSE (numeric) + weighted mean CrossEntropy (categorical).

    Returns (total, num_component, cat_component).
    When per_sample=True, each component has shape (batch_size,) instead of scalar.
    """

    def __init__(self, cat_loss_weight: float = 1.0) -> None:
        super().__init__()
        self.cat_loss_weight = cat_loss_weight

    def forward(
        self,
        num_out:    Tensor,
        x_num:      Tensor,
        cat_logits: list[Tensor],
        x_cat:      Tensor,
        per_sample: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if per_sample:
            return self._per_sample(num_out, x_num, cat_logits, x_cat)
        return self._aggregate(num_out, x_num, cat_logits, x_cat)

    # ------------------------------------------------------------------

    def _aggregate(
        self,
        num_out: Tensor,
        x_num:   Tensor,
        cat_logits: list[Tensor],
        x_cat:   Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        num_loss = F.mse_loss(num_out, x_num)
        cat_loss = torch.stack(
            [F.cross_entropy(logits, x_cat[:, i]) for i, logits in enumerate(cat_logits)]
        ).mean()
        total = num_loss + self.cat_loss_weight * cat_loss
        return total, num_loss, cat_loss

    def _per_sample(
        self,
        num_out: Tensor,
        x_num:   Tensor,
        cat_logits: list[Tensor],
        x_cat:   Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        num_ps = F.mse_loss(num_out, x_num, reduction="none").mean(dim=-1)
        cat_ps = torch.stack(
            [F.cross_entropy(logits, x_cat[:, i], reduction="none")
             for i, logits in enumerate(cat_logits)]
        ).mean(dim=0)
        total_ps = num_ps + self.cat_loss_weight * cat_ps
        return total_ps, num_ps, cat_ps
