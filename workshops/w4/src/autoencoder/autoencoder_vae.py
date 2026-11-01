"""VAE model, configuration, and ELBO loss for W4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from autoencoder.autoencoder_ae import Decoder


@dataclass(frozen=True)
class VAEConfig:
    """All hyperparameters that fully describe one VAE experiment."""
    numeric_dim:       int
    cat_cardinalities: tuple[int, ...]
    hidden_dims:       tuple[int, ...] = (64, 32)
    z_dim:             int   = 16
    dropout:           float = 0.0
    beta:              float = 1.0
    kl_anneal_epochs:  int   = 10
    logvar_clamp:      tuple[float, float] = (-10.0, 10.0)
    learning_rate:     float = 1e-3
    weight_decay:      float = 0.0
    batch_size:        int   = 512
    max_epochs:        int   = 50
    patience:          int   = 5
    cat_loss_weight:   float = 1.0
    seed:              int   = 110


class VAEEncoder(nn.Module):
    """Dense encoder: concat(x_num, one_hot(x_cat)) → trunk → mu_head + logvar_head."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self._cat_cardinalities = config.cat_cardinalities
        self._logvar_clamp = config.logvar_clamp
        input_dim = config.numeric_dim + sum(config.cat_cardinalities)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mu_head     = nn.Linear(prev, config.z_dim)
        self.logvar_head = nn.Linear(prev, config.z_dim)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> tuple[Tensor, Tensor]:
        cat_feats = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=card).float()
             for i, card in enumerate(self._cat_cardinalities)],
            dim=-1,
        )
        h = self.trunk(torch.cat([x_num, cat_feats], dim=-1))
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(*self._logvar_clamp)
        return mu, logvar


class VAE(nn.Module):
    """VAE: VAEEncoder + shared Decoder, reparameterization, and generative sampling."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config  = config
        self.encoder = VAEEncoder(config)
        self.decoder = Decoder(config)  # type: ignore[arg-type]  # duck-typed

    def encode(self, x_num: Tensor, x_cat: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(x_num, x_cat)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def decode(self, z: Tensor) -> tuple[Tensor, list[Tensor]]:
        return self.decoder(z)

    def forward(
        self, x_num: Tensor, x_cat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, list[Tensor]]:
        """Return (z, mu, logvar, num_out, cat_logits).

        In train mode z is sampled; in eval mode z = mu (deterministic).
        """
        mu, logvar = self.encode(x_num, x_cat)
        z = self.reparameterize(mu, logvar)
        num_out, cat_logits = self.decode(z)
        return z, mu, logvar, num_out, cat_logits

    def sample(self, n: int, device: torch.device | str) -> tuple[Tensor, list[Tensor]]:
        """Decode z ~ N(0, I); returns (num_out, cat_logits)."""
        z = torch.randn(n, self.config.z_dim, device=device)
        with torch.no_grad():
            return self.decode(z)


class ELBOLoss(nn.Module):
    """ELBO = recon + beta * kl.

    Reconstruction: SUM over features per sample, then mean over batch.
      - numeric: sum of squared errors (55 dims)
      - categorical: sum of cross-entropies (3 heads)
    KL: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)), mean over batch.
    mse_per_element is the per-element mean (for AE comparison).
    """

    def forward(
        self,
        num_out:    Tensor,
        x_num:      Tensor,
        cat_logits: list[Tensor],
        x_cat:      Tensor,
        mu:         Tensor,
        logvar:     Tensor,
        beta:       float,
        reduction:  str = "mean",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (total, recon, kl, mse_per_element).

        When reduction='none', recon and kl are per-sample vectors.
        """
        if reduction == "none":
            return self._per_sample(num_out, x_num, cat_logits, x_cat, mu, logvar, beta)
        return self._aggregate(num_out, x_num, cat_logits, x_cat, mu, logvar, beta)

    # ------------------------------------------------------------------

    def _aggregate(
        self,
        num_out: Tensor, x_num: Tensor,
        cat_logits: list[Tensor], x_cat: Tensor,
        mu: Tensor, logvar: Tensor, beta: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Recon: sum over feature dims, mean over batch
        num_recon = F.mse_loss(num_out, x_num, reduction="none").sum(dim=-1).mean()
        cat_recon = torch.stack(
            [F.cross_entropy(logits, x_cat[:, i], reduction="none")
             for i, logits in enumerate(cat_logits)],
            dim=1,
        ).sum(dim=1).mean()
        recon = num_recon + cat_recon

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        mse_per_element = F.mse_loss(num_out, x_num)

        total = recon + beta * kl
        return total, recon, kl, mse_per_element

    def _per_sample(
        self,
        num_out: Tensor, x_num: Tensor,
        cat_logits: list[Tensor], x_cat: Tensor,
        mu: Tensor, logvar: Tensor, beta: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        num_recon_i = F.mse_loss(num_out, x_num, reduction="none").sum(dim=-1)
        cat_recon_i = torch.stack(
            [F.cross_entropy(logits, x_cat[:, i], reduction="none")
             for i, logits in enumerate(cat_logits)],
            dim=1,
        ).sum(dim=1)
        recon_i = num_recon_i + cat_recon_i

        kl_i = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)

        mse_per_element = F.mse_loss(num_out, x_num)

        total_i = recon_i + beta * kl_i
        return total_i, recon_i, kl_i, mse_per_element
