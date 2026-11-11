"""VAE training loop, KL annealing, and checkpoint utilities."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
W4_SRC = PROJECT_ROOT / "workshops" / "w4" / "src"

for _p in (str(W4_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoencoder.autoencoder_vae import VAE, VAEConfig, ELBOLoss
from autoencoder.preprocessor import TabularPreprocessor
from constants import RUNS_DIR, CHECKPOINTS_DIR, SEED
from train.train_ae import EarlyStopping, build_loader, set_seed


class KLAnnealer:
    """Linear KL warm-up: beta(epoch) = target * min(1, epoch / warmup_epochs)."""

    def __init__(self, target_beta: float, warmup_epochs: int) -> None:
        self.target_beta   = target_beta
        self.warmup_epochs = warmup_epochs

    def beta(self, epoch: int) -> float:
        return self.target_beta * min(1.0, epoch / max(self.warmup_epochs, 1))


def active_units(mu_val: torch.Tensor, threshold: float = 0.01) -> int:
    """Count latent dimensions with Var(mu) > threshold (posterior collapse detector)."""
    return int((mu_val.var(dim=0) > threshold).sum().item())


class VAETrainer:
    """Orchestrates VAE training, validation, KL annealing, and history logging."""

    def __init__(
        self,
        model:  VAE,
        config: VAEConfig,
        device: torch.device | str,
    ) -> None:
        self.model  = model
        self.config = config
        self.device = torch.device(device)
        self._loss_fn   = ELBOLoss()
        self._optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._annealer  = KLAnnealer(config.beta, config.kl_anneal_epochs)
        self._best_state: dict | None = None
        self.best_epoch = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        """Train the model; return a history DataFrame."""
        early = EarlyStopping(self.config.patience)
        rows: list[dict] = []

        for epoch in range(1, self.config.max_epochs + 1):
            beta_t = self._annealer.beta(epoch)
            tr_total, tr_recon, tr_kl = self._train_epoch(train_loader, beta_t)
            va_total, va_recon, va_kl, va_mse, va_mu = self._validate_epoch(val_loader)
            au = active_units(va_mu)

            rows.append({
                "epoch":             epoch,
                "beta_t":            beta_t,
                "train_loss":        tr_total,
                "train_recon":       tr_recon,
                "train_kl":          tr_kl,
                "val_loss":          va_total,
                "val_recon":         va_recon,
                "val_kl":            va_kl,
                "val_mse_per_element": va_mse,
                "active_units":      au,
            })

            # Early stopping only counts after KL warm-up
            past_warmup = epoch > self.config.kl_anneal_epochs
            if past_warmup and va_total < early.best_loss:
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if past_warmup and early.step(va_total, epoch):
                print(
                    f"  Early stop at epoch {epoch} "
                    f"(best={early.best_epoch}, val={early.best_loss:.5f})"
                )
                break

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:3d}  beta_t={beta_t:.3f}  "
                    f"train={tr_total:.4f}  val={va_total:.4f}  "
                    f"(recon={va_recon:.4f}  kl={va_kl:.4f}  au={au})"
                )

        self.best_epoch = early.best_epoch
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, beta_t: float
    ) -> tuple[float, float, float]:
        self.model.train()
        total_loss = recon_loss = kl_loss = 0.0
        n = 0
        for batch in loader:
            t, r, k = self._step(batch, beta=beta_t, training=True)
            total_loss += t; recon_loss += r; kl_loss += k
            n += 1
        d = max(n, 1)
        return total_loss / d, recon_loss / d, kl_loss / d

    def _validate_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float, float, float, torch.Tensor]:
        """Validate with target beta; also collect mu for active_units."""
        self.model.eval()
        total_loss = recon_loss = kl_loss = mse_acc = 0.0
        mu_list: list[torch.Tensor] = []
        n = 0
        with torch.no_grad():
            for batch in loader:
                x_num, x_cat = batch
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                _, mu, logvar, num_out, cat_logits = self.model(x_num, x_cat)
                total, recon, kl, mse = self._loss_fn(
                    num_out, x_num, cat_logits, x_cat, mu, logvar,
                    beta=self.config.beta,
                )
                total_loss += float(total.item())
                recon_loss += float(recon.item())
                kl_loss    += float(kl.item())
                mse_acc    += float(mse.item())
                mu_list.append(mu.cpu())
                n += 1
        d = max(n, 1)
        mu_all = torch.cat(mu_list, dim=0)
        return total_loss / d, recon_loss / d, kl_loss / d, mse_acc / d, mu_all

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        beta: float,
        training: bool,
    ) -> tuple[float, float, float]:
        x_num, x_cat = batch
        x_num = x_num.to(self.device)
        x_cat = x_cat.to(self.device)

        _, mu, logvar, num_out, cat_logits = self.model(x_num, x_cat)
        total, recon, kl, _ = self._loss_fn(
            num_out, x_num, cat_logits, x_cat, mu, logvar, beta=beta,
        )

        if training:
            self._optimizer.zero_grad(set_to_none=True)
            total.backward()
            self._optimizer.step()

        return float(total.item()), float(recon.item()), float(kl.item())


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:         Path | str,
    model:        VAE,
    config:       VAEConfig,
    preprocessor: TabularPreprocessor,
    history:      pd.DataFrame,
    best_epoch:   int,
) -> None:
    """Save VAE model, config, preprocessor state, and training history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(config)
    cfg["hidden_dims"]       = list(cfg["hidden_dims"])
    cfg["cat_cardinalities"] = list(cfg["cat_cardinalities"])
    cfg["logvar_clamp"]      = list(cfg["logvar_clamp"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config":           cfg,
            "preprocessor":     preprocessor.state_dict(),
            "history":          history.to_dict("list"),
            "best_epoch":       best_epoch,
        },
        path,
    )


def load_checkpoint(
    path:   Path | str,
    device: torch.device | str | None = None,
) -> tuple[VAE, VAEConfig, TabularPreprocessor, pd.DataFrame, int]:
    """Restore VAE, config, preprocessor, history, and best_epoch from a checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)

    cfg = ckpt["config"]
    cfg["hidden_dims"]       = tuple(cfg["hidden_dims"])
    cfg["cat_cardinalities"] = tuple(cfg["cat_cardinalities"])
    cfg["logvar_clamp"]      = tuple(cfg["logvar_clamp"])
    config = VAEConfig(**cfg)

    model = VAE(config)
    model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model = model.to(device)

    preprocessor = TabularPreprocessor.from_state_dict(ckpt["preprocessor"])
    history      = pd.DataFrame(ckpt["history"])
    best_epoch   = int(ckpt["best_epoch"])

    return model, config, preprocessor, history, best_epoch


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE with a given beta.")
    parser.add_argument("--beta",       type=float, default=1.0, help="KL weight")
    parser.add_argument("--max-epochs", type=int,   default=50,  help="Maximum epochs")
    args = parser.parse_args()

    from autoencoder.data_loader import filter_normal, load_w1_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    df, y_all, _, categorical, numeric, split = load_w1_data()
    fit_normal, stop_normal = filter_normal(split, y_all)

    prep = TabularPreprocessor(numeric, categorical)
    prep.fit(df.iloc[fit_normal])

    x_num_fit,  x_cat_fit  = prep.transform(df.iloc[fit_normal])
    x_num_stop, x_cat_stop = prep.transform(df.iloc[stop_normal])

    config = VAEConfig(
        numeric_dim=prep.numeric_dim,
        cat_cardinalities=prep.cat_cardinalities,
        beta=args.beta,
        max_epochs=args.max_epochs,
        seed=SEED,
    )
    model   = VAE(config).to(device)
    trainer = VAETrainer(model, config, device)
    history = trainer.fit(
        build_loader(x_num_fit,  x_cat_fit,  config.batch_size, shuffle=True),
        build_loader(x_num_stop, x_cat_stop, config.batch_size, shuffle=False),
    )

    beta_str = str(args.beta).replace(".", "_")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(RUNS_DIR / f"vae_beta{args.beta}_history.csv", index=False)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        CHECKPOINTS_DIR / f"vae_beta{args.beta}.pt",
        model, config, prep, history, trainer.best_epoch,
    )
    print(f"Done. Checkpoint saved to checkpoints/vae_beta{args.beta}.pt")
