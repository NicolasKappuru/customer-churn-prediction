"""AE training loop, early stopping, and checkpoint utilities — Telco edition."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT  = Path(__file__).resolve().parents[3]
W4_TELCO_ROOT = PROJECT_ROOT / "src" / "w4_autoencoders"

for _p in (str(W4_TELCO_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoencoder.autoencoder_ae import AEConfig, Autoencoder, ReconstructionLoss
from autoencoder.preprocessor   import TabularPreprocessor
from constants import RUNS_DIR, CHECKPOINTS_DIR, SEED


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(x_num, x_cat)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience   = patience
        self.best_loss  = float("inf")
        self.best_epoch = 0
        self._strikes   = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self._strikes   = 0
            return False
        self._strikes += 1
        return self._strikes >= self.patience


class AETrainer:
    """Orchestrates training, validation, early stopping, and history logging."""

    def __init__(
        self,
        model:  Autoencoder,
        config: AEConfig,
        device: torch.device | str,
    ) -> None:
        self.model  = model
        self.config = config
        self.device = torch.device(device)
        self._loss_fn   = ReconstructionLoss(config.cat_loss_weight)
        self._optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._best_state: dict | None = None
        self.best_epoch = 0

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        early = EarlyStopping(self.config.patience)
        rows: list[dict] = []

        for epoch in range(1, self.config.max_epochs + 1):
            tr_total, tr_num, tr_cat = self._train_epoch(train_loader)
            va_total, va_num, va_cat = self._validate_epoch(val_loader)

            rows.append({
                "epoch":      epoch,
                "train_loss": tr_total,
                "train_num":  tr_num,
                "train_cat":  tr_cat,
                "val_loss":   va_total,
                "val_num":    va_num,
                "val_cat":    va_cat,
            })

            if va_total < early.best_loss:
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if early.step(va_total, epoch):
                print(f"  Early stop at epoch {epoch} (best={early.best_epoch}, val={early.best_loss:.6f})")
                break

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:3d}  train={tr_total:.5f}  val={va_total:.5f}"
                    f"  (num={va_num:.5f}  cat={va_cat:.5f})"
                )

        self.best_epoch = early.best_epoch
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return pd.DataFrame(rows)

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.train()
        return self._run_epoch(loader, training=True)

    def _validate_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(loader, training=False)

    def _run_epoch(self, loader: DataLoader, training: bool) -> tuple[float, float, float]:
        total_loss = num_loss = cat_loss = 0.0
        n_batches = 0
        for batch in loader:
            t, n, c = self._step(batch, training=training)
            total_loss += t
            num_loss   += n
            cat_loss   += c
            n_batches  += 1
        denom = max(n_batches, 1)
        return total_loss / denom, num_loss / denom, cat_loss / denom

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        training: bool,
    ) -> tuple[float, float, float]:
        x_num, x_cat = batch
        x_num = x_num.to(self.device)
        x_cat = x_cat.to(self.device)

        _, num_out, cat_logits = self.model(x_num, x_cat)
        total, num_l, cat_l = self._loss_fn(num_out, x_num, cat_logits, x_cat)

        if training:
            self._optimizer.zero_grad(set_to_none=True)
            total.backward()
            self._optimizer.step()

        return float(total.item()), float(num_l.item()), float(cat_l.item())


def save_checkpoint(
    path: Path | str,
    model:        Autoencoder,
    config:       AEConfig,
    preprocessor: TabularPreprocessor,
    history:      pd.DataFrame,
    best_epoch:   int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(config)
    cfg["hidden_dims"]       = list(cfg["hidden_dims"])
    cfg["cat_cardinalities"] = list(cfg["cat_cardinalities"])
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
    path: Path | str,
    device: torch.device | str | None = None,
) -> tuple[Autoencoder, AEConfig, TabularPreprocessor, pd.DataFrame, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)

    cfg = ckpt["config"]
    cfg["hidden_dims"]       = tuple(cfg["hidden_dims"])
    cfg["cat_cardinalities"] = tuple(cfg["cat_cardinalities"])
    config = AEConfig(**cfg)

    model = Autoencoder(config)
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
    parser = argparse.ArgumentParser(description="Train telco AE with a given z_dim.")
    parser.add_argument("--z-dim",      type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()

    from autoencoder.data_loader import load_telco_data, filter_normal

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    df, y_all, _, categorical, numeric, split = load_telco_data()
    fit_normal, stop_normal = filter_normal(split, y_all)

    prep = TabularPreprocessor(numeric, categorical)
    prep.fit(df.iloc[fit_normal])

    x_num_fit,  x_cat_fit  = prep.transform(df.iloc[fit_normal])
    x_num_stop, x_cat_stop = prep.transform(df.iloc[stop_normal])

    config = AEConfig(
        numeric_dim=prep.numeric_dim,
        cat_cardinalities=prep.cat_cardinalities,
        z_dim=args.z_dim,
        max_epochs=args.max_epochs,
        seed=SEED,
    )
    model   = Autoencoder(config).to(device)
    trainer = AETrainer(model, config, device)
    history = trainer.fit(
        build_loader(x_num_fit,  x_cat_fit,  config.batch_size, shuffle=True),
        build_loader(x_num_stop, x_cat_stop, config.batch_size, shuffle=False),
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(RUNS_DIR / f"ae_z{args.z_dim}_history.csv", index=False)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        CHECKPOINTS_DIR / f"ae_z{args.z_dim}.pt",
        model, config, prep, history, trainer.best_epoch,
    )
    print(f"Done. Checkpoint saved to checkpoints/ae_z{args.z_dim}.pt")
