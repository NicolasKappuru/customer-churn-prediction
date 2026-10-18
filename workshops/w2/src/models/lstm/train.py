"""Training and prediction loops for the LSTM."""

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import (
    BATCH_SIZE,
    EPOCHS,
    GRAD_CLIP_NORM,
    INFER_BATCH_SIZE,
    PATIENCE,
    SEED,
)


def resolve_device(device=None):
    """Return the requested device or the best available one."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _training_setup(model, lr, weight_decay, device, amp):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return criterion, optimizer, scaler, use_amp


def _batch_loss(model, x, y, criterion, device, amp):
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp,
    ):
        logits, _, _ = model(x)
        return criterion(logits, y)


def _optimizer_step(loss, model, optimizer, scaler, amp):
    if amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        return

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()


def _train_one_epoch(
    model,
    data,
    rows,
    phase,
    batch_size,
    optimizer,
    criterion,
    device,
    scaler,
    amp,
    seed,
):
    """Train one epoch and return the sample-weighted average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for x, y in data.iter_batches(
        rows,
        phase,
        batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
    ):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, x, y, criterion, device, amp)
        _optimizer_step(loss, model, optimizer, scaler, amp)
        total_loss += loss.item() * len(x)
        n_samples += len(x)

    if n_samples == 0:
        raise ValueError("No training batches were produced.")
    return total_loss / n_samples


def _evaluate_loss(model, data, rows, phase, batch_size, criterion, device, amp):
    """Return the sample-weighted average loss without gradient updates."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for x, y in data.iter_batches(
            rows,
            phase,
            batch_size,
            shuffle=False,
            drop_last=False,
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = _batch_loss(model, x, y, criterion, device, amp)
            total_loss += loss.item() * len(x)
            n_samples += len(x)

    return total_loss / max(n_samples, 1)


@dataclass
class _Trainer:
    """Bundles the objects needed by one training epoch."""

    model: nn.Module
    data: object
    phase: str
    batch_size: int
    criterion: nn.Module
    optimizer: object
    scaler: object
    device: torch.device
    amp: bool

    def train_epoch(self, rows, seed):
        return _train_one_epoch(
            self.model,
            self.data,
            rows,
            self.phase,
            self.batch_size,
            self.optimizer,
            self.criterion,
            self.device,
            self.scaler,
            self.amp,
            seed,
        )

    def evaluate(self, rows):
        return _evaluate_loss(
            self.model,
            self.data,
            rows,
            self.phase,
            self.batch_size,
            self.criterion,
            self.device,
            self.amp,
        )


def _make_trainer(model, data, phase, batch_size, lr, weight_decay, device, amp):
    criterion, optimizer, scaler, use_amp = _training_setup(
        model, lr, weight_decay, device, amp
    )
    return _Trainer(
        model=model,
        data=data,
        phase=phase,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        amp=use_amp,
    )


@dataclass
class _EarlyStopping:
    """Track the best stop loss and the weights that produced it."""

    patience: object
    best_loss: float = float("inf")
    best_epoch: int = 0
    stale_epochs: int = 0
    best_state: object = None

    def update(self, epoch, stop_loss, model):
        if stop_loss < self.best_loss - 1e-6:
            self.best_loss = stop_loss
            self.best_epoch = epoch
            self.stale_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.stale_epochs += 1
        return self.patience is not None and self.stale_epochs >= self.patience


def _log_epoch(epoch, train_loss, stop_loss, verbose, log_prefix=""):
    if verbose:
        print(
            f"{log_prefix}epoch {epoch:02d} | train={train_loss:.5f}"
            f" | stop={stop_loss:.5f}",
            flush=True,
        )


def _training_result(model, history, best_state, best_epoch, best_loss):
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_stop_loss": best_loss,
        "best_state": best_state,
    }


@torch.no_grad()
def predict_proba(
    model,
    data,
    rows,
    phase="fit",
    batch_size=INFER_BATCH_SIZE,
    device=None,
):
    """Return churn probabilities for ``rows`` in their original order."""
    device = resolve_device(device)
    model.eval()
    probabilities = []

    for x, _ in data.iter_batches(
        rows,
        phase,
        batch_size,
        shuffle=False,
        drop_last=False,
    ):
        x = x.to(device, non_blocking=True)
        logits, _, _ = model(x)
        probabilities.append(
            torch.sigmoid(logits).float().cpu().numpy().ravel()
        )

    return np.concatenate(probabilities) if probabilities else np.empty(0)


def _fit_with_early_stopping(
    trainer,
    train_rows,
    stop_rows,
    epochs,
    patience,
    seed,
    verbose,
    log_prefix="",
):
    stopper = _EarlyStopping(patience)
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_rows, seed + epoch)
        stop_loss = trainer.evaluate(stop_rows)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "stop_loss": stop_loss}
        )
        _log_epoch(epoch, train_loss, stop_loss, verbose, log_prefix)

        if stopper.update(epoch, stop_loss, trainer.model):
            if verbose:
                print(
                    f"{log_prefix}early stop at epoch {epoch} "
                    f"(best epoch {stopper.best_epoch})",
                    flush=True,
                )
            break

    return history, stopper


def train_model(
    model,
    data,
    train_rows,
    stop_rows,
    *,
    epochs=EPOCHS,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=BATCH_SIZE,
    patience=PATIENCE,
    device=None,
    phase="fit",
    seed=SEED,
    amp=True,
    verbose=False,
    log_prefix="",
):
    """Train with AdamW, early-stop on ``stop_rows`` and restore the best epoch."""
    device = resolve_device(device)
    set_seed(seed)
    model.to(device)
    trainer = _make_trainer(
        model, data, phase, batch_size, lr, weight_decay, device, amp
    )

    history, stopper = _fit_with_early_stopping(
        trainer,
        train_rows,
        stop_rows,
        epochs,
        patience,
        seed,
        verbose,
        log_prefix,
    )
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return _training_result(
        model,
        history,
        stopper.best_state,
        stopper.best_epoch,
        stopper.best_loss,
    )


def fit_fixed_epochs(
    model,
    data,
    rows,
    *,
    epochs,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=BATCH_SIZE,
    device=None,
    phase="development",
    seed=SEED,
    amp=True,
    verbose=False,
    log_prefix="",
):
    """Train for a fixed number of epochs without validation or early stopping."""
    device = resolve_device(device)
    set_seed(seed)
    model.to(device)
    trainer = _make_trainer(
        model, data, phase, batch_size, lr, weight_decay, device, amp
    )
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(rows, seed + epoch)
        history.append({"epoch": epoch, "train_loss": train_loss})
        if verbose:
            print(
                f"{log_prefix}epoch {epoch:02d} | train={train_loss:.5f}",
                flush=True,
            )

    return {"model": model, "history": history}
