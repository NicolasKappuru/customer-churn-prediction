"""Training and prediction loops for the trajectory models.

Same procedure as ``recurrent_models/train.py`` (AdamW, gradient clipping at
1.0, AMP on GPU, early stopping on ``stop`` with best-epoch restore); only the
criterion and the prediction output change.
"""

import copy
import random

import numpy as np
import torch

from .config import (
    BATCH_SIZE,
    EPOCHS,
    GRAD_CLIP_NORM,
    INFER_BATCH_SIZE,
    PATIENCE,
    SEED,
    T_ACTIVE,
    T_AMOUNT,
    T_LISTEN,
    T_PAY,
)
from .loss import TrajectoryLoss


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


class _Trainer:
    """Model, data, criterion and optimiser for one training run."""

    def __init__(self, model, data, phase, batch_size, lr, weight_decay,
                 loss_weights, device, amp):
        self.model = model
        self.data = data
        self.phase = phase
        self.batch_size = batch_size
        self.device = device
        self.criterion = TrajectoryLoss(loss_weights)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.amp = bool(amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _loss(self, x, y):
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.amp
        ):
            pred = self.model(x)
        return self.criterion(pred, y)  # computed in float32

    def _step(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

    def train_epoch(self, rows, seed):
        self.model.train()
        total, n = 0.0, 0
        for x, y in self.data.iter_batches(
            rows, self.phase, self.batch_size, shuffle=True, drop_last=True, seed=seed
        ):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._loss(x, y)
            self._step(loss)
            total += loss.item() * len(x)
            n += len(x)
        if n == 0:
            raise ValueError("No training batches were produced.")
        return total / n

    @torch.no_grad()
    def evaluate(self, rows):
        self.model.eval()
        total, n = 0.0, 0
        for x, y in self.data.iter_batches(rows, self.phase, self.batch_size):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            total += self._loss(x, y).item() * len(x)
            n += len(x)
        return total / max(n, 1)


def train_model(
    model,
    data,
    train_rows,
    stop_rows,
    *,
    loss_weights,
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
    trainer = _Trainer(model, data, phase, batch_size, lr, weight_decay,
                       loss_weights, device, amp)

    history = []
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_rows, seed + epoch)
        stop_loss = trainer.evaluate(stop_rows)
        history.append({"epoch": epoch, "train_loss": train_loss, "stop_loss": stop_loss})
        if verbose:
            print(f"{log_prefix}epoch {epoch:02d} | train={train_loss:.5f}"
                  f" | stop={stop_loss:.5f}", flush=True)

        if stop_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if patience is not None and stale >= patience:
                if verbose:
                    print(f"{log_prefix}early stop at epoch {epoch} "
                          f"(best epoch {best_epoch})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "best_stop_loss": best_loss,
    }


def fit_fixed_epochs(
    model,
    data,
    rows,
    *,
    loss_weights,
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
    trainer = _Trainer(model, data, phase, batch_size, lr, weight_decay,
                       loss_weights, device, amp)
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(rows, seed + epoch)
        history.append({"epoch": epoch, "train_loss": train_loss})
        if verbose:
            print(f"{log_prefix}epoch {epoch:02d} | train={train_loss:.5f}", flush=True)
    return {"model": model, "history": history}


def evaluate_loss(model, data, rows, *, loss_weights, phase="fit",
                  batch_size=INFER_BATCH_SIZE, device=None):
    """Weighted composite loss on ``rows`` (the Optuna objective on ``tune``)."""
    device = resolve_device(device)
    model.to(device)
    trainer = _Trainer(model, data, phase, batch_size, 1e-3, 0.0,
                       loss_weights, device, amp=False)
    return trainer.evaluate(rows)


@torch.no_grad()
def predict_trajectories(model, data, rows, phase="fit",
                         batch_size=INFER_BATCH_SIZE, device=None):
    """Predict days 83..89 for ``rows`` in their original order.

    Returns a dict of float32 arrays:

    - ``p_active`` ``(N, 7)``, ``p_pay`` ``(N, 7)``: probabilities.
    - ``listen`` ``(N, 7, 7)``: scaled listening amounts *if active*.
    - ``amount`` ``(N, 7, 2)``: scaled paid amount and plan days *if paying*.
    """
    device = resolve_device(device)
    model.to(device).eval()
    parts = {key: [] for key in ("p_active", "listen", "p_pay", "amount")}

    for x, _ in data.iter_batches(rows, phase, batch_size):
        pred = model(x.to(device, non_blocking=True)).float()
        parts["p_active"].append(torch.sigmoid(pred[..., T_ACTIVE]).cpu().numpy())
        parts["listen"].append(pred[..., T_LISTEN].cpu().numpy())
        parts["p_pay"].append(torch.sigmoid(pred[..., T_PAY]).cpu().numpy())
        parts["amount"].append(pred[..., T_AMOUNT].cpu().numpy())

    return {key: np.concatenate(value) for key, value in parts.items()}
