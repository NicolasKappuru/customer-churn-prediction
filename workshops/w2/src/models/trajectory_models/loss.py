"""Composite masked loss for the 11 trajectory targets.

    L = w_active * BCE(active)
      + w_listen * Huber(listening amounts | active day)
      + w_pay    * BCE(pay)
      + w_amount * Huber(paid amount, plan days | payment day)

No ``pos_weight`` on ``pay``: option B samples payments from ``sigmoid(logit)``,
so the probabilities must stay calibrated. Re-weighting the positives would
make every simulated user pay too often.

The weights are fixed once with ``calibrate_weights``: each term is divided
by the loss of a constant predictor on ``fit`` rows, so every term starts
near 1 and no single scale dominates the gradient.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    HUBER_DELTA,
    LOSS_TERMS,
    SEED,
    T_ACTIVE,
    T_AMOUNT,
    T_LISTEN,
    T_PAY,
)


def _masked_mean(values, mask):
    """Mean of ``values`` where ``mask`` is 1; 0 (with gradient) if empty."""
    mask = mask.to(values.dtype)
    total = mask.sum()
    if total.item() == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / total


def loss_terms(pred, y, huber_delta=HUBER_DELTA):
    """Return the four unweighted loss terms as a dict of scalars."""
    pred = pred.float()
    y = y.float()
    active = y[..., T_ACTIVE]
    pay = y[..., T_PAY]

    listen = F.huber_loss(
        pred[..., T_LISTEN], y[..., T_LISTEN], reduction="none", delta=huber_delta
    )
    amount = F.huber_loss(
        pred[..., T_AMOUNT], y[..., T_AMOUNT], reduction="none", delta=huber_delta
    )
    return {
        "active": F.binary_cross_entropy_with_logits(pred[..., T_ACTIVE], active),
        "listen": _masked_mean(listen, active[..., None].expand_as(listen)),
        "pay": F.binary_cross_entropy_with_logits(pred[..., T_PAY], pay),
        "amount": _masked_mean(amount, pay[..., None].expand_as(amount)),
    }


class TrajectoryLoss(nn.Module):
    """Weighted sum of ``loss_terms``; called as ``criterion(pred, y)``."""

    def __init__(self, weights, huber_delta=HUBER_DELTA):
        super().__init__()
        missing = set(LOSS_TERMS) - set(weights)
        if missing:
            raise ValueError(f"Missing loss weights: {sorted(missing)}")
        self.weights = {name: float(weights[name]) for name in LOSS_TERMS}
        self.huber_delta = huber_delta

    def forward(self, pred, y):
        terms = loss_terms(pred, y, self.huber_delta)
        return sum(self.weights[name] * terms[name] for name in LOSS_TERMS)


def _logit(p):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def constant_prediction(y):
    """Best constant predictor for each target, from a target array ``(N, 7, 11)``.

    Binary targets get the logit of their prevalence; amounts get their masked
    mean. Returns an array of shape ``(11,)``.
    """
    y = np.asarray(y, dtype=np.float64)
    active = y[..., T_ACTIVE] > 0
    pay = y[..., T_PAY] > 0
    const = np.zeros(y.shape[-1])

    const[T_LISTEN] = y[..., T_LISTEN][active].mean(axis=0) if active.any() else 0.0
    const[T_AMOUNT] = y[..., T_AMOUNT][pay].mean(axis=0) if pay.any() else 0.0
    const[T_ACTIVE] = _logit(active.mean())
    const[T_PAY] = _logit(pay.mean())
    return const


def calibrate_weights(data, rows, phase="fit", n_rows=50_000, block=4096, seed=SEED):
    """Weights = 1 / loss of the constant predictor, on a sample of ``rows``.

    Returns ``(weights, baseline_terms)``. Uses only ``rows`` (``fit``), so no
    information from ``stop``, ``tune`` or ``test`` enters the loss.
    """
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows)
    if len(rows) > n_rows:
        rows = np.sort(rng.choice(rows, n_rows, replace=False))
    ys = [data.make_batch(rows[i:i + block], phase)[1] for i in range(0, len(rows), block)]
    y = np.concatenate(ys)

    const = constant_prediction(y)
    pred = torch.from_numpy(np.broadcast_to(const, y.shape).astype(np.float32).copy())
    terms = loss_terms(pred, torch.from_numpy(y))
    baseline = {name: float(value) for name, value in terms.items()}
    weights = {name: 1.0 / max(value, 1e-6) for name, value in baseline.items()}
    return weights, baseline
