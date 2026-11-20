"""Option B: sample the predicted events and rebuild the subscription state.

The network predicts only the 13 event targets. A simulated week is produced by

1. sampling ``active`` and ``pay`` from their probabilities (no threshold:
   a daily payment probability rarely exceeds 0.5, so thresholding would make
   almost nobody pay),
2. writing the listening amounts on active days and the payment amounts on
   payment days (their predicted value, without extra noise),
3. sampling cancel / auto-renew on payment days,
4. recomputing the 7 state channels with the export's rules: a payment resets
   ``days_since_last`` to 0 and moves the expiry, otherwise recency grows and
   ``days_to_expire`` shrinks by one day.

The output uses the **raw export encoding** (log1p / signed log1p, flags), so
a simulated week can be appended to the window and fed back to the model
(``rollout``), or compared with the real tensor.

Expiry approximation: the export stores the real ``membership_expire_date``;
here it is derived as ``max(days_to_expire - 1, 0) + plan_days`` for a renewal
and 0 for a cancellation. ``rebuild_states`` with the *real* events measures
how much error this rule alone introduces.
"""

import numpy as np
import torch

from ..feature_preparation.scaling import transform_sequence
from ..feature_preparation.scaling.helpers import channel_index
from .config import (
    EVENT_AMOUNT_CHANNELS,
    HORIZON,
    INFER_BATCH_SIZE,
    INPUT_DAYS,
    LISTEN_AMOUNT_CHANNELS,
    SEED,
    STATE_CHANNELS,
    T_ACTIVE,
    T_AMOUNT,
    T_FLAGS,
    T_LISTEN,
    T_PAY,
)

MAX_DAYS = 3650


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _inv_slog(v):
    return np.sign(v) * np.expm1(np.abs(v))


# ---------------------------------------------------------------------------
# Scaling inverses
# ---------------------------------------------------------------------------

def unscale_listening(scaled, state):
    """Scaled listening amounts ``(..., 7)`` -> log1p values, clipped to the fit range."""
    ls = state["listening"]
    order = [ls["numeric_channels"].index(name) for name in LISTEN_AMOUNT_CHANNELS]
    mean = np.asarray(ls["mean"])[order]
    std = np.asarray(ls["std"])[order]
    upper = np.asarray(ls["upper"])[order]
    return np.clip(scaled * std + mean, 0.0, upper).astype(np.float32)


def unscale_amounts(scaled, state):
    """Scaled ``(tx_paid_sum, tx_plan_days_sum)`` ``(..., 2)`` -> raw encoding."""
    params = state["transaction"]["parameters"]
    out = np.empty_like(scaled, dtype=np.float32)
    for k, name in enumerate(EVENT_AMOUNT_CHANNELS):
        p = params[name]
        values = scaled[..., k] * p["std"] + p["mean"]
        if p.get("lower") is not None:
            values = np.clip(values, p["lower"], p["upper"])
        out[..., k] = values
    return out


def expected_listening(pred, state):
    """Point forecast of listening in log1p: ``p(active) * amount_if_active``."""
    return pred["p_active"][..., None] * unscale_listening(pred["listen"], state)


# ---------------------------------------------------------------------------
# State bookkeeping
# ---------------------------------------------------------------------------

def decode_last_state(raw_day, index):
    """Read the state carried by one raw day ``(N, 20)`` into plain integers."""
    return {
        "known": raw_day[:, index["tx_state_known"]] > 0,
        "cancel": raw_day[:, index["tx_last_is_cancel"]] > 0,
        "renew": raw_day[:, index["tx_last_is_auto_renew"]] > 0,
        "expiry_known": raw_day[:, index["tx_expiry_known"]] > 0,
        "ambiguous": raw_day[:, index["tx_state_ambiguous"]] > 0,
        "since": np.rint(np.expm1(raw_day[:, index["tx_days_since_last"]])).astype(np.int64),
        "to_expire": np.rint(_inv_slog(raw_day[:, index["tx_days_to_expire"]])).astype(np.int64),
    }


def _advance(state, pay, cancel, renew, plan_days):
    """One day of the export's state rule, vectorised over users."""
    s = {k: v.copy() for k, v in state.items()}
    # No payment: recency +1, expiry -1 (only where known).
    s["since"] = np.where(s["known"], s["since"] + 1, 0)
    s["to_expire"] = np.where(s["expiry_known"], s["to_expire"] - 1, 0)
    # Payment: reset recency, take the new flags and expiry.
    renewed = np.maximum(s["to_expire"], 0) + plan_days
    new_expiry = np.where(cancel, 0, np.where(state["expiry_known"], renewed, plan_days))
    s["known"] = s["known"] | pay
    s["expiry_known"] = s["expiry_known"] | pay
    s["since"] = np.where(pay, 0, s["since"])
    s["to_expire"] = np.where(pay, new_expiry, s["to_expire"])
    s["cancel"] = np.where(pay, cancel, s["cancel"])
    s["renew"] = np.where(pay, renew, s["renew"])
    s["ambiguous"] = np.where(pay, False, s["ambiguous"])
    return s


def _write_state(out_day, s, index):
    """Encode a state dict into the 7 state channels of one raw day."""
    known = s["known"].astype(np.float32)
    exp_known = (s["expiry_known"] & s["known"]).astype(np.float32)
    out_day[:, index["tx_state_known"]] = known
    out_day[:, index["tx_last_is_cancel"]] = s["cancel"] * known
    out_day[:, index["tx_last_is_auto_renew"]] = s["renew"] * known
    out_day[:, index["tx_state_ambiguous"]] = s["ambiguous"] * known
    out_day[:, index["tx_expiry_known"]] = exp_known
    out_day[:, index["tx_days_since_last"]] = (
        np.log1p(np.clip(s["since"], 0, MAX_DAYS)) * known
    )
    out_day[:, index["tx_days_to_expire"]] = (
        _slog(np.clip(s["to_expire"], -MAX_DAYS, MAX_DAYS)) * exp_known
    )


def rebuild_states(last_input_day, events, index):
    """Recompute the 7 state channels for a week of events.

    ``last_input_day``: raw day 82 ``(N, 20)``. ``events``: raw ``(N, 7, 20)``
    whose event channels (``tx_*`` 8..12) are filled. Returns a copy of
    ``events`` with the state channels written.
    """
    out = np.array(events, dtype=np.float32, copy=True)
    state = decode_last_state(last_input_day, index)
    for h in range(out.shape[1]):
        day = out[:, h, :]
        pay = day[:, index["tx_count"]] > 0
        cancel = pay & (day[:, index["tx_cancel_count"]] > 0)
        renew = pay & (day[:, index["tx_auto_renew_count"]] > 0)
        plan = np.rint(np.expm1(day[:, index["tx_plan_days_sum"]])).astype(np.int64)
        state = _advance(state, pay, cancel, renew, plan)
        _write_state(day, state, index)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_week(pred, last_input_day, scaler_state, index, rng):
    """Turn one prediction dict into a raw ``(N, 7, 20)`` simulated week."""
    n, horizon = pred["p_active"].shape
    out = np.zeros((n, horizon, len(index)), dtype=np.float32)

    active = rng.random((n, horizon)) < pred["p_active"]
    listen = unscale_listening(pred["listen"], scaler_state)
    for k, name in enumerate(LISTEN_AMOUNT_CHANNELS):
        out[:, :, index[name]] = np.where(active, listen[..., k], 0.0)
    out[:, :, index["active"]] = active

    pay = rng.random((n, horizon)) < pred["p_pay"]
    flags = rng.random((n, horizon, 2)) < pred["p_flags"]
    amounts = unscale_amounts(pred["amount"], scaler_state)
    plan_days = np.maximum(np.rint(np.expm1(amounts[..., 1])), 0)
    out[:, :, index["tx_count"]] = np.where(pay, np.log1p(1.0), 0.0)
    out[:, :, index["tx_paid_sum"]] = np.where(pay, amounts[..., 0], 0.0)
    out[:, :, index["tx_plan_days_sum"]] = np.where(pay, np.log1p(plan_days), 0.0)
    out[:, :, index["tx_cancel_count"]] = np.where(pay & flags[..., 0], np.log1p(1.0), 0.0)
    out[:, :, index["tx_auto_renew_count"]] = np.where(pay & flags[..., 1], np.log1p(1.0), 0.0)

    return rebuild_states(last_input_day, out, index)


@torch.no_grad()
def _predict_raw(model, raw_window, scaler_state, channel_specs, device, batch_size):
    """Model output for raw windows ``(N, 83, 20)``; same dict as ``predict_trajectories``."""
    model.eval()
    parts = {key: [] for key in ("p_active", "listen", "p_pay", "amount", "p_flags")}
    for start in range(0, len(raw_window), batch_size):
        raw = raw_window[start:start + batch_size]
        x = torch.from_numpy(transform_sequence(raw, scaler_state, channel_specs))
        out = model(x.to(device)).float()
        parts["p_active"].append(torch.sigmoid(out[..., T_ACTIVE]).cpu().numpy())
        parts["listen"].append(out[..., T_LISTEN].cpu().numpy())
        parts["p_pay"].append(torch.sigmoid(out[..., T_PAY]).cpu().numpy())
        parts["amount"].append(out[..., T_AMOUNT].cpu().numpy())
        parts["p_flags"].append(torch.sigmoid(out[..., T_FLAGS]).cpu().numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}


def simulate_next_week(model, raw_window, scaler_state, channel_specs, *,
                       rng, device="cpu", batch_size=INFER_BATCH_SIZE):
    """Raw window of the last 83 days ``(N, 83, 20)`` -> sampled raw week ``(N, 7, 20)``.

    This is the entry point for an environment: build the user's last 83 days
    (after applying an action), call this, and read the simulated week.
    """
    raw_window = np.asarray(raw_window, dtype=np.float32)
    if raw_window.shape[1] != INPUT_DAYS:
        raise ValueError(f"Expected {INPUT_DAYS} input days, got {raw_window.shape[1]}.")
    index = channel_index(channel_specs)
    pred = _predict_raw(model, raw_window, scaler_state, channel_specs, device, batch_size)
    return sample_week(pred, raw_window[:, -1, :], scaler_state, index, rng)


def rollout(model, raw_window, scaler_state, channel_specs, *, n_weeks,
            seed=SEED, device="cpu", batch_size=INFER_BATCH_SIZE):
    """Chain ``n_weeks`` simulated weeks; returns raw ``(N, 7 * n_weeks, 20)``."""
    rng = np.random.default_rng(seed)
    window = np.asarray(raw_window, dtype=np.float32)
    weeks = []
    for _ in range(n_weeks):
        week = simulate_next_week(model, window, scaler_state, channel_specs,
                                  rng=rng, device=device, batch_size=batch_size)
        weeks.append(week)
        window = np.concatenate([window[:, HORIZON:, :], week], axis=1)
    return np.concatenate(weeks, axis=1)


def simulate_from_predictions(pred, raw_inputs_last_day, scaler_state, channel_specs,
                              *, seed=SEED):
    """Sample one week per user from stored predictions (used in the evaluation)."""
    index = channel_index(channel_specs)
    return sample_week(pred, raw_inputs_last_day, scaler_state, index,
                       np.random.default_rng(seed))


__all__ = [
    "STATE_CHANNELS",
    "decode_last_state",
    "expected_listening",
    "rebuild_states",
    "rollout",
    "sample_week",
    "simulate_from_predictions",
    "simulate_next_week",
    "unscale_amounts",
    "unscale_listening",
]
