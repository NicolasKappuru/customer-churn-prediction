"""Decode model outputs and raw tensor values back to the export encoding.

Only what the evaluation needs: undo the standard scaling of the predicted
amounts and read the subscription state of a raw day. Nothing is sampled.
"""

import numpy as np

from .config import EVENT_AMOUNT_CHANNELS, LISTEN_AMOUNT_CHANNELS


def _inv_slog(x):
    """Inverse of ``sign(x) * log1p(|x|)``."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.expm1(np.abs(x))


def unscale_listening(listen, scaler_state):
    """Scaled listening amounts ``(..., 7)`` -> raw ``log1p`` values (clipped at 0)."""
    state = scaler_state["listening"]
    order = [state["numeric_channels"].index(name) for name in LISTEN_AMOUNT_CHANNELS]
    mean = np.asarray(state["mean"])[order]
    std = np.asarray(state["std"])[order]
    return np.maximum(listen * std + mean, 0.0)


def unscale_amounts(amount, scaler_state):
    """Scaled ``(paid, plan_days)`` ``(..., 2)`` -> raw encoding (signed log1p, log1p)."""
    params = scaler_state["transaction"]["parameters"]
    mean = np.asarray([params[name]["mean"] for name in EVENT_AMOUNT_CHANNELS])
    std = np.asarray([params[name]["std"] for name in EVENT_AMOUNT_CHANNELS])
    out = amount * std + mean
    out[..., 1] = np.maximum(out[..., 1], 0.0)          # plan days are non-negative
    return out


def expected_listening(pred, scaler_state):
    """Point forecast of the 7 listening channels: ``p(active) x amount if active``, ``(N, 7, 7)``."""
    return pred["p_active"][..., None] * unscale_listening(pred["listen"], scaler_state)


def decode_last_state(last_day, index):
    """Subscription state on one raw day ``(N, 14)``: days to expire and whether it is known.

    The v6 export has no separate expiry-validity flag; ``tx_state_known`` marks
    the days where ``tx_days_to_expire`` is defined.
    """
    return {
        "to_expire": np.rint(_inv_slog(last_day[:, index["tx_days_to_expire"]])).astype(int),
        "expiry_known": last_day[:, index["tx_state_known"]] > 0,
    }
