"""Apply a fitted scaler state to raw sequence blocks."""

import numpy as np

from ...constants import N_CHANNELS
from .helpers import channel_index, required_mask


def _as_sequence(sequence):
    """Return a float32 ``(users, days, channels)`` array or raise."""
    raw = np.asarray(sequence, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[2] != N_CHANNELS:
        raise ValueError(f"Expected (users, days, {N_CHANNELS}), got {raw.shape}.")
    return raw


def _transform_listening(output, raw, state, index):
    """Scale active listening days and leave inactive days at zero."""
    numeric_index = [index[name] for name in state["numeric_channels"]]
    active = raw[:, :, index[state["active_channel"]]] > 0
    values = np.minimum(
        raw[:, :, numeric_index], np.asarray(state["upper"])
    )
    values = (
        values - np.asarray(state["mean"])
    ) / np.asarray(state["std"])
    values = np.where(active[:, :, None], values, 0.0)
    for position, channel in enumerate(numeric_index):
        output[:, :, channel] = values[:, :, position]


def _scaled_masked_column(raw_column, mask, parameters):
    """Clip and standardise one channel, writing zeros outside its mask."""
    values = raw_column[mask].astype(np.float32)
    if parameters["lower"] is not None:
        values = np.clip(values, parameters["lower"], parameters["upper"])
    values = (values - parameters["mean"]) / parameters["std"]
    column = np.zeros(mask.shape, dtype=np.float32)
    column[mask] = values
    return column


def _transform_transactions(output, raw, state, index):
    """Scale event/state channels and leave flags untouched."""
    for name, parameters in state["parameters"].items():
        if parameters["encoding"] == "flag":
            continue
        channel = index[name]
        mask = required_mask(raw, index[parameters["mask"]])
        output[:, :, channel] = _scaled_masked_column(
            raw[:, :, channel], mask, parameters
        )


def transform_sequence(sequence, state, channel_specs):
    """Apply a fitted state to raw ``(users, days, 20)`` sequences."""
    raw = _as_sequence(sequence)
    index = channel_index(channel_specs)
    output = np.array(raw, dtype=np.float32, copy=True)
    _transform_listening(output, raw, state["listening"], index)
    _transform_transactions(output, raw, state["transaction"], index)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite values after sequence transformation.")
    return output
