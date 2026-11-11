"""Listening-channel scaler."""

import numpy as np

from ...constants import BLOCK_ROWS, LISTENING_SLICE, QUANTILE_SAMPLE, SEED
from .helpers import (
    channel_index,
    iter_blocks,
    mean_std,
    sample,
    sample_keep,
)


def _listening_layout(channel_specs, index):
    """Return listening names, numeric names, numeric indices and active index."""
    specs = channel_specs[LISTENING_SLICE]
    names = tuple(item["name"] for item in specs)
    numeric_names = tuple(
        item["name"] for item in specs if item["encoding"] != "flag"
    )
    numeric_index = tuple(index[name] for name in numeric_names)
    return names, numeric_names, numeric_index, index["active"]


def _sample_listening_values(
    view, rows, numeric_index, active_index, quantile_sample, block_rows, seed
):
    """Collect the bounded sample used for listening clipping quantiles."""
    rows = np.asarray(rows)
    n_blocks = max(1, int(np.ceil(len(rows) / block_rows)))
    keep = sample_keep(quantile_sample, n_blocks)
    rng = np.random.default_rng(seed)
    samples = []

    for _, block in iter_blocks(view, rows, block_rows):
        active = block[:, :, active_index] > 0
        values = block[:, :, numeric_index][active]
        if len(values):
            samples.append(sample(values, keep, rng))

    return np.concatenate(samples, axis=0).astype(np.float64) if samples else None


def _listening_clip_upper(values):
    """Validate the sample and return the per-channel p99.9 clipping bounds."""
    if values is None or len(values) == 0:
        raise ValueError("The fit rows contain no active listening records.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite, non-negative log1p listening values.")
    return np.quantile(values, 0.999, axis=0)


def _collect_listening_moments(
    view, rows, numeric_index, active_index, upper, block_rows
):
    """Accumulate moments over every clipped active-day value in the phase."""
    count = np.zeros(len(numeric_index), dtype=np.float64)
    total = np.zeros(len(numeric_index), dtype=np.float64)
    total_sq = np.zeros(len(numeric_index), dtype=np.float64)

    for _, block in iter_blocks(view, rows, block_rows):
        active = block[:, :, active_index] > 0
        values = np.minimum(
            block[:, :, numeric_index][active], upper
        ).astype(np.float64)
        if len(values):
            count += len(values)
            total += values.sum(axis=0)
            total_sq += (values ** 2).sum(axis=0)
    return count, total, total_sq


def _build_listening_state(
    names, numeric_names, upper, count, total, total_sq
):
    """Assemble the serialisable listening scaler state."""
    mean, std = mean_std(count, total, total_sq)
    std[std < 1e-6] = 1.0
    return {
        "channel_slice": [0, 8],
        "channels": list(names),
        "numeric_channels": list(numeric_names),
        "active_channel": "active",
        "encoding": "log1p",
        "upper": upper.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_values": int(count[0]) if len(count) else 0,
    }


def fit_listening_scaler(
    view,
    rows,
    channel_specs,
    quantile_sample=QUANTILE_SAMPLE,
    block_rows=BLOCK_ROWS,
    seed=SEED,
):
    """Fit clipping and standardisation for the eight listening channels."""
    index = channel_index(channel_specs)
    names, numeric_names, numeric_index, active_index = _listening_layout(
        channel_specs, index
    )
    values = _sample_listening_values(
        view, rows, numeric_index, active_index, quantile_sample, block_rows, seed
    )
    upper = _listening_clip_upper(values)
    count, total, total_sq = _collect_listening_moments(
        view, rows, numeric_index, active_index, upper, block_rows
    )
    return _build_listening_state(
        names, numeric_names, upper, count, total, total_sq
    )
