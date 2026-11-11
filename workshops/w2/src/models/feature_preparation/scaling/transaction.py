"""Transaction and subscription-state scaler."""

import numpy as np

from ...constants import BLOCK_ROWS, QUANTILE_SAMPLE, SEED, TRANSACTION_SLICE
from .helpers import (
    add_scalar_moments,
    channel_index,
    iter_blocks,
    mean_std,
    required_mask,
    sample,
    sample_keep,
)


def _transaction_layout(channel_specs):
    """Return all transaction specs and the subset that needs scaling."""
    specs = channel_specs[TRANSACTION_SLICE]
    scaled_specs = tuple(
        item for item in specs if item["encoding"] != "flag"
    )
    if not scaled_specs:
        raise ValueError("No scaled transaction channels were found.")
    return specs, scaled_specs


def _masked_column(block, spec, index):
    """Return the values of one channel on the days selected by its mask."""
    mask = required_mask(block, index[spec["mask"]])
    return block[:, :, index[spec["name"]]][mask]


def _collect_transaction_quantile_samples(
    view, rows, scaled_specs, index, quantile_sample, block_rows, seed
):
    """Collect the bounded per-channel samples used for clipping limits."""
    rows = np.asarray(rows)
    n_blocks = max(1, int(np.ceil(len(rows) / block_rows)))
    keep = sample_keep(quantile_sample, n_blocks, len(scaled_specs))
    rng = np.random.default_rng(seed)
    samples = {item["name"]: [] for item in scaled_specs}

    for _, block in iter_blocks(view, rows, block_rows):
        for item in scaled_specs:
            values = _masked_column(block, item, index)
            if len(values):
                samples[item["name"]].append(sample(values, keep, rng))
    return samples


def _transaction_clip_limits(samples, scaled_specs):
    """Compute clipping bounds from each channel's sampled values."""
    limits = {}
    for item in scaled_specs:
        collected = samples[item["name"]]
        values = (
            np.concatenate(collected).astype(np.float64)
            if collected else np.empty(0)
        )
        if not len(values):
            limits[item["name"]] = None
            continue
        lower = (
            float(np.quantile(values, 0.001))
            if item["encoding"] == "signed_log1p"
            else 0.0
        )
        limits[item["name"]] = (lower, float(np.quantile(values, 0.999)))
    return limits


def _collect_transaction_moments(
    view, rows, scaled_specs, index, limits, block_rows
):
    """Accumulate moments over every clipped, masked value in the phase."""
    accumulators = {item["name"]: [0.0, 0.0, 0.0] for item in scaled_specs}

    for _, block in iter_blocks(view, rows, block_rows):
        for item in scaled_specs:
            name = item["name"]
            if limits[name] is None:
                continue
            values = np.clip(
                _masked_column(block, item, index),
                limits[name][0],
                limits[name][1],
            ).astype(np.float64)
            add_scalar_moments(accumulators[name], values)
    return accumulators


def _build_transaction_parameter(spec, limit, moments):
    """Assemble one serialisable transaction-channel state."""
    if limit is None:
        return {
            "encoding": spec["encoding"],
            "mask": spec["mask"],
            "lower": None,
            "upper": None,
            "mean": 0.0,
            "std": 1.0,
            "n_values": 0,
            "empty_fit": True,
        }

    count, total, total_sq = moments
    mean, std = mean_std(count, total, total_sq)
    return {
        "encoding": spec["encoding"],
        "mask": spec["mask"],
        "lower": limit[0],
        "upper": limit[1],
        "mean": float(mean),
        "std": float(max(std, 1e-6)),
        "n_values": int(count),
        "empty_fit": False,
    }


def _build_transaction_parameters(scaled_specs, limits, accumulators):
    """Build the per-channel state dictionary."""
    return {
        item["name"]: _build_transaction_parameter(
            item, limits[item["name"]], accumulators[item["name"]]
        )
        for item in scaled_specs
    }


def fit_transaction_scaler(
    view,
    rows,
    channel_specs,
    quantile_sample=QUANTILE_SAMPLE,
    block_rows=BLOCK_ROWS,
    seed=SEED,
):
    """Fit clipping and standardisation for the twelve transaction/state channels."""
    index = channel_index(channel_specs)
    specs, scaled_specs = _transaction_layout(channel_specs)
    samples = _collect_transaction_quantile_samples(
        view, rows, scaled_specs, index, quantile_sample, block_rows, seed
    )
    limits = _transaction_clip_limits(samples, scaled_specs)
    accumulators = _collect_transaction_moments(
        view, rows, scaled_specs, index, limits, block_rows
    )
    return {
        "channel_slice": [8, 20],
        "channels": [item["name"] for item in specs],
        "parameters": _build_transaction_parameters(
            scaled_specs, limits, accumulators
        ),
    }
