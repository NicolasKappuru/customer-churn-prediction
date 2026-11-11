"""Shared helpers for sequence scaling."""

import numpy as np


def channel_index(channel_specs):
    """Map every channel name to its position in the tensor."""
    return {item["name"]: position for position, item in enumerate(channel_specs)}


def iter_blocks(view, rows, block_rows):
    """Yield ``(row_ids, float32_block)`` for the requested rows."""
    rows = np.asarray(rows)
    for start in range(0, len(rows), block_rows):
        index = rows[start:start + block_rows]
        yield index, np.asarray(view[index], dtype=np.float32)


def sample(values, keep, rng):
    """Take at most ``keep`` values without materialising a permutation."""
    if len(values) <= keep:
        return values
    return values[rng.choice(len(values), keep, replace=False)]


def sample_keep(quantile_sample, n_blocks, n_channels=1):
    """Distribute the bounded quantile sample across blocks and channels."""
    return max(1, quantile_sample // max(1, n_blocks * n_channels))


def required_mask(sequence, index):
    """Boolean mask for the days where the channel's mask is positive."""
    return sequence[:, :, index] > 0


def mean_std(count, total, total_sq):
    """Return mean and standard deviation from accumulated moments."""
    denominator = np.maximum(count, 1)
    mean = total / denominator
    std = np.sqrt(np.maximum(total_sq / denominator - mean ** 2, 0))
    return mean, std


def add_scalar_moments(accumulator, values):
    """Accumulate count, sum and sum-of-squares for one channel."""
    if not len(values):
        return
    accumulator[0] += len(values)
    accumulator[1] += values.sum()
    accumulator[2] += (values ** 2).sum()
