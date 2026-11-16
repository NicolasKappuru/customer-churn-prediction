"""Input/target batches for trajectory prediction.

Wraps the ``PreparedSequenceData`` of the classifier (same export, split and
scalers) and changes only what each batch contains:

- ``x``: scaled days ``0..82`` of the model channels, shape ``(B, 83, 14)``.
- ``y``: days ``83..89`` of the 11 targets, shape ``(B, 7, 11)``; layout in
  ``config.TARGET_NAMES``.

Binary targets are read from the **raw** tensor (after scaling, a raw
``tx_count`` of 1 is about -0.1, not 1). Amounts are taken from the scaled
tensor, which already writes 0 outside their mask; the loss ignores those days.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..feature_preparation.scaling import transform_sequence
from ..feature_preparation.scaling.helpers import channel_index
from .config import (
    EVENT_AMOUNT_CHANNELS,
    EXPORT_DIR,
    FEATURE_DIR,
    INPUT_DAYS,
    LISTEN_AMOUNT_CHANNELS,
    N_TARGETS,
    T_ACTIVE,
    T_AMOUNT,
    T_LISTEN,
    T_PAY,
)


def build_targets(raw, scaled, index):
    """Return the ``(B, 7, 11)`` target array from raw and scaled target days."""
    targets = np.zeros((raw.shape[0], raw.shape[1], N_TARGETS), dtype=np.float32)
    listen = [index[name] for name in LISTEN_AMOUNT_CHANNELS]
    amounts = [index[name] for name in EVENT_AMOUNT_CHANNELS]

    active = raw[:, :, index["active"]] > 0
    pay = raw[:, :, index["tx_count"]] > 0
    targets[:, :, T_LISTEN] = np.where(active[..., None], scaled[:, :, listen], 0.0)
    targets[:, :, T_ACTIVE] = active
    targets[:, :, T_PAY] = pay
    targets[:, :, T_AMOUNT] = np.where(pay[..., None], scaled[:, :, amounts], 0.0)
    return targets


@dataclass
class TrajectoryData:
    """Same rows, split and scalers as the classifier; new ``(x, y)`` pairs.

    ``base`` is a ``recurrent_models.data.PreparedSequenceData`` (or any object
    with ``view``, ``split``, ``states``, ``channel_specs`` and ``meta``).
    """

    base: object

    def __post_init__(self):
        self.index = channel_index(self.base.channel_specs)

    # Pass-through so the rest of the code can use it like the base object.
    @property
    def meta(self):
        return self.base.meta

    @property
    def split(self):
        return self.base.split

    @property
    def states(self):
        return self.base.states

    @property
    def channel_specs(self):
        return self.base.channel_specs

    @property
    def view(self):
        return self.base.view

    @property
    def input_dim(self):
        return len(self.base.keep)

    def rows(self, part):
        return self.base.split[part]

    def raw(self, rows):
        """Raw (unscaled, log1p-encoded) float32 block for ``rows``."""
        if hasattr(self.base, "raw_rows"):
            return self.base.raw_rows(rows)   # RAM cache when available
        return np.asarray(self.base.view[np.asarray(rows)], dtype=np.float32)

    def make_batch(self, rows, phase):
        """Return numpy ``(x, y, raw)`` for one block of rows."""
        raw = self.raw(rows)
        if hasattr(self.base, "scaled_rows"):
            scaled = self.base.scaled_rows(rows, phase)   # scaled once per phase when cached
        else:
            scaled = transform_sequence(raw, self.base.states[phase], self.base.channel_specs)
        x = scaled[:, :INPUT_DAYS, self.base.keep]   # model channels only; targets use every channel
        y = build_targets(raw[:, INPUT_DAYS:, :], scaled[:, INPUT_DAYS:, :], self.index)
        return x, y, raw

    def iter_batches(
        self,
        rows,
        phase,
        batch_size,
        shuffle=False,
        drop_last=False,
        seed=None,
    ):
        """Yield ``(x, y)`` tensors; same contract as the classifier iterator."""
        if phase not in self.base.states:
            raise ValueError(
                f"Unknown phase {phase!r}; expected one of {sorted(self.base.states)}."
            )
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for BatchNorm training.")

        rows = np.asarray(rows, dtype=np.int64)
        if shuffle:
            rows = rows[np.random.default_rng(seed).permutation(len(rows))]
        if drop_last:
            rows = rows[:(len(rows) // batch_size) * batch_size]
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            x, y, _ = self.make_batch(chunk, phase)
            yield torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


def load_trajectory_data(feature_dir=FEATURE_DIR, export_dir=EXPORT_DIR):
    """Load the classifier artifacts and wrap them for trajectory prediction."""
    from ..recurrent_models.data import load_prepared_data

    return TrajectoryData(load_prepared_data(feature_dir, export_dir))
