"""Prepared-data access and block batching for the LSTM."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..feature_preparation.data import (
    load_cohorts,
    load_metadata,
    load_sequence_view,
)
from ..feature_preparation.scaling import load_preprocessing, transform_sequence
from .config import EXPORT_DIR, FEATURE_DIR


@dataclass
class PreparedSequenceData:
    """Labels, split indices, memory-mapped sequences and phase scalers."""

    meta: dict
    cohort_df: object
    y_all: np.ndarray
    cohort_labels: np.ndarray
    view: object
    split: dict
    states: dict
    channel_specs: list

    @property
    def input_dim(self):
        return len(self.channel_specs)

    def rows(self, part):
        return self.split[part]

    def labels(self, part):
        return self.y_all[self.split[part]]

    def _prepare_rows(self, rows, batch_size, shuffle, drop_last, seed):
        rows = np.asarray(rows, dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(seed)
            rows = rows[rng.permutation(len(rows))]
        if drop_last:
            usable = (len(rows) // batch_size) * batch_size
            rows = rows[:usable]
        return rows

    def iter_batches(
        self,
        rows,
        phase,
        batch_size,
        shuffle=False,
        drop_last=False,
        seed=None,
    ):
        """Yield transformed ``(x, y)`` tensors from memory-mapped rows."""
        if phase not in self.states:
            raise ValueError(
                f"Unknown phase {phase!r}; expected one of {sorted(self.states)}."
            )
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for BatchNorm training.")

        rows = self._prepare_rows(rows, batch_size, shuffle, drop_last, seed)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            x = transform_sequence(
                self.view[chunk], self.states[phase], self.channel_specs
            )
            y = self.y_all[chunk].astype(np.float32).reshape(-1, 1)
            yield torch.from_numpy(x), torch.from_numpy(y)


def load_prepared_data(feature_dir=FEATURE_DIR, export_dir=EXPORT_DIR):
    """Load the artifacts written by ``feature_preparation.prepare``."""
    feature_dir = Path(feature_dir)
    export_dir = Path(export_dir)

    meta = load_metadata(export_dir)
    cohort_df, y_all, cohort_labels = load_cohorts(export_dir, meta)
    view = load_sequence_view(export_dir, meta)

    with np.load(feature_dir / "split_indices.npz") as loaded:
        split = {key: loaded[key] for key in loaded.files}
    states = {
        phase: load_preprocessing(feature_dir / "preprocessing" / f"{phase}.json")
        for phase in ("fit", "development")
    }
    channel_specs = meta["sequences"]["lstm"]["channel_specs"]

    return PreparedSequenceData(
        meta=meta,
        cohort_df=cohort_df,
        y_all=y_all,
        cohort_labels=cohort_labels,
        view=view,
        split=split,
        states=states,
        channel_specs=channel_specs,
    )
