"""Prepared-data access and block batching for the LSTM.

- The v6 exports hold exactly the 14 ``MODEL_CHANNELS``; ``keep`` checks that
  by name and selects them in that order (an identity selection).
- When the tensor fits in ``W2_RAM_CACHE_GB`` (default 8), it is loaded into RAM
  once and each phase is scaled once (``raw_rows`` / ``scaled_rows``), instead
  of reading and scaling the memory map on every batch.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..feature_preparation.data import (
    load_cohorts,
    load_metadata,
    load_sequence_view,
)
from ..feature_preparation.scaling import load_preprocessing, transform_sequence
from ..constants import MODEL_CHANNELS
from .config import EXPORT_DIR, FEATURE_DIR

RAM_LIMIT_GB = float(os.environ.get("W2_RAM_CACHE_GB", "8"))


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
    keep: np.ndarray = None          # tensor positions of MODEL_CHANNELS
    use_cache: bool = False
    _raw: np.ndarray = field(default=None, repr=False)
    _scaled: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.keep is None:
            names = [spec["name"] for spec in self.channel_specs]
            missing = [c for c in MODEL_CHANNELS if c not in names]
            if missing:
                raise ValueError(f"MODEL_CHANNELS not in the export: {missing}")
            self.keep = np.asarray([names.index(c) for c in MODEL_CHANNELS], dtype=np.int64)

    @property
    def input_dim(self):
        return len(self.keep)

    # ---- RAM cache
    def _raw_all(self):
        if self._raw is None:
            start = time.perf_counter()
            self._raw = np.asarray(self.view[np.arange(self.view.shape[0])])
            print(f"loaded raw tensor into RAM: {self._raw.shape} {self._raw.dtype}"
                  f" in {time.perf_counter() - start:.1f}s")
        return self._raw

    def raw_rows(self, rows):
        """Raw (unscaled) float32 block for ``rows``."""
        rows = np.asarray(rows, dtype=np.int64)
        if not self.use_cache:
            return np.asarray(self.view[rows], dtype=np.float32)
        return self._raw_all()[rows].astype(np.float32)

    def scaled_rows(self, rows, phase):
        """Scaled float32 block (all channels) for ``rows`` with the ``phase`` scaler."""
        rows = np.asarray(rows, dtype=np.int64)
        if not self.use_cache:
            return transform_sequence(self.raw_rows(rows), self.states[phase], self.channel_specs)
        if phase not in self._scaled:
            self._scaled.clear()                 # one phase in memory at a time
            start = time.perf_counter()
            raw, block = self._raw_all(), 16384
            out = np.empty(raw.shape, dtype=np.float16)
            for i in range(0, len(raw), block):
                out[i:i + block] = transform_sequence(
                    raw[i:i + block].astype(np.float32), self.states[phase], self.channel_specs)
            self._scaled[phase] = out
            print(f"scaled tensor for phase {phase!r} cached in {time.perf_counter() - start:.1f}s")
        return self._scaled[phase][rows].astype(np.float32)

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
            x = self.scaled_rows(chunk, phase)[:, :, self.keep]
            y = self.y_all[chunk].astype(np.float32).reshape(-1, 1)
            yield torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


def _check_preparation(feature_dir, export_dir, meta):
    """Stop if ``feature_dir`` was prepared from a different export."""
    path = feature_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run feature_preparation.prepare() first.")
    contract = json.loads(path.read_text())["contract"]
    for key in ("hash_train_feb", "hash_test"):
        if contract.get(key) != meta.get(key):
            raise ValueError(
                f"{feature_dir} was prepared from another export than {export_dir} ({key} differs). "
                "Run prepare() again for this run.")


def load_prepared_data(feature_dir=FEATURE_DIR, export_dir=EXPORT_DIR, cache=None):
    """Load the artifacts written by ``feature_preparation.prepare``.

    ``cache=None`` holds the tensor in RAM when it fits in ``W2_RAM_CACHE_GB``.
    """
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
    _check_preparation(feature_dir, export_dir, meta)

    shape = view.shape
    need_gb = np.prod(shape) * 2 * 2 / 1e9            # raw float16 + one scaled float16 copy
    use_cache = (need_gb <= RAM_LIMIT_GB) if cache is None else bool(cache)
    print(f"sequence data: {shape[0]:,} rows | RAM cache {'on' if use_cache else 'off'}"
          f" (needs {need_gb:.1f} GB, limit {RAM_LIMIT_GB:g} GB)")

    return PreparedSequenceData(
        meta=meta,
        cohort_df=cohort_df,
        y_all=y_all,
        cohort_labels=cohort_labels,
        view=view,
        split=split,
        states=states,
        channel_specs=channel_specs,
        use_cache=use_cache,
    )
