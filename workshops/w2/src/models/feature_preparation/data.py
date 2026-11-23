"""Sequence export loading (contract v6).

Reads the export written by ``data_processing/sequences/dataset_export.py``::

    seq_lstm_<cohort>.npy       (users, 90, channels) float16
    labels_<cohort>.parquet     row, msno, is_churn  (same order as the array)
    metadata.json               contract: cutoffs, channels, shapes, row hashes

The two labelled cohorts are stacked (February development first, then March
test). ``row`` is the position in that stack, and every split index refers to it.
The tensors stay on disk: :func:`load_sequence_view` memory-maps both arrays
and exposes them as one ``(rows, days, channels)`` view.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    DATASET_VERSION,
    DEV_COHORT,
    EXPECTED_CUTOFFS,
    MODEL_CHANNELS,
    N_DAYS,
    TEST_COHORT,
)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def load_metadata(input_dir, expected_cutoffs=EXPECTED_CUTOFFS):
    """Read ``metadata.json`` and check version, cutoffs and channels."""
    path = Path(input_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The export is missing or incomplete "
            "(metadata.json is written last): run the preprocessing notebook."
        )
    meta = json.loads(path.read_text())

    if meta.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"Expected dataset_version {DATASET_VERSION}, got {meta.get('dataset_version')!r}."
        )
    if meta.get("cutoffs") != dict(expected_cutoffs):
        raise ValueError(
            f"Expected cutoffs {dict(expected_cutoffs)}, got {meta.get('cutoffs')}. "
            "Regenerate the export with the preprocessing notebook."
        )

    channels = meta["sequences"]["lstm"]["channels"]
    missing = [c for c in MODEL_CHANNELS if c not in channels]
    if missing:
        raise ValueError(f"MODEL_CHANNELS not in the export: {missing}")
    return meta


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _order_hash(msno_series):
    """SHA-1 of the ``msno`` order (same recipe as ``dataset_export.hash_msno``)."""
    digest = hashlib.sha1()
    for value in msno_series:
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_cohorts(input_dir, meta, dev_cohort=DEV_COHORT, test_cohort=TEST_COHORT):
    """Stack the labels of both cohorts; return ``(cohort_df, y_all, cohort_labels)``."""
    input_dir = Path(input_dir)
    frames = []
    for cohort in (dev_cohort, test_cohort):
        frame = pd.read_parquet(
            input_dir / meta["labels"][cohort]["file"], columns=["row", "msno", "is_churn"]
        )
        # The label file must follow the array row order.
        frame = frame.sort_values("row", kind="stable").reset_index(drop=True)
        if not np.array_equal(frame["row"].to_numpy(), np.arange(len(frame))):
            raise ValueError(f"{cohort}: label rows are not 0..N-1 without gaps.")
        if not frame["msno"].is_unique:
            raise ValueError(f"{cohort}: duplicated msno.")
        if _order_hash(frame["msno"]) != meta[f"hash_{cohort}"]:
            raise ValueError(f"{cohort}: labels do not match the published row hash.")
        frame = frame.drop(columns="row")
        frame.insert(1, "cohort", cohort)
        frames.append(frame)

    cohort_df = pd.concat(frames, ignore_index=True)
    cohort_df["cohort"] = cohort_df["cohort"].astype("category")
    cohort_df["row"] = np.arange(len(cohort_df))
    cohort_labels = cohort_df["cohort"].to_numpy(dtype=str)

    y_all = cohort_df["is_churn"].to_numpy(dtype=np.int32)
    if not set(np.unique(y_all)) <= {0, 1}:
        raise ValueError("is_churn must be 0/1.")

    for cohort in (dev_cohort, test_cohort):
        mask = cohort_labels == cohort
        print(
            f"{cohort:9s} rows: {mask.sum():>9,}"
            f" | label month: {meta.get('label_months', {}).get(cohort, '?')}"
            f" | churn rate: {y_all[mask].mean():.4f}"
        )
    return cohort_df, y_all, cohort_labels


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

class StackedSequenceView:
    """Two memory-mapped cohort arrays read as one ``(rows, days, channels)`` array.

    Supports an integer, a slice or an integer array (any order) on the first
    axis, optionally followed by more indices for the other axes. It always
    returns an in-memory ``numpy`` array.
    """

    def __init__(self, parts):
        self.parts = list(parts)
        tails = {part.shape[1:] for part in self.parts}
        dtypes = {part.dtype for part in self.parts}
        if len(tails) != 1 or len(dtypes) != 1:
            raise ValueError(f"Cohort arrays differ in shape/dtype: {tails}, {dtypes}.")
        self.offsets = np.cumsum([0] + [len(part) for part in self.parts])
        self.shape = (int(self.offsets[-1]), *tails.pop())
        self.dtype = dtypes.pop()
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def _take(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise IndexError("Row index must be one-dimensional.")
        if rows.size and (rows.min() < 0 or rows.max() >= self.shape[0]):
            raise IndexError(f"Row index out of range for {self.shape[0]} rows.")
        out = np.empty((len(rows), *self.shape[1:]), dtype=self.dtype)
        part_of = np.searchsorted(self.offsets, rows, side="right") - 1
        for p, part in enumerate(self.parts):
            mask = part_of == p
            if mask.any():
                local = rows[mask] - self.offsets[p]
                order = np.argsort(local, kind="stable")   # sequential reads on the memmap
                block = part[local[order]]
                out[np.flatnonzero(mask)[order]] = block
        return out

    def __getitem__(self, index):
        rest = ()
        if isinstance(index, tuple):
            index, rest = index[0], index[1:]
        if isinstance(index, (int, np.integer)):
            row = int(index) + (self.shape[0] if index < 0 else 0)
            result = self._take([row])[0]          # row axis dropped
            return result[rest] if rest else result
        if isinstance(index, slice):
            result = self._take(np.arange(self.shape[0])[index])
        else:
            result = self._take(index)
        return result[(slice(None), *rest)] if rest else result

    def __array__(self, dtype=None, copy=None):
        array = self._take(np.arange(self.shape[0]))
        return array.astype(dtype) if dtype is not None else array


def load_sequence_view(input_dir, meta, dev_cohort=DEV_COHORT, test_cohort=TEST_COHORT):
    """Memory-map ``seq_lstm_<cohort>.npy`` for both cohorts and stack them lazily."""
    input_dir = Path(input_dir)
    spec = meta["sequences"]["lstm"]
    n_channels = len(spec["channels"])
    parts = []
    for cohort in (dev_cohort, test_cohort):
        array = np.load(input_dir / spec[cohort]["file"], mmap_mode="r")
        expected = (meta[f"n_{cohort}"], N_DAYS, n_channels)
        if array.shape != expected:
            raise ValueError(f"{cohort}: array shape {array.shape}, expected {expected}.")
        if array.dtype != np.float16:
            raise ValueError(f"{cohort}: expected float16, got {array.dtype}.")
        if spec[cohort].get("row_hash") != meta[f"hash_{cohort}"]:
            raise ValueError(f"{cohort}: sequence row hash differs from the labels hash.")
        parts.append(array)
    return StackedSequenceView(parts)
