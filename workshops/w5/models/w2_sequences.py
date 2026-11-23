"""Bridge between the w2 sequence export and TimeGAN.

The w2 tensors are ``(users, 90, 14)`` float16, already encoded per channel
(``log1p`` listening, ``signed_log1p`` amounts/days, 0/1 flags) and not scaled.
This module:

1. loads the rows of one split part (``fit`` by default, so the TimeGAN scaler
   never sees stop/tune/test), optionally only one class (e.g. churners);
2. turns TimeGAN's continuous output back into a valid w2 tensor, following
   each channel's ``encoding`` and ``mask`` from the export's ``metadata.json``:
   flags are sampled 0/1, a masked channel is 0 where its mask is 0, and
   unsigned ``log1p`` channels are non-negative.

Run it from ``workshops/w5`` or anywhere: the w2 package path is added here.
"""

import sys
from pathlib import Path

import numpy as np

W2_DIR = Path(__file__).resolve().parents[2] / "w2"
if str(W2_DIR) not in sys.path:
    sys.path.insert(0, str(W2_DIR))

from src.models.constants import ARTIFACTS_NAME, EXPORT_NAME, MODEL_CHANNELS  # noqa: E402
from src.models.feature_preparation.data import (  # noqa: E402
    load_cohorts,
    load_metadata,
    load_sequence_view,
)

EXPORT_DIR = W2_DIR / EXPORT_NAME
FEATURE_DIR = W2_DIR / "artifacts" / ARTIFACTS_NAME / "feature_preparation"
ONE_EVENT = float(np.log1p(1.0))       # encoded value of a single transaction


def load_w2_sequences(part="fit", label=None, export_dir=EXPORT_DIR, feature_dir=FEATURE_DIR,
                      max_rows=None, seed=0):
    """Return ``(X, y, channel_specs)`` for one part of the w2 split.

    ``X`` is float32 ``(n, 90, 14)`` in ``MODEL_CHANNELS`` order, ``y`` the churn
    labels. ``label=1`` keeps only churners (TimeGAN is unconditional: train one
    model per class to generate labelled data). ``max_rows`` subsamples.
    """
    meta = load_metadata(export_dir)
    _, y_all, _ = load_cohorts(export_dir, meta)
    view = load_sequence_view(export_dir, meta)

    with np.load(Path(feature_dir) / "split_indices.npz") as split:
        if part not in split.files:
            raise ValueError(f"Unknown part {part!r}; expected one of {split.files}.")
        rows = np.sort(split[part])
    if label is not None:
        rows = rows[y_all[rows] == label]
    if max_rows is not None and len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))

    specs_by_name = {s["name"]: s for s in meta["sequences"]["lstm"]["channel_specs"]}
    names = meta["sequences"]["lstm"]["channels"]
    keep = [names.index(c) for c in MODEL_CHANNELS]
    X = np.asarray(view[rows], dtype=np.float32)[:, :, keep]
    if not np.isfinite(X).all():
        raise ValueError("The export holds non-finite values.")
    specs = [specs_by_name[c] for c in MODEL_CHANNELS]
    print(f"w2 {part} rows: {len(rows):,} | label={label} | X {X.shape} "
          f"| churn rate {y_all[rows].mean():.4f}")
    return X, y_all[rows], specs


def postprocess(X_hat, channel_specs, seed=0):
    """Make TimeGAN output a valid w2 tensor (same encoding as the export).

    * ``flag`` channels: sampled ``Bernoulli(clip(x, 0, 1))``. Sampling keeps
      the daily rate; a 0.5 threshold would erase rare flags.
    * ``tx_state_known`` is cumulative (true from the first event onwards).
    * ``tx_count``: a transaction happens with probability ``x / log1p(1)``;
      on those days the count is at least one, otherwise 0.
    * every channel with a ``mask`` is 0 where the mask channel is 0
      (no listening on inactive days, no amounts without a transaction, ...).
    * unsigned ``log1p`` channels are clipped at 0.
    """
    rng = np.random.default_rng(seed)
    X = np.array(X_hat, dtype=np.float32, copy=True)
    index = {s["name"]: i for i, s in enumerate(channel_specs)}

    for name, i in index.items():
        if channel_specs[i]["encoding"] == "flag":
            X[:, :, i] = rng.random(X.shape[:2]) < np.clip(X[:, :, i], 0.0, 1.0)
    if "tx_state_known" in index:
        i = index["tx_state_known"]
        X[:, :, i] = np.maximum.accumulate(X[:, :, i], axis=1)
    if "tx_count" in index:
        i = index["tx_count"]
        event = rng.random(X.shape[:2]) < np.clip(X[:, :, i] / ONE_EVENT, 0.0, 1.0)
        X[:, :, i] = np.where(event, np.maximum(X[:, :, i], ONE_EVENT), 0.0)

    # Flags and tx_count are final; apply the masks to everything else.
    for spec in channel_specs:
        i, mask = index[spec["name"]], spec.get("mask")
        if spec["encoding"] == "log1p":
            X[:, :, i] = np.maximum(X[:, :, i], 0.0)
        if mask in index and mask != spec["name"]:
            X[:, :, i] = np.where(X[:, :, index[mask]] > 0, X[:, :, i], 0.0)
    return X
