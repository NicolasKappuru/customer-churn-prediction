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


def postprocess(X_hat, channel_specs, seed=0, rebuild_state=False, conditional_amounts=True,
                upper=None):
    """Make TimeGAN output a valid w2 tensor (same encoding as the export).

    * ``flag`` channels: sampled ``Bernoulli(clip(x, 0, 1))``. Sampling keeps
      the daily rate; a 0.5 threshold would erase rare flags.
    * ``tx_state_known`` is cumulative (true from the first event onwards).
    * ``tx_count``: a transaction happens with probability ``x / log1p(1)``;
      on those days the count is at least one, otherwise 0.
    * every channel with a ``mask`` is 0 where the mask channel is 0
      (no listening on inactive days, no amounts without a transaction, ...).
    * unsigned ``log1p`` channels are clipped at 0.
    * ``conditional_amounts=True``: a channel masked by an event (listening by
      ``active``, amounts by ``tx_count``) is divided by that event's probability
      on the days the event is sampled. The generator outputs roughly the
      *expected* value ``p * amount``; once the event is sampled, the value
      must be the amount *given* the event. ``upper`` (per-channel maximum of the
      real training users, e.g. ``X_train.max(axis=(0, 1))``) caps the result;
      without it the generator's own maximum is used, which is too low when the
      generator is smooth. No effect when the generator outputs sharp events
      (``p`` close to 1).
    * ``rebuild_state=True``: the three state channels are recomputed from the
      generated payments (``rebuild_payment_state``) instead of taken from the
      generator, so payments and expiry stay coupled.
    """
    rng = np.random.default_rng(seed)
    X = np.array(X_hat, dtype=np.float32, copy=True)
    index = {s["name"]: i for i, s in enumerate(channel_specs)}

    if conditional_amounts:
        X_raw = X.copy()
        prob = {}
        if "active" in index:
            prob["active"] = np.clip(X_raw[:, :, index["active"]], 0.0, 1.0)
        if "tx_count" in index:
            prob["tx_count"] = np.clip(X_raw[:, :, index["tx_count"]] / ONE_EVENT, 0.0, 1.0)
        for spec in channel_specs:
            mask, i = spec.get("mask"), index[spec["name"]]
            if mask in prob and mask != spec["name"] and spec["encoding"] != "flag":
                cap = upper[i] if upper is not None else X_raw[:, :, i].max()
                X[:, :, i] = np.minimum(X_raw[:, :, i] / np.maximum(prob[mask], 1e-3), cap)

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

    if rebuild_state:
        X = rebuild_payment_state(X, index)

    # Flags and tx_count are final; apply the masks to everything else.
    for spec in channel_specs:
        i, mask = index[spec["name"]], spec.get("mask")
        if spec["encoding"] == "log1p":
            X[:, :, i] = np.maximum(X[:, :, i], 0.0)
        if mask in index and mask != spec["name"]:
            X[:, :, i] = np.where(X[:, :, index[mask]] > 0, X[:, :, i], 0.0)
    return X


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _inv_slog(x):
    return np.sign(x) * np.expm1(np.abs(x))


def rebuild_payment_state(X, index):
    """Recompute ``tx_state_known``, ``tx_days_since_last`` and ``tx_days_to_expire``.

    The generator's day-0 state is kept (it stands for the payment before the
    window); every later day follows the w2 builder rules, checked on real
    ``fit`` users:

    * no payment: ``days_since_last + 1`` and ``days_to_expire - 1``
      (98.7 % of real known days);
    * payment: ``days_since_last = 0`` and
      ``days_to_expire = max(previous - 1, 0) + plan_days`` (95 % exact);
    * the state becomes known at the first payment and stays known.

    Expects ``tx_count`` already sampled (call it after the event channels).
    """
    X = X.copy()
    k_known, k_since, k_exp = index["tx_state_known"], index["tx_days_since_last"], index["tx_days_to_expire"]
    pay = X[:, :, index["tx_count"]] > 0
    plan = np.rint(np.expm1(np.clip(X[:, :, index["tx_plan_days_sum"]], 0, None)))

    known = X[:, 0, k_known] > 0
    since = np.where(known, np.rint(np.expm1(np.clip(X[:, 0, k_since], 0, None))), 0.0)
    exp = np.where(known, np.rint(_inv_slog(X[:, 0, k_exp])), 0.0)
    for t in range(X.shape[1]):
        if t > 0:
            since = np.where(known, since + 1, 0.0)
            exp = np.where(known, exp - 1, 0.0)
        p = pay[:, t]
        exp = np.where(p, np.where(known, np.maximum(exp, 0.0), 0.0) + plan[:, t], exp)
        since = np.where(p, 0.0, since)
        known = known | p
        since = np.clip(since, 0, 3650)
        exp = np.clip(exp, -3650, 3650)
        X[:, t, k_known] = known
        X[:, t, k_since] = np.where(known, np.log1p(since), 0.0)
        X[:, t, k_exp] = np.where(known, _slog(exp), 0.0)
    return X
