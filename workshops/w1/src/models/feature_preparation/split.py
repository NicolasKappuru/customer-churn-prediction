"""Shared fit / stop / tune / test split.

Development (February) is split once by a stable ``msno`` hash, stratified by
label, into ``fit``, ``stop`` and ``tune``; test is the whole March cohort,
kept out of every decision. The hash makes the assignment reproducible without
depending on any RNG order.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    DEV_COHORT,
    TEST_COHORT,
    SEED,
    SPLIT_HASH_SALT,
    SPLIT_SHARES,
)


# ---------------------------------------------------------------------------
# Reproducible assignment
# ---------------------------------------------------------------------------

def _msno_hash_unit(msno):
    """Map an ``msno`` to a stable value in ``[0, 1)`` (salt, no RNG)."""
    h = hashlib.sha1((SPLIT_HASH_SALT + str(msno)).encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def _stratified_hash_assign(rows, shares, y_all, hash_unit):
    """Assign rows to parts by their per-class hash rank, giving exact shares."""
    assign = np.empty(len(rows), dtype=object)
    names = list(shares)
    cum = np.cumsum([shares[n] for n in names])

    for cls in (0, 1):
        cls_rows = rows[y_all[rows] == cls]
        # Normalised rank of each row's hash inside its class, in [0, 1).
        ranks = np.argsort(
            np.argsort(hash_unit[cls_rows])
        ) / max(len(cls_rows), 1)
        # Cumulative shares cut the rank axis into fit / stop / tune.
        part_idx = np.searchsorted(cum, ranks, side="right")
        part_idx = np.clip(part_idx, 0, len(names) - 1)
        positions = np.searchsorted(rows, cls_rows)
        assign[positions] = np.asarray(names, dtype=object)[part_idx]
    return assign


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def make_shared_split(
    cohort_df,
    cohort_labels,
    y_all,
    smoke_users=None,
    seed=SEED,
    dev_cohort=DEV_COHORT,
    test_cohort=TEST_COHORT,
):
    """Create the fit/stop/tune/test split and return ``(split, dev_rows)``.

    An optional ``smoke_users`` shrinks every part keeping both classes.
    """
    hash_unit = cohort_df["msno"].map(_msno_hash_unit).to_numpy()
    test_rows = np.flatnonzero(cohort_labels == test_cohort)
    dev_rows = np.flatnonzero(cohort_labels == dev_cohort)
    # Only development is split; test stays whole as the outer evaluation.
    assignment = _stratified_hash_assign(dev_rows, SPLIT_SHARES, y_all, hash_unit)

    split = {
        "fit": np.sort(dev_rows[assignment == "fit"]),
        "stop": np.sort(dev_rows[assignment == "stop"]),
        "tune": np.sort(dev_rows[assignment == "tune"]),
        "test": test_rows,
    }
    if smoke_users is not None:
        rng = np.random.default_rng(seed)
        small = {}

        for part, idx in split.items():
            take = min(smoke_users, len(idx))
            pos = idx[y_all[idx] == 1]
            neg = idx[y_all[idx] == 0]
            # Keep the part's class balance; at least one churn row.
            n_pos = max(1, int(round(take * len(pos) / len(idx))))
            n_pos = min(n_pos, len(pos))
            n_neg = min(take - n_pos, len(neg))
            chosen = np.concatenate(
                [
                    rng.choice(pos, n_pos, replace=False),
                    rng.choice(neg, n_neg, replace=False),
                ]
            )
            small[part] = np.sort(chosen)

        split = small
        dev_rows = np.sort(
            np.concatenate([split[part] for part in ("fit", "stop", "tune")])
        )
        print(
            "SMOKE split sizes:",
            {key: len(value) for key, value in split.items()},
        )

    return split, dev_rows


# ---------------------------------------------------------------------------
# Validate and persist
# ---------------------------------------------------------------------------

def validate_and_save_split(
    split,
    dev_rows,
    cohort_df,
    cohort_labels,
    y_all,
    meta,
    out_dir,
    dev_cohort=DEV_COHORT,
    test_cohort=TEST_COHORT,
):
    """Check the split invariants, save indices and return ``(summary, fit_rate)``."""
    seen = {}

    for part, idx in split.items():
        assert len(idx) > 0 and np.all(np.diff(idx) > 0), (
            f"{part} not sorted/unique"
        )
        assert set(np.unique(y_all[idx])) == {0, 1}, f"{part} missing a class"

        for other, other_idx in seen.items():
            assert not np.intersect1d(idx, other_idx).size, (
                f"{part} overlaps {other}"
            )

        seen[part] = idx

    # A cross-month test: every part keeps its own cohort.
    assert set(cohort_labels[split["test"]]) == {test_cohort}
    assert all(
        set(cohort_labels[split[part]]) == {dev_cohort}
        for part in ("fit", "stop", "tune")
    )
    expected_dev_rows = np.sort(
        np.concatenate([split[part] for part in ("fit", "stop", "tune")])
    )
    assert np.array_equal(dev_rows, expected_dev_rows)

    days_to_expire = cohort_df["days_to_expire"].to_numpy()
    summary = pd.DataFrame(
        [
            {
                "part": part,
                "cohort": cohort_labels[idx[0]],
                "label_month": meta["label_months"][cohort_labels[idx[0]]],
                "n_users": len(idx),
                "churn_rate": round(float(y_all[idx].mean()), 5),
                "n_churn": int(y_all[idx].sum()),
                "n_stay": int((y_all[idx] == 0).sum()),
                "days_to_expire_p01": float(
                    np.percentile(days_to_expire[idx], 1)
                ),
                "days_to_expire_p50": float(
                    np.percentile(days_to_expire[idx], 50)
                ),
                "days_to_expire_p99": float(
                    np.percentile(days_to_expire[idx], 99)
                ),
            }
            for part, idx in split.items()
        ]
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "split_indices.npz",
        **split,
        development=dev_rows,
    )
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    fit_rate = float(y_all[split["fit"]].mean())
    print(
        "saved split_indices.npz and split_summary.csv | fit churn rate:",
        round(fit_rate, 5),
    )
    return summary, fit_rate
