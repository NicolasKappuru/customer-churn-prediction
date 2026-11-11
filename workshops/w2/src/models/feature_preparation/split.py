"""Shared fit / stop / tune / test split for the sequence models.

February (development) is split once by a stable ``msno`` hash, stratified by
label, into ``fit``, ``stop`` and ``tune``. March is the complete ``test``
cohort and takes part in no decision. The hash makes the assignment
reproducible without depending on any RNG order.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    DEV_COHORT,
    DEV_PARTS,
    SPLIT_HASH_SALT,
    SPLIT_SHARES,
    TEST_COHORT,
)


# ---------------------------------------------------------------------------
# Reproducible assignment
# ---------------------------------------------------------------------------

def _msno_hash_unit(msno, salt=SPLIT_HASH_SALT):
    """Map an ``msno`` to a stable value in ``[0, 1)`` (salt, no RNG)."""
    digest = hashlib.sha1((salt + str(msno)).encode()).hexdigest()
    return int(digest[:15], 16) / float(16 ** 15)


def _hash_unit_series(cohort_df, hash_salt):
    """Return the stable hash unit for every stacked row."""
    return cohort_df["msno"].map(
        lambda value: _msno_hash_unit(value, hash_salt)
    ).to_numpy()


def _stratified_hash_assign(rows, shares, y_all, hash_unit):
    """Assign rows to parts by their per-class hash rank, giving exact shares."""
    assignment = np.empty(len(rows), dtype=object)
    names = list(shares)
    cumulative = np.cumsum([shares[name] for name in names])

    for label in (0, 1):
        class_rows = rows[y_all[rows] == label]
        ranks = np.argsort(
            np.argsort(hash_unit[class_rows])
        ) / max(len(class_rows), 1)
        part_index = np.searchsorted(cumulative, ranks, side="right")
        part_index = np.clip(part_index, 0, len(names) - 1)
        positions = np.searchsorted(rows, class_rows)
        assignment[positions] = np.asarray(names, dtype=object)[part_index]
    return assignment


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def make_shared_split(
    cohort_df,
    cohort_labels,
    y_all,
    dev_cohort=DEV_COHORT,
    test_cohort=TEST_COHORT,
    shares=SPLIT_SHARES,
    hash_salt=SPLIT_HASH_SALT,
):
    """Create the fit/stop/tune/test split and return ``(split, dev_rows)``."""
    hash_unit = _hash_unit_series(cohort_df, hash_salt)
    test_rows = np.flatnonzero(cohort_labels == test_cohort)
    dev_rows = np.flatnonzero(cohort_labels == dev_cohort)
    assignment = _stratified_hash_assign(dev_rows, shares, y_all, hash_unit)

    split = {
        part: np.sort(dev_rows[assignment == part])
        for part in DEV_PARTS
    }
    split["test"] = test_rows
    return split, dev_rows


# ---------------------------------------------------------------------------
# Validate and persist
# ---------------------------------------------------------------------------

def _validate_parts(split, y_all):
    """Every part must be sorted, unique, non-empty, disjoint and have both classes."""
    seen = {}
    for part, index in split.items():
        assert len(index) > 0 and np.all(np.diff(index) > 0), (
            f"{part} not sorted/unique"
        )
        assert set(np.unique(y_all[index])) == {0, 1}, f"{part} missing a class"
        for other, other_index in seen.items():
            assert not np.intersect1d(index, other_index).size, (
                f"{part} overlaps {other}"
            )
        seen[part] = index


def _validate_cohort_membership(split, cohort_labels, dev_cohort, test_cohort):
    """Test is March only; development parts are February only."""
    assert set(cohort_labels[split["test"]]) == {test_cohort}
    assert all(
        set(cohort_labels[split[part]]) == {dev_cohort}
        for part in DEV_PARTS
    )


def _validate_development_rows(split, dev_rows):
    """The development rows must be exactly fit + stop + tune."""
    expected = np.sort(np.concatenate([split[part] for part in DEV_PARTS]))
    assert np.array_equal(dev_rows, expected)


def _validate_expected_sizes(split, expected_sizes):
    """Enforce the full-run cohort sizes."""
    observed = {part: int(len(index)) for part, index in split.items()}
    if observed != expected_sizes:
        raise ValueError(
            f"Unexpected split sizes: {observed}; expected {expected_sizes}."
        )


def _summary_row(part, index, cohort_labels, y_all, meta):
    """Build one diagnostic row for a split part."""
    cohort = cohort_labels[index[0]]
    cohort_size = max(int(np.sum(cohort_labels == cohort)), 1)
    return {
        "part": part,
        "cohort": cohort,
        "label_month": meta["label_months"][cohort],
        "n_users": len(index),
        "share_of_cohort": round(len(index) / cohort_size, 6),
        "churn_rate": round(float(y_all[index].mean()), 5),
        "n_churn": int(y_all[index].sum()),
        "n_stay": int((y_all[index] == 0).sum()),
    }


def _build_summary(split, cohort_labels, y_all, meta):
    """Return the per-part diagnostic frame."""
    return pd.DataFrame(
        [
            _summary_row(part, index, cohort_labels, y_all, meta)
            for part, index in split.items()
        ]
    )


def _save_split(split, dev_rows, summary, out_dir):
    """Write the indices and the split summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "split_indices.npz",
        **split,
        development=dev_rows,
    )
    summary.to_csv(out_dir / "split_summary.csv", index=False)


def validate_and_save_split(
    split,
    dev_rows,
    cohort_labels,
    y_all,
    meta,
    out_dir,
    dev_cohort=DEV_COHORT,
    test_cohort=TEST_COHORT,
    expected_sizes=None,
):
    """Check split invariants, save indices and return the summary frame."""
    _validate_parts(split, y_all)
    _validate_cohort_membership(split, cohort_labels, dev_cohort, test_cohort)
    _validate_development_rows(split, dev_rows)
    if expected_sizes is not None:
        _validate_expected_sizes(split, expected_sizes)

    summary = _build_summary(split, cohort_labels, y_all, meta)
    _save_split(split, dev_rows, summary, out_dir)

    fit_rate = float(y_all[split["fit"]].mean())
    print(
        "saved split_indices.npz and split_summary.csv | fit churn rate:",
        round(fit_rate, 5),
    )
    return summary, fit_rate
