"""Tabular preprocessing.

Numeric path: median imputation -> signed log1p -> clip (p0.5 / p99.5) ->
standardisation; categorical path: one-hot with the fit levels. The fitted
state is learned only on a phase's training rows and reused to transform other
rows, so no test information leaks into preprocessing.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def numeric_values(frame, columns):
    """Copy numeric columns and represent non-finite values as missing."""
    values = frame[columns].to_numpy(dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values


def category_values(series):
    """Convert a categorical column to strings, filling nulls with ``__MISSING__``."""
    return series.astype("string").fillna("__MISSING__").to_numpy(dtype=str)


# ---------------------------------------------------------------------------
# Fit and transform
# ---------------------------------------------------------------------------

def fit_tabular_preprocessor(frame, features, categorical):
    """Fit the numeric transform and categorical levels on ``frame`` and return the state."""
    numeric = [column for column in features if column not in categorical]
    raw = numeric_values(frame, numeric)
    median = np.array(
        [
            np.nanmedian(column) if np.isfinite(column).any() else 0.0
            for column in raw.T
        ]
    )
    x = np.where(np.isnan(raw), median, raw)
    # signed log1p keeps the sign while compressing long tails.
    x = np.sign(x) * np.log1p(np.abs(x))
    lower, upper = np.quantile(x, [0.005, 0.995], axis=0)
    x = np.clip(x, lower, upper)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    levels = {
        column: sorted(set(category_values(frame[column])))
        for column in categorical
    }
    return {
        "numeric": numeric,
        "categorical": list(categorical),
        "median": median.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "levels": levels,
    }


def transform_tabular(frame, state):
    """Apply a fitted state to ``frame``; learns nothing from it."""
    raw = numeric_values(frame, state["numeric"])
    x = np.where(np.isnan(raw), np.asarray(state["median"]), raw)
    x = np.sign(x) * np.log1p(np.abs(x))
    x = np.clip(x, state["lower"], state["upper"])
    pieces = [(x - np.asarray(state["mean"])) / np.asarray(state["std"])]
    # One-hot only against the fitted levels; unseen levels become all-zero.
    for column in state["categorical"]:
        values = category_values(frame[column])
        pieces.append(
            np.column_stack(
                [values == level for level in state["levels"][column]]
            )
        )
    out = np.concatenate(pieces, axis=1).astype(np.float32)
    assert np.isfinite(out).all()
    return out


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def make_phase_rows(split, dev_rows):
    """Use only February rows to fit either preprocessing phase."""
    return {
        "fit": split["fit"],
        "development": dev_rows,
    }


class PhasePreprocessor:
    """Fit and cache one tabular state per phase and transform rows with it."""

    def __init__(self, cohort_df, phase_rows, features, categorical):
        self.cohort_df = cohort_df
        self.phase_rows = phase_rows
        self.features = features
        self.categorical = categorical
        self._cache = {}

    def state(self, phase):
        """Return the fitted state for ``phase``, fitting it on first use."""
        if phase not in self._cache:
            rows = self.phase_rows[phase]
            self._cache[phase] = fit_tabular_preprocessor(
                self.cohort_df.iloc[rows], self.features, self.categorical
            )
        return self._cache[phase]

    def batch(self, rows, phase):
        """Transform labelled cohort rows using the February phase state."""
        return transform_tabular(
            self.cohort_df.iloc[np.asarray(rows)], self.state(phase)
        )
