"""Uncertainty of the LSTM vs GRU comparison on the test set.

- ``paired_bootstrap``: 95% percentile intervals for AUC-ROC, AUC-PR, F1 and
  log-loss of each model and of their difference. Both models are scored on
  the *same* resampled users in every replicate (paired), so the interval of
  the difference accounts for their correlation.
- ``delong_test``: DeLong test for the difference between two correlated
  ROC-AUCs (fast version of Sun & Xu, 2014).

Each bootstrap replicate is a vector of per-user counts (weights) instead of a
copied subset, and the scores are sorted once per model, so a replicate costs
O(n) instead of O(n log n). With ``weights = 1`` the metrics match scikit-learn.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from ..constants import EPS, SEED

METRICS = ("AUC-ROC", "AUC-PR", "F1", "Log-loss")


class _ModelScores:
    """Precomputed, sort-once views of one model's test predictions."""

    def __init__(self, y, proba, threshold):
        y = np.asarray(y).astype(np.float64)
        proba = np.clip(np.asarray(proba, dtype=np.float64), EPS, 1 - EPS)

        self.order = np.argsort(-proba, kind="mergesort")
        self.y_sorted = y[self.order]
        p_sorted = proba[self.order]
        # Last position of each group of tied scores (descending order).
        self.group_ends = np.r_[
            np.flatnonzero(np.diff(p_sorted)), len(p_sorted) - 1
        ]

        pred = (proba >= threshold).astype(np.float64)
        self.tp = y * pred
        self.fp = (1 - y) * pred
        self.fn = y * (1 - pred)
        self.log_loss = -(y * np.log(proba) + (1 - y) * np.log(1 - proba))

    def metrics(self, w):
        """AUC-ROC, AUC-PR (average precision), F1 and log-loss under weights ``w``."""
        w_sorted = w[self.order]
        tps = np.cumsum(w_sorted * self.y_sorted)[self.group_ends]
        fps = np.cumsum(w_sorted * (1 - self.y_sorted))[self.group_ends]
        n_pos, n_neg = tps[-1], fps[-1]

        tpr = np.r_[0.0, tps / n_pos]
        fpr = np.r_[0.0, fps / n_neg]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

        denom = tps + fps
        precision = np.divide(tps, denom, out=np.zeros_like(tps), where=denom > 0)
        ap = float(np.sum(np.diff(tpr) * precision))

        tp, fp, fn = w @ self.tp, w @ self.fp, w @ self.fn
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else 0.0
        log_loss = float(w @ self.log_loss / w.sum())
        return np.array([auc, ap, f1, log_loss])


def paired_bootstrap(runs, tunes, *, n_boot=1000, seed=SEED, alpha=0.05):
    """Point estimate and 95% CI per model, plus the first-minus-second difference.

    ``runs``/``tunes`` are the dicts used by ``compare_models``. The difference is
    ``first model - second model`` in the order of ``runs`` (e.g. LSTM - GRU).
    A difference is flagged as significant when its interval excludes 0.
    """
    names = list(runs)
    y = np.asarray(runs[names[0]]["y_test"])
    n = len(y)
    scorers = {
        name: _ModelScores(y, runs[name]["proba_test"], tunes[name]["threshold"])
        for name in names
    }

    ones = np.ones(n)
    point = {name: scorer.metrics(ones) for name, scorer in scorers.items()}

    rng = np.random.default_rng(seed)
    draws = {name: np.empty((n_boot, len(METRICS))) for name in names}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        for name, scorer in scorers.items():
            draws[name][b] = scorer.metrics(w)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for i, metric in enumerate(METRICS):
        row = {"Métrica": metric}
        for name in names:
            low, high = np.percentile(draws[name][:, i], [lo, hi])
            row[name.upper()] = point[name][i]
            row[f"{name.upper()} IC95"] = f"[{low:.4f}, {high:.4f}]"
        if len(names) == 2:
            a, b_ = names
            diff = draws[a][:, i] - draws[b_][:, i]
            low, high = np.percentile(diff, [lo, hi])
            row[f"Δ {a.upper()}-{b_.upper()}"] = point[a][i] - point[b_][i]
            row["Δ IC95"] = f"[{low:.4f}, {high:.4f}]"
            row["Significativo"] = bool(low > 0 or high < 0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Métrica")


def _fast_delong(scores_pos_first, n_pos):
    """AUCs and covariance for k models; columns are positives first, then negatives."""
    k, total = scores_pos_first.shape
    n_neg = total - n_pos
    pos = scores_pos_first[:, :n_pos]
    neg = scores_pos_first[:, n_pos:]

    tx = np.vstack([rankdata(row) for row in pos])
    ty = np.vstack([rankdata(row) for row in neg])
    tz = np.vstack([rankdata(row) for row in scores_pos_first])

    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    cov = np.atleast_2d(np.cov(v01)) / n_pos + np.atleast_2d(np.cov(v10)) / n_neg
    return aucs, cov


def delong_test(y, proba_a, proba_b):
    """Two-sided DeLong test for AUC(a) - AUC(b) on the same test users."""
    y = np.asarray(y).astype(int)
    order = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)]
    scores = np.vstack([np.asarray(proba_a), np.asarray(proba_b)])[:, order]
    aucs, cov = _fast_delong(scores, int((y == 1).sum()))

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": float(diff),
        "z": float(z),
        "p_value": float(2 * norm.sf(abs(z))),
    }
