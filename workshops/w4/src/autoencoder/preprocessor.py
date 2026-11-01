"""Tabular feature preprocessor for the W4 autoencoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


_SKEWNESS_THRESHOLD = 1.0
_CLIP_LOW  = 0.001
_CLIP_HIGH = 0.999


class TabularPreprocessor:
    """Fit-once preprocessor: impute → signed-log1p → clip → standardize (numeric);
    vocabulary encoding (categorical).  Constant numeric columns are detected in fit,
    excluded from x_num (and from the model), and reinserted in inverse_transform.
    Designed to be saved inside checkpoints."""

    def __init__(self, numeric_cols: list[str], categorical_cols: list[str]) -> None:
        self._numeric_cols     = list(numeric_cols)
        self._categorical_cols = list(categorical_cols)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        """Learn all statistics from frame (fit_normal rows only)."""
        raw = self._to_float64(frame)

        # 1. Imputation medians (ignore NaN)
        self._medians: np.ndarray = np.nanmedian(raw, axis=0)

        # 2. Fill NaN for downstream stats
        filled = np.where(np.isfinite(raw), raw, self._medians)

        # 3. Detect skewed columns
        col_skew = pd.DataFrame(filled, columns=self._numeric_cols).skew().values
        self._log_mask: np.ndarray = np.abs(col_skew) > _SKEWNESS_THRESHOLD

        # 4. Apply signed log1p on skewed columns
        log_filled = filled.copy()
        log_filled[:, self._log_mask] = (
            np.sign(filled[:, self._log_mask])
            * np.log1p(np.abs(filled[:, self._log_mask]))
        )

        # 5. Quantile clip bounds
        self._clip_lo: np.ndarray = np.quantile(log_filled, _CLIP_LOW,  axis=0)
        self._clip_hi: np.ndarray = np.quantile(log_filled, _CLIP_HIGH, axis=0)
        clipped = np.clip(log_filled, self._clip_lo, self._clip_hi)

        # 6. Standardisation parameters
        self._mean: np.ndarray = clipped.mean(axis=0)
        std = clipped.std(axis=0)

        # 7. Detect constant columns (std == 0 after clipping) and record their
        #    pre-transform values so inverse_transform can reinsert them.
        self._const_mask: np.ndarray = std == 0
        self._non_const_mask: np.ndarray = ~self._const_mask
        # Store original-scale (pre-log) values for constant columns.
        self._const_orig_values: np.ndarray = filled[0, self._const_mask].copy()

        self._std: np.ndarray = np.where(std == 0, 1.0, std)

        # 8. Binary column detection on original filled values (before any transform)
        self._binary_mask: np.ndarray = np.array(
            [np.all(np.isin(np.unique(filled[:, i]), [0.0, 1.0])) for i in range(filled.shape[1])],
            dtype=bool,
        )

        # 9. Categorical vocabularies + mode index (most frequent level in fit)
        self._cat_vocabs: list[dict] = []
        self._cat_modes:  list[int]  = []
        for col in self._categorical_cols:
            vals = _flatten_cat(frame[col])
            vocab: dict = {"__MISSING__": 0}
            for v in sorted(v for v in set(vals) if v != "__MISSING__"):
                vocab[v] = len(vocab)
            self._cat_vocabs.append(vocab)

            idx_arr = np.array([vocab.get(v, vocab["__MISSING__"]) for v in vals], dtype=np.int64)
            self._cat_modes.append(int(np.bincount(idx_arr).argmax()))

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x_num float32, x_cat int64).

        x_num contains only non-constant numeric columns (shape: n × numeric_dim).
        """
        arr = self._to_float64(frame)
        arr = np.where(np.isfinite(arr), arr, self._medians)

        arr[:, self._log_mask] = (
            np.sign(arr[:, self._log_mask])
            * np.log1p(np.abs(arr[:, self._log_mask]))
        )
        arr = np.clip(arr, self._clip_lo, self._clip_hi)
        arr = (arr - self._mean) / self._std

        # Exclude constant columns from the model's input
        x_num = torch.tensor(arr[:, self._non_const_mask].astype(np.float32))

        cat_cols = []
        for col, vocab in zip(self._categorical_cols, self._cat_vocabs):
            vals = _flatten_cat(frame[col])
            idx = np.array([vocab.get(v, vocab["__MISSING__"]) for v in vals], dtype=np.int64)
            cat_cols.append(idx)
        x_cat = torch.tensor(np.stack(cat_cols, axis=1))

        return x_num, x_cat

    # ------------------------------------------------------------------
    # Inverse transform
    # ------------------------------------------------------------------

    def inverse_transform(
        self,
        x_num: torch.Tensor,
        cat_logits_or_idx: list[torch.Tensor],
    ) -> pd.DataFrame:
        """Reconstruct original-scale DataFrame from non-constant numerics and cat logits.

        Returns all 59 columns; constant columns are reinserted with their fit-time value.
        """
        arr = x_num.detach().cpu().numpy().astype(np.float64)

        # De-standardise using non-constant stats only
        mean_nc = self._mean[self._non_const_mask]
        std_nc  = self._std[self._non_const_mask]
        arr = arr * std_nc + mean_nc

        # Inverse signed log1p for non-constant columns
        lm_nc = self._log_mask[self._non_const_mask]
        arr[:, lm_nc] = np.sign(arr[:, lm_nc]) * np.expm1(np.abs(arr[:, lm_nc]))

        # Rebuild full numeric array, reinserting constant columns
        n = arr.shape[0]
        full = np.zeros((n, len(self._numeric_cols)), dtype=np.float64)
        full[:, self._non_const_mask] = arr
        full[:, self._const_mask]     = self._const_orig_values  # broadcast over rows

        # Round binary columns
        full[:, self._binary_mask] = np.round(np.clip(full[:, self._binary_mask], 0.0, 1.0))

        result = pd.DataFrame(full, columns=self._numeric_cols)

        for i, (col, vocab) in enumerate(zip(self._categorical_cols, self._cat_vocabs)):
            t = cat_logits_or_idx[i]
            idx = t.argmax(dim=-1).cpu().numpy() if t.ndim == 2 else t.cpu().numpy()
            inv = {v: k for k, v in vocab.items()}
            result[col] = [inv.get(int(j), "__MISSING__") for j in idx]

        return result[self._numeric_cols + self._categorical_cols]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise state for embedding in a checkpoint."""
        return {
            "numeric_cols":      self._numeric_cols,
            "categorical_cols":  self._categorical_cols,
            "medians":           self._medians.tolist(),
            "log_mask":          self._log_mask.tolist(),
            "clip_lo":           self._clip_lo.tolist(),
            "clip_hi":           self._clip_hi.tolist(),
            "mean":              self._mean.tolist(),
            "std":               self._std.tolist(),
            "binary_mask":       self._binary_mask.tolist(),
            "const_mask":        self._const_mask.tolist(),
            "non_const_mask":    self._non_const_mask.tolist(),
            "const_orig_values": self._const_orig_values.tolist(),
            "cat_vocabs":        self._cat_vocabs,
            "cat_modes":         self._cat_modes,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "TabularPreprocessor":
        """Restore a fitted preprocessor from a state dict."""
        obj = cls(state["numeric_cols"], state["categorical_cols"])
        obj._medians           = np.array(state["medians"])
        obj._log_mask          = np.array(state["log_mask"],       dtype=bool)
        obj._clip_lo           = np.array(state["clip_lo"])
        obj._clip_hi           = np.array(state["clip_hi"])
        obj._mean              = np.array(state["mean"])
        obj._std               = np.array(state["std"])
        obj._binary_mask       = np.array(state["binary_mask"],    dtype=bool)
        obj._const_mask        = np.array(state["const_mask"],     dtype=bool)
        obj._non_const_mask    = np.array(state["non_const_mask"], dtype=bool)
        obj._const_orig_values = np.array(state["const_orig_values"])
        obj._cat_vocabs        = state["cat_vocabs"]
        obj._cat_modes         = state["cat_modes"]
        obj._fitted            = True
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def numeric_dim(self) -> int:
        """Number of non-constant numeric features (= x_num width)."""
        return int(self._non_const_mask.sum())

    @property
    def numeric_dim_all(self) -> int:
        """Total numeric columns including constants (= 56)."""
        return len(self._numeric_cols)

    @property
    def n_const(self) -> int:
        """Number of constant numeric columns excluded from x_num."""
        return int(self._const_mask.sum())

    @property
    def cat_cardinalities(self) -> tuple[int, ...]:
        return tuple(len(v) for v in self._cat_vocabs)

    @property
    def input_dim(self) -> int:
        return self.numeric_dim + sum(self.cat_cardinalities)

    @property
    def feature_names(self) -> list[str]:
        """All 59 original feature names (numeric + categorical)."""
        return self._numeric_cols + self._categorical_cols

    @property
    def numeric_cols_nc(self) -> list[str]:
        """Numeric column names excluding constant columns."""
        return [c for c, m in zip(self._numeric_cols, self._non_const_mask) if m]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_float64(self, frame: pd.DataFrame) -> np.ndarray:
        arr = frame[self._numeric_cols].to_numpy(dtype=np.float64)
        arr[~np.isfinite(arr)] = np.nan
        return arr


def _flatten_cat(series: pd.Series) -> list:
    """Convert CategoricalDtype (possibly with float codes) to plain Python values."""
    if hasattr(series, "cat"):
        series = series.astype(object)
    return [v if pd.notna(v) else "__MISSING__" for v in series]
