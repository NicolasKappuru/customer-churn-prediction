"""Reconstruction evaluator: metrics, panels, and difference matrices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
W4_SRC = PROJECT_ROOT / "workshops" / "w4" / "src"
for _p in (str(W4_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoencoder.autoencoder_ae import Autoencoder, ReconstructionLoss
from autoencoder.preprocessor import TabularPreprocessor

_BATCH_SIZE = 2048


class ReconstructionEvaluator:
    """Compute reconstruction metrics and diagnostic tables for an AE."""

    def __init__(
        self,
        model:        Autoencoder,
        preprocessor: TabularPreprocessor,
        device:       torch.device | str,
    ) -> None:
        self._model  = model
        self._prep   = preprocessor
        self._device = torch.device(device)
        self._loss_fn = ReconstructionLoss(model.config.cat_loss_weight)

    # ------------------------------------------------------------------
    # Batched prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the model in eval mode in batches; return (num_out, cat_logits)."""
        self._model.eval()
        num_outs: list[torch.Tensor] = []
        cat_lists: list[list[torch.Tensor]] = [[] for _ in self._prep.cat_cardinalities]

        with torch.no_grad():
            for i in range(0, len(x_num), _BATCH_SIZE):
                bn = x_num[i : i + _BATCH_SIZE].to(self._device)
                bc = x_cat[i : i + _BATCH_SIZE].to(self._device)
                _, n_out, c_logits = self._model(bn, bc)
                num_outs.append(n_out.cpu())
                for j, logits in enumerate(c_logits):
                    cat_lists[j].append(logits.cpu())

        return (
            torch.cat(num_outs, dim=0),
            [torch.cat(lst, dim=0) for lst in cat_lists],
        )

    # ------------------------------------------------------------------
    # Error per client
    # ------------------------------------------------------------------

    def per_sample_error(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> npt.NDArray[np.float32]:
        """Return per-client total reconstruction error (shape: n_samples)."""
        num_out, cat_logits = self.predict(x_num, x_cat)
        total_ps, _, _ = self._loss_fn(num_out, x_num, cat_logits, x_cat, per_sample=True)
        return total_ps.numpy()

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------

    def summary_metrics(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> dict:
        """Global MSE (numeric) and per-categorical accuracy."""
        num_out, cat_logits = self.predict(x_num, x_cat)

        mse = float(torch.mean((num_out - x_num) ** 2).item())

        cat_acc: dict[str, float] = {}
        for i, (col, logits) in enumerate(zip(self._prep._categorical_cols, cat_logits)):
            preds   = logits.argmax(dim=-1)
            targets = x_cat[:, i]
            cat_acc[col] = float((preds == targets).float().mean().item())

        return {"mse_numeric": mse, "cat_accuracy": cat_acc}

    # ------------------------------------------------------------------
    # Per-feature metrics
    # ------------------------------------------------------------------

    def per_feature_metrics(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> pd.DataFrame:
        """DataFrame with columns: feature, mse, pearson_corr (standardised space).

        Only covers non-constant numeric features (constant ones are excluded from x_num).
        """
        num_out, _ = self.predict(x_num, x_cat)
        orig  = x_num.numpy()
        recon = num_out.numpy()

        rows = []
        for i, feat in enumerate(self._prep.numeric_cols_nc):
            mse = float(np.mean((orig[:, i] - recon[:, i]) ** 2))
            r, _ = pearsonr(orig[:, i], recon[:, i])
            rows.append({"feature": feat, "mse": mse, "pearson_corr": float(r)})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Metrics by class
    # ------------------------------------------------------------------

    def metrics_by_class(
        self,
        errors: npt.NDArray[np.float32],
        y:      npt.NDArray[np.int32],
    ) -> pd.DataFrame:
        """Mean and median error for churn (1) vs normal (0) clients."""
        rows = []
        for label, name in [(0, "normal"), (1, "churn")]:
            mask = y == label
            rows.append({
                "class":      name,
                "n":          int(mask.sum()),
                "mean_error": float(errors[mask].mean()),
                "median_error": float(np.median(errors[mask])),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Sample panel
    # ------------------------------------------------------------------

    def sample_panel(
        self,
        x_num:    torch.Tensor,
        x_cat:    torch.Tensor,
        df_orig:  pd.DataFrame,
    ) -> pd.DataFrame:
        """DataFrame (2*n × 59) showing original vs reconstructed for n clients."""
        num_out, cat_logits = self.predict(x_num, x_cat)
        recon_df = self._prep.inverse_transform(num_out, cat_logits)
        orig_df  = df_orig[self._prep.feature_names].reset_index(drop=True)
        recon_df = recon_df.reset_index(drop=True)
        return pd.concat([orig_df, recon_df], keys=["original", "reconstructed"])

    # ------------------------------------------------------------------
    # Categorical baseline
    # ------------------------------------------------------------------

    def categorical_baseline(self, x_cat: torch.Tensor) -> dict[str, float]:
        """Accuracy of always predicting the most frequent level seen in fit.

        Returns a dict mapping each categorical column name to its baseline accuracy
        on x_cat. Compare with summary_metrics()['cat_accuracy'] to quantify AE gain.
        """
        result: dict[str, float] = {}
        for i, col in enumerate(self._prep._categorical_cols):
            mode_idx = self._prep._cat_modes[i]
            targets  = x_cat[:, i]
            result[col] = float((targets == mode_idx).float().mean().item())
        return result

    # ------------------------------------------------------------------
    # Difference matrix
    # ------------------------------------------------------------------

    def difference_matrix(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> npt.NDArray[np.float32]:
        """Return |x_num - num_out| in standardised space; shape (n, numeric_dim)."""
        num_out, _ = self.predict(x_num, x_cat)
        return (x_num - num_out).abs().numpy()
