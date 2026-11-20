"""Side-by-side reconstruction comparator for AE vs VAE."""

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

from autoencoder.preprocessor import TabularPreprocessor

_BATCH_SIZE = 2048


class ReconstructionComparator:
    """Compare AE and VAE reconstruction quality on the same test set."""

    def __init__(
        self,
        models:       dict,   # {"AE": ReconstructionEvaluator, "VAE": VAEEvaluator}
        preprocessor: TabularPreprocessor,
        device:       torch.device | str,
    ) -> None:
        self._models = models
        self._prep   = preprocessor
        self._device = torch.device(device)
        self._cache: dict[str, tuple[torch.Tensor, list[torch.Tensor]]] = {}

    # ------------------------------------------------------------------
    # Prediction (cached)
    # ------------------------------------------------------------------

    def predict_all(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, list[torch.Tensor]]]:
        """Run each model once; return {name: (num_out, cat_logits)}."""
        for name, ev in self._models.items():
            if name not in self._cache:
                result = ev.predict(x_num, x_cat)
                # VAEEvaluator returns (num_out, cat_logits, mu, logvar)
                self._cache[name] = result[0], result[1]
        return dict(self._cache)

    def per_sample_mse(
        self,
        name:  str,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> npt.NDArray[np.float32]:
        """MSE per sample (mean over 55 numerics) for a given model."""
        preds = self.predict_all(x_num, x_cat)
        num_out, _ = preds[name]
        return (num_out - x_num).pow(2).mean(dim=-1).numpy()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        y:     npt.NDArray,
    ) -> pd.DataFrame:
        """One row per model: mse_global, mse_normal, mse_churn, ratio, pearson, cat acc."""
        preds = self.predict_all(x_num, x_cat)
        rows: list[dict] = []

        for name, (num_out, cat_logits) in preds.items():
            mse_ps   = (num_out - x_num).pow(2).mean(dim=-1).numpy()
            mask_n   = y == 0
            mask_c   = y == 1
            mse_g    = float(mse_ps.mean())
            mse_n    = float(mse_ps[mask_n].mean())
            mse_c    = float(mse_ps[mask_c].mean()) if mask_c.any() else float("nan")
            ratio    = mse_c / mse_n if mse_n > 0 else float("nan")

            pearson_vals = self._pearson_per_feature(num_out, x_num)
            mean_p  = float(np.nanmean(pearson_vals))
            med_p   = float(np.nanmedian(pearson_vals))

            cat_acc = self._cat_accuracy(cat_logits, x_cat)

            rows.append({
                "model":                     name,
                "mse_global":                mse_g,
                "mse_normal":                mse_n,
                "mse_churn":                 mse_c,
                "churn_to_normal_ratio":     ratio,
                "mean_pearson":              mean_p,
                "median_pearson":            med_p,
                "acc_city":                  cat_acc[0],
                "acc_registered_via":        cat_acc[1],
                "acc_last_payment_method_id": cat_acc[2],
            })

        return pd.DataFrame(rows)

    def baseline_row(self, x_cat: torch.Tensor) -> pd.DataFrame:
        """Most-frequent-level baseline accuracy row (from fit modes)."""
        ev = next(iter(self._models.values()))
        bl = ev.categorical_baseline(x_cat)
        cols = list(bl.keys())
        row = {
            "model":                     "Most-frequent baseline",
            "mse_global":                float("nan"),
            "mse_normal":                float("nan"),
            "mse_churn":                 float("nan"),
            "churn_to_normal_ratio":     float("nan"),
            "mean_pearson":              float("nan"),
            "median_pearson":            float("nan"),
            "acc_city":                  bl.get(cols[0], float("nan")) if len(cols) > 0 else float("nan"),
            "acc_registered_via":        bl.get(cols[1], float("nan")) if len(cols) > 1 else float("nan"),
            "acc_last_payment_method_id": bl.get(cols[2], float("nan")) if len(cols) > 2 else float("nan"),
        }
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Error by class
    # ------------------------------------------------------------------

    def error_by_class(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        y:     npt.NDArray,
    ) -> pd.DataFrame:
        """Long-form DataFrame: model, class, mse_per_sample (for violin/box plot)."""
        preds = self.predict_all(x_num, x_cat)
        rows: list[dict] = []
        for name, (num_out, _) in preds.items():
            mse_ps = (num_out - x_num).pow(2).mean(dim=-1).numpy()
            for label, cls_name in [(0, "normal"), (1, "churn")]:
                mask = y == label
                for v in mse_ps[mask]:
                    rows.append({"model": name, "class": cls_name, "mse_per_sample": float(v)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Per-feature comparison
    # ------------------------------------------------------------------

    def per_feature_comparison(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> pd.DataFrame:
        """DataFrame: feature, mse_ae, mse_vae, corr_ae, corr_vae, mse_ratio_vae_ae."""
        preds = self.predict_all(x_num, x_cat)
        feats = self._prep.numeric_cols_nc
        orig  = x_num.numpy()

        stats: dict[str, dict] = {}
        for name, (num_out, _) in preds.items():
            recon = num_out.numpy()
            for i, feat in enumerate(feats):
                if feat not in stats:
                    stats[feat] = {}
                mse = float(np.mean((orig[:, i] - recon[:, i]) ** 2))
                corr = self._pearson_safe(orig[:, i], recon[:, i])
                stats[feat][f"mse_{name.lower()}"]  = mse
                stats[feat][f"corr_{name.lower()}"] = corr

        rows = []
        for feat, d in stats.items():
            mse_ae  = d.get("mse_ae",  float("nan"))
            mse_vae = d.get("mse_vae", float("nan"))
            ratio   = mse_vae / mse_ae if mse_ae > 0 else float("nan")
            rows.append({
                "feature":         feat,
                "mse_ae":          mse_ae,
                "mse_vae":         mse_vae,
                "corr_ae":         d.get("corr_ae",  float("nan")),
                "corr_vae":        d.get("corr_vae", float("nan")),
                "mse_ratio_vae_ae": ratio,
            })

        return pd.DataFrame(rows).sort_values("mse_vae", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Panel helpers
    # ------------------------------------------------------------------

    def select_panel_rows(
        self,
        y:        npt.NDArray,
        n_normal: int = 4,
        n_churn:  int = 4,
        seed:     int = 110,
    ) -> npt.NDArray[np.intp]:
        """Return stratified index: n_normal no-churn + n_churn churn clients."""
        rng     = np.random.default_rng(seed)
        idx_n   = np.where(y == 0)[0]
        idx_c   = np.where(y == 1)[0]
        chosen  = np.concatenate([
            rng.choice(idx_n, size=min(n_normal, len(idx_n)), replace=False),
            rng.choice(idx_c, size=min(n_churn,  len(idx_c)), replace=False),
        ])
        return chosen

    def panel_table(
        self,
        rows:          npt.NDArray[np.intp],
        x_num:         torch.Tensor,
        x_cat:         torch.Tensor,
        df_test_orig:  pd.DataFrame,
    ) -> pd.DataFrame:
        """DataFrame (clients × 3) × features in original scale: original, AE, VAE."""
        preds   = self.predict_all(x_num, x_cat)
        x_num_r = x_num[rows]
        x_cat_r = x_cat[rows]

        orig_df  = df_test_orig.iloc[rows][self._prep.feature_names].reset_index(drop=True)
        orig_df["source"] = "original"

        frames = [orig_df]
        for name, (num_out, cat_logits) in preds.items():
            rec_df = self._prep.inverse_transform(num_out[rows], [l[rows] for l in cat_logits])
            rec_df["source"] = name
            frames.append(rec_df)

        return pd.concat(frames, ignore_index=True)

    def difference_matrices(
        self,
        rows:  npt.NDArray[np.intp],
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> dict[str, pd.DataFrame]:
        """Return {name: DataFrame clients × features} with |x - x̂| in standardised space."""
        preds  = self.predict_all(x_num, x_cat)
        feats  = self._prep.numeric_cols_nc
        x_sub  = x_num[rows].numpy()
        result: dict[str, pd.DataFrame] = {}
        for name, (num_out, _) in preds.items():
            diff = np.abs(x_sub - num_out[rows].numpy())
            result[name] = pd.DataFrame(diff, columns=feats)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pearson_per_feature(
        self, num_out: torch.Tensor, x_num: torch.Tensor
    ) -> npt.NDArray[np.float64]:
        orig  = x_num.numpy()
        recon = num_out.numpy()
        vals  = np.full(orig.shape[1], np.nan)
        for i in range(orig.shape[1]):
            vals[i] = self._pearson_safe(orig[:, i], recon[:, i])
        return vals

    @staticmethod
    def _pearson_safe(a: npt.NDArray, b: npt.NDArray) -> float:
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return float("nan")
        r, _ = pearsonr(a, b)
        return float(r)

    def _cat_accuracy(
        self, cat_logits: list[torch.Tensor], x_cat: torch.Tensor
    ) -> list[float]:
        accs = []
        for i, logits in enumerate(cat_logits):
            preds = logits.argmax(dim=-1)
            accs.append(float((preds == x_cat[:, i]).float().mean().item()))
        return accs
