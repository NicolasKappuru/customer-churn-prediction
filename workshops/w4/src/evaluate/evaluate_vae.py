"""VAE evaluator: reconstruction, KL diagnostics, latent analysis, generative sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
W4_SRC = PROJECT_ROOT / "workshops" / "w4" / "src"
for _p in (str(W4_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoencoder.autoencoder_vae import VAE, ELBOLoss
from autoencoder.preprocessor import TabularPreprocessor
from evaluate.evaluate_ae import ReconstructionEvaluator

_BATCH_SIZE = 2048


class VAEEvaluator:
    """Evaluator for a trained VAE; composes ReconstructionEvaluator for shared methods."""

    def __init__(
        self,
        model:        VAE,
        preprocessor: TabularPreprocessor,
        device:       torch.device | str,
    ) -> None:
        self._model  = model
        self._prep   = preprocessor
        self._device = torch.device(device)
        self._loss_fn = ELBOLoss()

        # Wrap AE evaluator but swap the internal model for the VAE decode path
        self._ae_eval = _VAEAsAE(model, preprocessor, device)

    # ------------------------------------------------------------------
    # Delegates to ReconstructionEvaluator (shared AE/VAE metrics)
    # ------------------------------------------------------------------

    def predict(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return (num_out, cat_logits, mu, logvar) using z = mu (deterministic)."""
        self._model.eval()
        num_outs:   list[torch.Tensor] = []
        cat_lists:  list[list[torch.Tensor]] = [[] for _ in self._prep.cat_cardinalities]
        mu_list:    list[torch.Tensor] = []
        logvar_list: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(x_num), _BATCH_SIZE):
                bn = x_num[i : i + _BATCH_SIZE].to(self._device)
                bc = x_cat[i : i + _BATCH_SIZE].to(self._device)
                _, mu, logvar, n_out, c_logits = self._model(bn, bc)
                num_outs.append(n_out.cpu())
                for j, logits in enumerate(c_logits):
                    cat_lists[j].append(logits.cpu())
                mu_list.append(mu.cpu())
                logvar_list.append(logvar.cpu())

        return (
            torch.cat(num_outs, dim=0),
            [torch.cat(lst, dim=0) for lst in cat_lists],
            torch.cat(mu_list, dim=0),
            torch.cat(logvar_list, dim=0),
        )

    def summary_metrics(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> dict:
        """Global MSE (numeric) and per-categorical accuracy."""
        return self._ae_eval.summary_metrics(x_num, x_cat)

    def per_feature_metrics(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> pd.DataFrame:
        """DataFrame: feature, mse, pearson_corr."""
        return self._ae_eval.per_feature_metrics(x_num, x_cat)

    def metrics_by_class(
        self,
        errors: npt.NDArray[np.float32],
        y:      npt.NDArray[np.int32],
    ) -> pd.DataFrame:
        return self._ae_eval.metrics_by_class(errors, y)

    def sample_panel(
        self,
        x_num:   torch.Tensor,
        x_cat:   torch.Tensor,
        df_orig: pd.DataFrame,
    ) -> pd.DataFrame:
        return self._ae_eval.sample_panel(x_num, x_cat, df_orig)

    def difference_matrix(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> npt.NDArray[np.float32]:
        return self._ae_eval.difference_matrix(x_num, x_cat)

    def categorical_baseline(self, x_cat: torch.Tensor) -> dict[str, float]:
        return self._ae_eval.categorical_baseline(x_cat)

    # ------------------------------------------------------------------
    # VAE-specific metrics
    # ------------------------------------------------------------------

    def kl_per_sample(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> npt.NDArray[np.float32]:
        """KL divergence per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))."""
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        return kl.numpy()

    def anomaly_scores(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Per-sample anomaly scores: reconstruction and reconstruction + KL."""
        num_out, cat_logits, mu, logvar = self.predict(x_num, x_cat)
        _, recon_i, kl_i, _ = self._loss_fn(
            num_out, x_num, cat_logits, x_cat, mu, logvar,
            beta=1.0, reduction="none",
        )
        recon_arr = recon_i.numpy()
        kl_arr    = kl_i.numpy()
        return {
            "recon":         recon_arr,
            "recon_plus_kl": recon_arr + kl_arr,
        }

    def kl_per_dimension(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> npt.NDArray[np.float32]:
        """Mean KL per latent dimension (z_dim values) for posterior collapse analysis."""
        kl_dims = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl_dims.mean(dim=0).numpy()

    def latent_separability(
        self,
        x_num:       torch.Tensor,
        x_cat:       torch.Tensor,
        y:           npt.NDArray[np.int32],
        sample_size: int = 20_000,
        seed:        int = 110,
    ) -> float:
        """Silhouette score of mu on a stratified subsample (churn vs normal)."""
        from sklearn.metrics import silhouette_score

        _, _, mu, _ = self.predict(x_num, x_cat)
        mu_np = mu.numpy()

        rng    = np.random.default_rng(seed)
        n      = len(y)
        actual = min(sample_size, n)

        # Stratified subsample
        classes, counts = np.unique(y, return_counts=True)
        idx_list: list[npt.NDArray] = []
        for cls, cnt in zip(classes, counts):
            cls_idx   = np.where(y == cls)[0]
            n_draw    = max(1, int(round(actual * cnt / n)))
            drawn     = rng.choice(cls_idx, size=min(n_draw, len(cls_idx)), replace=False)
            idx_list.append(drawn)
        idx = np.concatenate(idx_list)

        return float(silhouette_score(mu_np[idx], y[idx]))

    def generate_samples(self, n: int) -> pd.DataFrame:
        """Decode z ~ N(0, I) and inverse-transform to 59 original-scale columns."""
        self._model.eval()
        num_out, cat_logits = self._model.sample(n, self._device)
        return self._prep.inverse_transform(num_out.cpu(), [l.cpu() for l in cat_logits])


# ---------------------------------------------------------------------------
# Internal helper: adapts VAE to look like Autoencoder for ReconstructionEvaluator
# ---------------------------------------------------------------------------

class _VAEAsAE:
    """Thin wrapper that makes VAE.predict() look like AE for ReconstructionEvaluator."""

    def __init__(
        self,
        model:        VAE,
        preprocessor: TabularPreprocessor,
        device:       torch.device | str,
    ) -> None:
        from autoencoder.autoencoder_ae import ReconstructionLoss

        class _FakeAE:
            def __init__(self, vae: VAE) -> None:
                self._vae = vae
                self.config = vae.config  # duck-typed: cat_cardinalities etc.

            def eval(self) -> "_FakeAE":
                self._vae.eval()
                return self

            def __call__(self, x_num: torch.Tensor, x_cat: torch.Tensor):
                _, mu, _, n_out, c_logits = self._vae(x_num, x_cat)
                return mu, n_out, c_logits  # (z, num_out, cat_logits) shape expected by AE eval

        fake = _FakeAE(model)
        self._inner = ReconstructionEvaluator(fake, preprocessor, device)  # type: ignore[arg-type]

    def summary_metrics(self, x_num, x_cat):
        return self._inner.summary_metrics(x_num, x_cat)

    def per_feature_metrics(self, x_num, x_cat):
        return self._inner.per_feature_metrics(x_num, x_cat)

    def metrics_by_class(self, errors, y):
        return self._inner.metrics_by_class(errors, y)

    def sample_panel(self, x_num, x_cat, df_orig):
        return self._inner.sample_panel(x_num, x_cat, df_orig)

    def difference_matrix(self, x_num, x_cat):
        return self._inner.difference_matrix(x_num, x_cat)

    def categorical_baseline(self, x_cat):
        return self._inner.categorical_baseline(x_cat)
