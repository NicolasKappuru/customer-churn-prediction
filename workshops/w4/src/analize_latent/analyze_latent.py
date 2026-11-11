"""Latent space extraction, projection, and separability analysis for AE and VAE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[4]
W4_SRC = PROJECT_ROOT / "workshops" / "w4" / "src"
for _p in (str(W4_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_BATCH_SIZE = 2048


class LatentExtractor:
    """Extract AE and VAE latent representations from (x_num, x_cat) tensors."""

    def __init__(
        self,
        ae_model:     object,
        vae_model:    object,
        preprocessor: object,
        device:       torch.device | str,
    ) -> None:
        self._ae   = ae_model
        self._vae  = vae_model
        self._prep = preprocessor
        self._dev  = torch.device(device)

    def ae_latent(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> npt.NDArray[np.float32]:
        """Encode with the AE encoder; return (n, 16) float32 array."""
        self._ae.eval()
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(x_num), _BATCH_SIZE):
                bn = x_num[i : i + _BATCH_SIZE].to(self._dev)
                bc = x_cat[i : i + _BATCH_SIZE].to(self._dev)
                z = self._ae.encode(bn, bc)
                parts.append(z.cpu())
        return torch.cat(parts).numpy().astype(np.float32)

    def vae_latent(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> npt.NDArray[np.float32]:
        """Encode with the VAE encoder; return mu (n, 16) float32 — deterministic."""
        self._vae.eval()
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(x_num), _BATCH_SIZE):
                bn = x_num[i : i + _BATCH_SIZE].to(self._dev)
                bc = x_cat[i : i + _BATCH_SIZE].to(self._dev)
                mu, _ = self._vae.encode(bn, bc)
                parts.append(mu.cpu())
        return torch.cat(parts).numpy().astype(np.float32)

    @staticmethod
    def vae_active_dims(
        mu: npt.NDArray[np.float32], threshold: float = 0.01
    ) -> npt.NDArray[np.bool_]:
        """Boolean mask: True for dimensions where Var(mu) > threshold."""
        return mu.var(axis=0) > threshold


class LatentProjector:
    """2D projections of latent vectors via t-SNE and UMAP."""

    def __init__(self, seed: int = 110) -> None:
        self._seed = seed

    def tsne(self, latent: npt.NDArray) -> npt.NDArray[np.float32]:
        """Project to 2D with t-SNE (perplexity=30, init=pca)."""
        proj = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            random_state=self._seed,
            n_jobs=-1,
        ).fit_transform(latent)
        return proj.astype(np.float32)

    def umap(self, latent: npt.NDArray) -> npt.NDArray[np.float32]:
        """Project to 2D with UMAP (n_neighbors=15, min_dist=0.1)."""
        try:
            import umap
        except ImportError as e:
            raise ImportError("Install umap-learn: pip install umap-learn") from e
        proj = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            random_state=self._seed,
            n_jobs=-1,
        ).fit_transform(latent)
        return proj.astype(np.float32)


class SeparabilityAnalyzer:
    """Quantify how well a latent space separates churn from normal clients."""

    def __init__(self, seed: int = 110) -> None:
        self._seed = seed

    def silhouette(self, latent: npt.NDArray, y: npt.NDArray) -> float:
        """Silhouette score on raw latent vectors (Euclidean)."""
        return float(silhouette_score(latent, y, metric="euclidean",
                                      random_state=self._seed))

    def silhouette_standardized(self, latent: npt.NDArray, y: npt.NDArray) -> float:
        """Silhouette after StandardScaler (comparable across models with different scales)."""
        z = StandardScaler().fit_transform(latent)
        return float(silhouette_score(z, y, metric="euclidean",
                                      random_state=self._seed))

    def linear_probe_auc(self, latent: npt.NDArray, y: npt.NDArray) -> float:
        """Mean ROC-AUC of a logistic regression probe (5-fold stratified CV)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=self._seed),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self._seed)
        scores = cross_val_score(pipe, latent, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    def latent_stats(self, latent: npt.NDArray) -> pd.DataFrame:
        """Per-dimension mean and std; reveals whether the space approximates N(0,1)."""
        return pd.DataFrame({
            "dim":  np.arange(latent.shape[1]),
            "mean": latent.mean(axis=0),
            "std":  latent.std(axis=0),
        })
