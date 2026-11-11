"""Anomaly detection with AE and VAE reconstruction scores."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
W4_SRC = PROJECT_ROOT / "workshops" / "w4" / "src"
for _p in (str(W4_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass(frozen=True)
class AnomalyConfig:
    """Configuration for anomaly scoring and threshold selection."""
    threshold_criterion: str        = "max_f1"
    score_names:         tuple[str, ...] = ("AE_recon", "VAE_recon", "VAE_recon_kl")
    batch_size:          int        = 4096


class AnomalyScorer:
    """Compute per-client anomaly scores from AE and VAE reconstruction errors."""

    def __init__(
        self,
        ae_model:  object,   # Autoencoder
        vae_model: object,   # VAE
        device:    torch.device | str,
        config:    AnomalyConfig,
    ) -> None:
        self._ae   = ae_model
        self._vae  = vae_model
        self._dev  = torch.device(device)
        self._cfg  = config

    def scores(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Return {name: per-client score} for all three score types."""
        ae_recon              = self._ae_scores(x_num, x_cat)
        vae_recon, vae_rec_kl = self._vae_scores(x_num, x_cat)
        return {
            "AE_recon":      ae_recon,
            "VAE_recon":     vae_recon,
            "VAE_recon_kl":  vae_rec_kl,
        }

    # ------------------------------------------------------------------

    def _ae_scores(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> npt.NDArray[np.float32]:
        self._ae.eval()
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(x_num), self._cfg.batch_size):
                bn = x_num[i : i + self._cfg.batch_size].to(self._dev)
                bc = x_cat[i : i + self._cfg.batch_size].to(self._dev)
                _, num_out, cat_logits = self._ae(bn, bc)
                parts.append(self._recon_per_sample(num_out, bn, cat_logits, bc).cpu())
        return torch.cat(parts).numpy()

    def _vae_scores(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        self._vae.eval()
        recon_parts: list[torch.Tensor] = []
        kl_parts:    list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(x_num), self._cfg.batch_size):
                bn = x_num[i : i + self._cfg.batch_size].to(self._dev)
                bc = x_cat[i : i + self._cfg.batch_size].to(self._dev)
                _, mu, logvar, num_out, cat_logits = self._vae(bn, bc)
                recon_i = self._recon_per_sample(num_out, bn, cat_logits, bc)
                kl_i    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
                recon_parts.append(recon_i.cpu())
                kl_parts.append(kl_i.cpu())
        recon = torch.cat(recon_parts).numpy()
        kl    = torch.cat(kl_parts).numpy()
        return recon, recon + kl

    @staticmethod
    def _recon_per_sample(
        num_out:    torch.Tensor,
        x_num:      torch.Tensor,
        cat_logits: list[torch.Tensor],
        x_cat:      torch.Tensor,
    ) -> torch.Tensor:
        """Sum of squared errors (numeric) + sum of CE (categorical) per sample."""
        num_err = F.mse_loss(num_out, x_num, reduction="none").sum(dim=-1)
        cat_err = torch.stack(
            [F.cross_entropy(logits, x_cat[:, k], reduction="none")
             for k, logits in enumerate(cat_logits)],
            dim=1,
        ).sum(dim=1)
        return num_err + cat_err


class ThresholdSelector:
    """Select the score threshold that maximises F1 for the churn class on a tune set."""

    def __init__(self, criterion: str = "max_f1") -> None:
        self.criterion = criterion
        self.threshold_: float | None = None

    def fit(self, scores: npt.NDArray, y: npt.NDArray) -> float:
        """Return threshold maximising churn F1; also stored in self.threshold_."""
        precision, recall, thresholds = precision_recall_curve(y, scores)
        # precision/recall have len n+1; thresholds has len n
        f1 = np.where(
            (precision[:-1] + recall[:-1]) > 0,
            2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
            0.0,
        )
        best_idx = int(np.argmax(f1))
        self.threshold_ = float(thresholds[best_idx])
        return self.threshold_


class AnomalyEvaluator:
    """Compute ranking and decision metrics for an anomaly score."""

    def ranking_metrics(
        self, scores: npt.NDArray, y: npt.NDArray
    ) -> dict[str, float]:
        """AUC-ROC, average precision, and churn prevalence."""
        return {
            "roc_auc":           float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "churn_prevalence":  float(y.mean()),
        }

    def decision_metrics(
        self,
        scores:    npt.NDArray,
        y:         npt.NDArray,
        threshold: float,
    ) -> dict:
        """Full confusion-matrix metrics at a given threshold."""
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pred_rate = float(y_pred.mean())
        return {
            "threshold":             float(threshold),
            "precision_churn":       float(prec),
            "recall_churn":          float(rec),
            "f1_churn":              float(f1),
            "specificity":           float(spec),
            "balanced_accuracy":     float(balanced_accuracy_score(y, y_pred)),
            "mcc":                   float(matthews_corrcoef(y, y_pred)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "predicted_churn_rate":  pred_rate,
        }

    def curves(
        self, scores: npt.NDArray, y: npt.NDArray
    ) -> dict[str, npt.NDArray]:
        """ROC (fpr, tpr) and PR (precision, recall) curve arrays."""
        fpr, tpr, _ = roc_curve(y, scores)
        prec, rec, _ = precision_recall_curve(y, scores)
        return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}


class CumulativeResults:
    """Merge W1/W2 supervised baselines with W4 anomaly detector results."""

    _KEEP = [
        "model", "family", "workshop", "supervision",
        "roc_auc", "average_precision",
        "precision_churn", "recall_churn", "f1_churn",
        "threshold",
    ]

    def load_w1(self, path: Path | str) -> pd.DataFrame:
        """Load W1 test_metrics.csv and normalise columns."""
        df = pd.read_csv(path)
        df = df.rename(columns={"model_id": "model"})
        df["workshop"]   = "W1"
        df["supervision"] = "supervised"
        return df[self._KEEP]

    def load_w2(self, path: Path | str) -> pd.DataFrame:
        """Load W2 test_metrics.csv and normalise columns."""
        df = pd.read_csv(path)
        df = df.rename(columns={"model_id": "model"})
        df["workshop"]    = "W2"
        df["supervision"] = "supervised"
        return df[self._KEEP]

    def build(self, anomaly_rows: pd.DataFrame) -> pd.DataFrame:
        """Combine W1 + W2 baselines with anomaly results, sorted by roc_auc desc."""
        w1_path = (
            PROJECT_ROOT / "workshops" / "w1" / "artifacts"
            / "sherlock_w1" / "metrics" / "test_metrics.csv"
        )
        w2_path = (
            PROJECT_ROOT / "workshops" / "w2" / "artifacts"
            / "sherlock_w2" / "lstm" / "evaluation" / "metrics" / "test_metrics.csv"
        )

        frames: list[pd.DataFrame] = []
        for loader, path in [(self.load_w1, w1_path), (self.load_w2, w2_path)]:
            if Path(path).is_file():
                frames.append(loader(path))

        frames.append(anomaly_rows[self._KEEP])
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("roc_auc", ascending=False).reset_index(drop=True)
