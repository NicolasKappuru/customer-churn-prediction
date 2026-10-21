"""Load and run the trained w2 LSTM churn model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CHECKPOINT = (
    PROJECT_ROOT
    / "workshops" / "w2" / "artifacts" / "sherlock_w2"
    / "lstm" / "checkpoints" / "lstm.pt"
)
_SCALER = (
    PROJECT_ROOT
    / "workshops" / "w2" / "artifacts" / "sherlock_w2"
    / "feature_preparation" / "preprocessing" / "development.json"
)

_CHANNELS = [
    "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs",
    "active",
    "tx_count", "tx_paid_sum", "tx_plan_days_sum", "tx_cancel_count",
    "tx_auto_renew_count", "tx_state_known", "tx_last_is_cancel",
    "tx_last_is_auto_renew", "tx_days_since_last", "tx_days_to_expire",
    "tx_expiry_known", "tx_state_ambiguous",
]


class _LSTM(nn.Module):
    """Inline copy of the w2 architecture — no dependency on w2/src."""

    def __init__(self, hidden_dim: int, layer_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(20, hidden_dim, layer_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, dtype=x.dtype)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.bn(out[:, -1, :]))


class NeuralNetworkModel:
    """Load the w2 LSTM checkpoint and predict churn for a 90-day sequence.

    Input: (90, 20) or (1, 90, 20) float32 array produced by SequenceBuilder.
    The scaler fitted on the development set is applied internally.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = _CHECKPOINT,
        scaler_path: str | Path = _SCALER,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        params = ckpt["params"]
        self._net = _LSTM(int(params["hidden_dim"]), int(params["layer_dim"]))
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
        self.threshold = float(ckpt["threshold"])
        self.epochs = int(ckpt["epochs"])
        self.params = params

        scaler_path = Path(scaler_path)
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        with scaler_path.open() as f:
            self._state = json.load(f)
        self._ch = {name: i for i, name in enumerate(_CHANNELS)}

    # ------------------------------------------------------------------
    # Scaling (mirrors transform.py from w2/src)
    # ------------------------------------------------------------------

    def _scale(self, seq: np.ndarray) -> np.ndarray:
        """Apply development scaler to a (1, 90, 20) float32 array."""
        raw = np.asarray(seq, dtype=np.float32)
        out = raw.copy()

        ls = self._state["listening"]
        active_ch = self._ch[ls["active_channel"]]
        active = raw[:, :, active_ch] > 0
        n_idx = [self._ch[n] for n in ls["numeric_channels"]]
        upper = np.array(ls["upper"], dtype=np.float32)
        mean = np.array(ls["mean"], dtype=np.float32)
        std = np.array(ls["std"], dtype=np.float32)
        vals = np.minimum(raw[:, :, n_idx], upper)
        vals = (vals - mean) / std
        for pos, ch in enumerate(n_idx):
            out[:, :, ch] = np.where(active, vals[:, :, pos], 0.0)

        for name, p in self._state["transaction"]["parameters"].items():
            if p["encoding"] == "flag":
                continue
            ch = self._ch[name]
            mask_ch = self._ch[p["mask"]]
            mask = raw[:, :, mask_ch] > 0
            col = np.zeros(mask.shape, dtype=np.float32)
            if mask.any():
                v = raw[:, :, ch][mask].astype(np.float32)
                if p["lower"] is not None:
                    v = np.clip(v, p["lower"], p["upper"])
                col[mask] = (v - p["mean"]) / p["std"]
            out[:, :, ch] = col
        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict_probability(self, sequence) -> float:
        """Return churn probability for a (90, 20) or (1, 90, 20) array."""
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim == 2:
            seq = seq[None]
        scaled = self._scale(seq)
        with torch.no_grad():
            logit = self._net(torch.from_numpy(scaled))
            return float(torch.sigmoid(logit).squeeze())

    def predict(self, sequence) -> str:
        """Return 'Churn' or 'No churn' using the threshold from tuning."""
        return "Churn" if self.predict_probability(sequence) >= self.threshold else "No churn"
