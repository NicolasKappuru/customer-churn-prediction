"""GRU test of notebook 07, section 7.5 (manual segment, listening sequence aligned to the expiry).

Reads ``cache/07_manual_seq_k7.npz`` (written by the notebook: ``X`` = customers x 90 days x 8
channels ending 7 days before each customer's expiry, ``y`` = is_churn) and writes
``cache/07_manual_gru.csv`` with the per-fold ROC AUC / average precision of a small GRU, using the
same 5 folds as the notebook. It is a separate script only because it needs PyTorch, which the
environment that runs the notebooks does not have:

    python 07_manual_gru.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 110                                   # same as _common.SEED
CACHE = Path(__file__).resolve().parent / "cache"
RUNS = {"GRU, 2 channels (active, seconds)": [0, 7], "GRU, 8 channels (active + 7 log counters)": list(range(8))}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, channels, hidden=48):
        super().__init__()
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.out(self.gru(x)[1][-1]).squeeze(-1)


def folds(X, y, epochs=8):
    rows = []
    for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
        torch.manual_seed(SEED + f)
        model = Net(X.shape[2]).to(DEV)
        opt = torch.optim.Adam(model.parameters(), 2e-3)
        Xt, yt = torch.from_numpy(X[tr]).to(DEV), torch.from_numpy(y[tr]).to(DEV)
        for _ in range(epochs):
            model.train()
            for b in torch.randperm(len(tr), device=DEV).split(512):
                opt.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(model(Xt[b]), yt[b]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.cat([model(torch.from_numpy(X[te[i:i + 4096]]).to(DEV)) for i in range(0, len(te), 4096)]).cpu().numpy()
        rows.append({"fold": f, "roc_auc": roc_auc_score(y[te], p), "avg_precision": average_precision_score(y[te], p)})
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    z = np.load(CACHE / "07_manual_seq_k7.npz")
    X, y = z["X"].astype(np.float32), z["y"].astype(np.float32)
    out = []
    for name, ch in RUNS.items():
        print(name, X.shape, DEV, flush=True)
        out += [{"model": name, **r} for r in folds(np.ascontiguousarray(X[:, :, ch]), y)]
    pd.DataFrame(out).to_csv(CACHE / "07_manual_gru.csv", index=False)
    print(pd.DataFrame(out).groupby("model")[["roc_auc", "avg_precision"]].agg(["mean", "std"]).round(4))
