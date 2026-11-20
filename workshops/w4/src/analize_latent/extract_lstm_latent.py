"""Extract LSTM latent vectors (32-dim BN output) for a given set of msno values.

Run from workshops/w2 as cwd:
    python <this_script> --sample <csv_with_msno> --out <output.npz>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Only W2 in sys.path — never mix with W1/W4 to avoid package name collisions
W2_ROOT = Path(__file__).resolve().parents[4] / "workshops" / "w2"
sys.path.insert(0, str(W2_ROOT))

import numpy as np
import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LSTM latent vectors.")
    parser.add_argument("--sample",     required=True, help="CSV with msno column")
    parser.add_argument("--out",        required=True, help="Output .npz path")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load W2 prepared data ────────────────────────────────────────────────
    from src.models.lstm.data import load_prepared_data
    from src.models.lstm.model import LSTMModel

    data = load_prepared_data()   # uses config defaults (relative to cwd = workshops/w2)

    # ── Read requested msno ──────────────────────────────────────────────────
    sample_df = pd.read_csv(args.sample)
    requested = sample_df["msno"].values

    # Map msno -> global row index via cohort_df (test cohort only)
    cohort_test = data.cohort_df[data.cohort_df["cohort"] == "test"].reset_index(drop=True)
    msno_to_row = dict(zip(cohort_test["msno"], cohort_test.index))

    rows = np.array([msno_to_row[m] for m in requested], dtype=np.int64)
    missing = [m for m in requested if m not in msno_to_row]
    if missing:
        raise ValueError(f"{len(missing)} msno values not found in W2 test cohort: {missing[:5]}")

    y_rows = data.y_all[rows]
    print(f"Matched {len(rows):,} clients  churn={y_rows.mean():.3%}")

    # ── Load LSTM checkpoint ─────────────────────────────────────────────────
    ckpt_path = W2_ROOT / "artifacts" / "sherlock_w2" / "lstm" / "checkpoints" / "lstm.pt"
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    params    = ckpt["params"]

    model = LSTMModel(
        input_dim  = data.input_dim,
        hidden_dim = params["hidden_dim"],
        layer_dim  = params["layer_dim"],
        output_dim = 1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # ── Extract 32-dim BN representation ────────────────────────────────────
    latent_parts: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, _ in data.iter_batches(
            rows, phase="development", batch_size=args.batch_size, shuffle=False
        ):
            x_batch = x_batch.to(device)
            h0 = torch.zeros(params["layer_dim"], x_batch.size(0),
                             params["hidden_dim"], device=device, dtype=x_batch.dtype)
            c0 = torch.zeros_like(h0)
            out, _, _ = model.lstm(x_batch, (h0, c0))
            h = model.bn(out[:, -1, :])          # (batch, 32)
            latent_parts.append(h.cpu().numpy().astype(np.float32))

    latent = np.concatenate(latent_parts, axis=0)
    print(f"Latent shape: {latent.shape}  churn_rate={y_rows.mean():.3%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, msno=requested, latent=latent, is_churn=y_rows.astype(np.int8))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
