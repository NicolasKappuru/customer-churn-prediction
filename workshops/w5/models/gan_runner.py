"""Train (or reload) one GAN on one class of the w2 sequences.

Every GAN of w5 goes through ``train_or_load`` so the notebook, the CLI and the
evaluation read the same files. A GAN is plugged in as a function with the
signature of ``TimeGAN.timegan``::

    train_fn(ori_data, parameters) -> (generated_list, artifacts)

called with ``parameters["return_artifacts"] = True``. ``artifacts`` must hold
``"history"`` (list of dicts, one row per logged iteration, with at least a
``"phase"`` and an ``"iteration"`` key) and may hold ``"checkpoint"`` (anything
``torch.save`` can write). The Wasserstein TimeGAN only needs to follow this
contract to reuse the whole notebook.

Layout of one run (``out_root / <gan> / label<k>``)::

    synthetic.npy     (n, 90, 14) float16, w2 encoding (after ``postprocess``)
    raw.npy           continuous generator output before ``postprocess`` (float16)
    upper.npy         per-channel maximum of the real training users (cap of ``postprocess``)
    history.csv       losses per logged iteration
    checkpoint.pt     generator weights + scale (to sample more users later)
    run.json          parameters, sizes and timing
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from w2_sequences import MODEL_CHANNELS, postprocess


def run_dir(out_root, gan_name, label):
    tag = "all" if label is None else f"label{label}"
    return Path(out_root) / gan_name / tag


def train_or_load(gan_name, train_fn, X, channel_specs, params, out_root, *,
                  label=None, retrain=False, verbose=True):
    """Return ``{"synthetic", "raw", "history", "run", "dir"}`` for one GAN/class.

    Reuses the saved run when ``synthetic.npy`` exists and the parameters match;
    set ``retrain=True`` to train again.
    """
    out_dir = run_dir(out_root, gan_name, label)
    run_path = out_dir / "run.json"
    if not retrain and (out_dir / "synthetic.npy").is_file() and run_path.is_file():
        run = json.loads(run_path.read_text())
        same = all(run.get(k) == v for k, v in params.items())
        if same and run.get("train_rows") == len(X):
            if verbose:
                print(f"[{gan_name} | label={label}] reusing {out_dir}")
            return _load(out_dir)
        if verbose:
            print(f"[{gan_name} | label={label}] saved run has other parameters: retraining")

    if verbose:
        print(f"[{gan_name} | label={label}] training on {len(X):,} users {X.shape[1:]}")
    start = time.perf_counter()
    generated, artifacts = train_fn(X, {**params, "return_artifacts": True})
    seconds = time.perf_counter() - start

    raw = np.stack(generated).astype(np.float32)
    # Conditional amounts are capped at the real range of the training users.
    synthetic = postprocess(raw, channel_specs, seed=params.get("seed", 0), upper=X.max(axis=(0, 1)))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "raw.npy", raw.astype(np.float16))
    np.save(out_dir / "upper.npy", X.max(axis=(0, 1)).astype(np.float32))
    np.save(out_dir / "synthetic.npy", synthetic.astype(np.float16))
    pd.DataFrame(artifacts.get("history", [])).to_csv(out_dir / "history.csv", index=False)
    if artifacts.get("checkpoint") is not None:
        torch.save(artifacts["checkpoint"], out_dir / "checkpoint.pt")
    run_path.write_text(json.dumps({
        **params, "gan": gan_name, "label": label, "train_rows": len(X),
        "train_seconds": round(seconds, 1), "channels": list(MODEL_CHANNELS),
        "shape": list(synthetic.shape),
        "encoding": "same as the w2 export (log1p / signed_log1p / flags), float16",
    }, indent=2, default=_json_default))
    if verbose:
        print(f"[{gan_name} | label={label}] {seconds / 60:.1f} min -> {out_dir}")
    return _load(out_dir)


def load_run(out_root, gan_name, label):
    """Read a saved run without training or checking parameters (read-only)."""
    out_dir = run_dir(out_root, gan_name, label)
    if not (out_dir / "synthetic.npy").is_file():
        raise FileNotFoundError(f"no saved run in {out_dir}")
    return _load(out_dir)


def _load(out_dir):
    history_path = out_dir / "history.csv"
    history = pd.read_csv(history_path) if history_path.stat().st_size > 1 else pd.DataFrame()
    raw_path = out_dir / "raw.npy"
    return {
        "synthetic": np.load(out_dir / "synthetic.npy").astype(np.float32),
        "raw": np.load(raw_path).astype(np.float32) if raw_path.is_file() else None,
        "history": history,
        "run": json.loads((out_dir / "run.json").read_text()),
        "dir": out_dir,
    }


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
