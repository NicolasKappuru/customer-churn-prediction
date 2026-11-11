"""Persistence helpers for the w2 feature-preparation artifacts.

These functions only describe and write the prepared data. They keep the
orchestrator (:mod:`prepare`) free of JSON/parquet details.
"""

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import N_DAYS


def environment_versions():
    """Return the versions that matter for reproducing the prepared data."""
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch

        environment["torch"] = torch.__version__
    except Exception:
        environment["torch"] = None
    return environment


def build_contract(config, meta, split):
    """Describe the data and split contract and return ``(payload, digest)``."""
    payload = {
        "dataset_version": meta["dataset_version"],
        "cutoffs": meta["cutoffs"],
        "label_months": meta["label_months"],
        "channels": meta["sequences"]["lstm"]["channels"],
        "hash_train_feb": meta["hash_train_feb"],
        "hash_test": meta["hash_test"],
        "split_protocol": config["split_protocol"],
        "split_shares": config["split_shares"],
        "split_sizes": {part: int(len(index)) for part, index in split.items()},
        "n_days": N_DAYS,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]
    return payload, digest


def save_row_index(cohort_df, split, out_dir):
    """Save the labels of the selected rows with their part assigned."""
    rows = np.sort(np.concatenate(list(split.values())))
    parts = np.empty(len(cohort_df), dtype=object)
    for part, index in split.items():
        parts[index] = part

    frame = cohort_df.iloc[rows].loc[:, ["row", "cohort", "msno", "is_churn"]].copy()
    frame["cohort"] = frame["cohort"].astype(str)
    frame["part"] = parts[rows]

    path = Path(out_dir) / "row_index.parquet"
    frame.to_parquet(path, index=False)
    return path


def save_run_metadata(
    config,
    meta,
    split,
    phase_rows,
    phase_files,
    row_index,
    fit_rate,
    out_dir,
):
    """Write ``run_config.json`` describing the prepared artifacts."""
    contract, contract_hash = build_contract(config, meta, split)
    metadata = {
        "config": config,
        "environment": environment_versions(),
        "contract": contract,
        "contract_hash": contract_hash,
        "fit_churn_rate": fit_rate,
        "phase_rows": {
            phase: int(len(rows)) for phase, rows in phase_rows.items()
        },
        "preprocessing": phase_files,
        "row_index": row_index.relative_to(out_dir).as_posix(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (Path(out_dir) / "run_config.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata
