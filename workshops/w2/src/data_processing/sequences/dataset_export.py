"""Publish the sequence dataset: arrays, labels and ``metadata.json``.

Layout of one export directory::

    seq_lstm_<cohort>.npy       (users, 90, 20) float16
    labels_<cohort>.parquet     row, msno, is_churn  (same order as the array)
    sequence_audit.json         audit summary per cohort (written in section 8)
    metadata.json               contract: cutoffs, channels, shapes, row hashes

``metadata.json`` is written last (atomically), so its presence means the export
is complete.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from w2.src.data_processing.constants import LSTM_CHANNELS
from w2.src.data_processing.sequences.transactions_sequences import sequence_channel_specs

_TRANSACTION_SOURCES = ["transactions.parquet", "transactions_v2.parquet"]
_TIE_POLICY = {"version": 1, "order": ["transaction_date", "_ord"],
               "meaning": "deterministic source order; intraday chronology unknown"}


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def export_sequence_dataset(export_dir, lstm, labels, users, audit, metadata, dataset_version):
    """Check alignment, write the labels and publish ``metadata.json``.

    ``lstm``, ``labels``, ``users`` and ``audit`` are dicts keyed by cohort;
    ``metadata`` holds the run configuration (cutoffs, windows, ...).
    """
    root = Path(export_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").unlink(missing_ok=True)  # invalid until the end

    hashes = {split: _check_alignment(split, lstm[split], labels[split], rows)
              for split, rows in users.items()}

    meta = {
        **metadata,
        "dataset_version": dataset_version,
        **{f"n_{split}": len(lstm[split]) for split in hashes},
        **{f"hash_{split}": digest for split, digest in hashes.items()},
        "sequences": {"lstm": _write_sequences(root, lstm, hashes)},
        "labels": _write_labels(root, labels, hashes),
        "transaction_sources": _TRANSACTION_SOURCES,
        "transaction_tie_policy": _TIE_POLICY,
        "sequence_audit": audit,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    _write_json_atomic(root / "metadata.json", meta)
    return meta


def verify_export(export_dir, users):
    """Re-read the published export and check it against the cohort users."""
    root = Path(export_dir)
    meta = json.loads((root / "metadata.json").read_text())
    for split, rows in users.items():
        expected = hash_msno(rows.sort("row"))
        array = np.load(root / meta["sequences"]["lstm"][split]["file"], mmap_mode="r")
        labels = pl.read_parquet(root / meta["labels"][split]["file"])
        if len(array) != rows.height or array.dtype != np.float16:
            raise ValueError(f"{split}: unexpected array shape or dtype.")
        if hash_msno(labels) != expected or meta[f"hash_{split}"] != expected:
            raise ValueError(f"{split}: labels or metadata do not match the users.")
    return meta


def export_summary(export_dir):
    """One row per published file with its size in MB."""
    files = sorted(p for p in Path(export_dir).iterdir() if p.is_file())
    return pd.DataFrame({"file": [p.name for p in files],
                         "MB": [round(p.stat().st_size / 1e6, 2) for p in files]})


def hash_msno(frame):
    """SHA-1 of the ``msno`` order (same recipe as the model notebooks' ``_order_hash``)."""
    digest = hashlib.sha1()
    for value in frame["msno"].to_list():
        digest.update(str(value).encode())
        digest.update(b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

def _check_alignment(split, array, labels, users):
    """Users, array and labels must have the same rows in the same order."""
    users = users.sort("row")
    if users["uid"].n_unique() != users.height:
        raise ValueError(f"{split}: duplicated uid.")
    if len(array) != users.height or labels.height != users.height:
        raise ValueError(f"{split}: array/labels/users have different lengths.")
    digest = hash_msno(users)
    if hash_msno(labels) != digest:
        raise ValueError(f"{split}: labels are not in the sequence row order.")
    return digest


def _write_sequences(root, arrays, hashes):
    """Make sure every array is on disk and describe it."""
    spec = {"channels": LSTM_CHANNELS, "channel_specs": sequence_channel_specs(),
            "dtype": "float16", "order": "oldest_to_newest_calendar_day"}
    for split, array in arrays.items():
        if array.dtype != np.float16:
            raise ValueError(f"{split}: the sequence export requires float16.")
        path = root / f"seq_lstm_{split}.npy"
        if _is_memmap_of(array, path):
            array.flush()
        else:
            np.save(path, array)
        spec[split] = {"file": path.name, "shape": list(array.shape), "row_hash": hashes[split]}
    return spec


def _write_labels(root, labels, hashes):
    """Write ``labels_<cohort>.parquet`` and describe each file."""
    spec = {"columns": ["row", "msno", "is_churn"], "positive": 1}
    for split, frame in labels.items():
        path = root / f"labels_{split}.parquet"
        frame.write_parquet(path)
        spec[split] = {"file": path.name, "n": frame.height,
                       "churn_rate": float(frame["is_churn"].mean()),
                       "row_hash": hashes[split]}
    return spec


def _is_memmap_of(array, path):
    filename = getattr(array, "filename", None)
    return filename is not None and Path(filename).resolve() == Path(path).resolve()


def _write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)
