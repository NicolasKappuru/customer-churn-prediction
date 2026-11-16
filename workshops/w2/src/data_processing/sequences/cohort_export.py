"""Build, write and audit the sequence array of one cohort.

The builder computes the 20 channels of ``LSTM_CHANNELS`` (8 listening + 12
transaction); the audit and the causality checks are defined on them. The file
that is published, ``seq_lstm_<split>.npy``, keeps only ``channels`` (by default
the 14 ``EXPORT_CHANNELS``), as float16 aligned with the cohort's ``row`` order.
"""

import gc
from pathlib import Path

import numpy as np

from w2.src.data_processing.constants import EXPORT_CHANNELS, LSTM_CHANNELS
from w2.src.data_processing.sequences.log_sequences import build_log_sequences, audit_log_sequences
from w2.src.data_processing.sequences.lstm_sequences import build_lstm_sequences, audit_lstm_sequences
from w2.src.data_processing.utils import sequence_start


def write_cohort_sequences(user_day, transactions, users, cutoff, export_dir, split,
                           chunk_size=4096, channels=EXPORT_CHANNELS):
    """Write ``seq_lstm_<split>.npy`` for one cohort and return ``(array, audit)``.

    The listening channels are built in memory and combined with the transaction
    channels block by block into a temporary 20-channel file, which is audited
    (shapes, finite values, binary flags, masks). The published file keeps only
    ``channels``; the temporary file is deleted.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    channels = list(channels)
    keep = [LSTM_CHANNELS.index(name) for name in channels]
    final_path = export_dir / f"seq_lstm_{split}.npy"
    full_path = export_dir / f"_build_seq_lstm_{split}.npy"

    # Unscaled log1p listening; scalers are fit later in the model notebook.
    listening = build_log_sequences(user_day, users, cutoff)
    full, preflight = build_lstm_sequences(
        transactions, users, cutoff, listening, output_path=full_path, chunk_size=chunk_size)

    audit = {
        "listening": audit_log_sequences(listening, chunk_size=chunk_size),
        "lstm": audit_lstm_sequences(full, listening, chunk_size=chunk_size),
        "transaction_preflight": preflight,
        "start": str(sequence_start(cutoff).date()),
        "cutoff": cutoff,
        "built_channels": len(LSTM_CHANNELS),
        "exported_channels": channels,
    }
    del listening
    gc.collect()

    out = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.float16,
                                    shape=(full.shape[0], full.shape[1], len(keep)))
    known = LSTM_CHANNELS.index("tx_state_known")
    expiry_known = LSTM_CHANNELS.index("tx_expiry_known")
    differs = 0
    for start in range(0, len(full), chunk_size):
        block = np.asarray(full[start:start + chunk_size])
        out[start:start + chunk_size] = block[:, :, keep]
        differs += int((block[:, :, known] != block[:, :, expiry_known]).sum())
    out.flush()
    # tx_days_to_expire is scaled on the days where tx_state_known = 1 once
    # tx_expiry_known is dropped; this counts the days where the two differ.
    audit["expiry_known_differs_days"] = differs

    del full
    gc.collect()
    full_path.unlink(missing_ok=True)
    return np.load(final_path, mmap_mode="r+"), audit
