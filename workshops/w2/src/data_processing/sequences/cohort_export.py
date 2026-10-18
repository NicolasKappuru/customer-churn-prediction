"""Build, write and audit the sequence array of one cohort.

``seq_lstm_<split>.npy`` holds the twenty combined channels (eight listening +
twelve transaction), as float16 aligned with the cohort's ``row`` order.
"""

import gc
from pathlib import Path

from w2.src.data_processing.sequences.log_sequences import build_log_sequences, audit_log_sequences
from w2.src.data_processing.sequences.lstm_sequences import build_lstm_sequences, audit_lstm_sequences
from w2.src.data_processing.utils import sequence_start


def write_cohort_sequences(user_day, transactions, users, cutoff, export_dir, split,
                           chunk_size=4096):
    """Write ``seq_lstm_<split>.npy`` for one cohort and return ``(lstm, audit)``.

    The listening channels are built in memory, combined with the transaction
    channels block by block and written to disk; only the combined file is kept.
    ``audit`` validates both parts (shapes, finite values, binary flags, masks)
    and summarizes coverage.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Unscaled log1p listening; scalers are fit later in the model notebook.
    listening = build_log_sequences(user_day, users, cutoff)
    lstm, preflight = build_lstm_sequences(
        transactions, users, cutoff, listening,
        output_path=export_dir / f"seq_lstm_{split}.npy", chunk_size=chunk_size)

    audit = {
        "listening": audit_log_sequences(listening, chunk_size=chunk_size),
        "lstm": audit_lstm_sequences(lstm, listening, chunk_size=chunk_size),
        "transaction_preflight": preflight,
        "start": str(sequence_start(cutoff).date()),
        "cutoff": cutoff,
    }
    del listening
    gc.collect()
    return lstm, audit
