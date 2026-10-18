"""Small helpers to look at one user's sequence as a table (for the notebook)."""

import numpy as np
import pandas as pd

from w2.src.data_processing.constants import SEQ_DAYS
from w2.src.data_processing.utils import sequence_start


def day_frame(seq, channels, cutoff, days=None, decimals=3):
    """Return a ``(90, C)`` user sequence as a DataFrame with ``day`` and ``date``.

    ``days`` selects day indices (0..89); by default all 90 days are returned.
    """
    values = np.asarray(seq, dtype=np.float32)
    if values.shape != (SEQ_DAYS, len(channels)):
        raise ValueError(f"Expected shape {(SEQ_DAYS, len(channels))}, got {values.shape}.")
    frame = pd.DataFrame(values, columns=channels).round(decimals)
    dates = pd.date_range(sequence_start(cutoff), periods=SEQ_DAYS).date
    frame.insert(0, "date", dates)
    frame.insert(0, "day", np.arange(SEQ_DAYS))
    return frame if days is None else frame.iloc[list(days)]

