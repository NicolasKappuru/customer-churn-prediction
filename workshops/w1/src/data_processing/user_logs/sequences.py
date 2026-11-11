"""user_logs.sequences.

Preprocessing parameters for the daily listening sequences: per-channel clipping
and standardization, learned on the training window only.
"""

import numpy as np
import polars as pl

from w1.src.data_processing.constants import LOG_COLS, SEQ_DAYS
from w1.src.data_processing.utils import _date_int, sequence_start


def fit_sequence_params(user_day, train_users, train_cutoff):
    """Learn daily clipping and sequence standardization from the training window only.

    Returns ``log_clip_p999`` (per-channel 99.9th percentile), ``seq_days``,
    ``seq_mean`` and ``seq_std`` (mean/std of the ``log1p`` of the clipped daily
    values).
    """
    start, cutoff = _date_int(sequence_start(train_cutoff, SEQ_DAYS)), _date_int(train_cutoff)
    window = (user_day.join(train_users.select("uid"), on="uid", how="semi")
                       .filter((pl.col("date") >= start) & (pl.col("date") <= cutoff)))
    if window.is_empty():
        raise ValueError("No training listening records inside the sequence window.")

    clip, mean, std = {}, np.empty(len(LOG_COLS)), np.empty(len(LOG_COLS))
    for i, col in enumerate(LOG_COLS):
        upper = window.select(pl.col(col).quantile(0.999, interpolation="linear")).item()
        cell = window.select(pl.col(col).clip(upper_bound=upper).log1p()).to_series()
        clip[col] = float(upper)
        mean[i] = float(cell.mean())
        std[i] = float(cell.std(ddof=0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Non-finite daily sequence preprocessing parameters.")
    std[std < 1e-6] = 1.0

    return {
        "log_clip_p999": clip,
        "seq_days": int(SEQ_DAYS),
        "seq_mean": mean.tolist(),
        "seq_std": std.tolist(),
    }
