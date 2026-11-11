"""Daily listening tensor ``(users, 90, 8)``: seven log1p counters plus ``active``."""

import numpy as np
import polars as pl

from w2.src.data_processing.constants import LOG_COLS, SEQ_CHANNELS, SEQ_DAYS
from w2.src.data_processing.utils import sequence_start, _date_int, _date

N_LOG = len(LOG_COLS)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def build_log_sequences(user_day, users, cutoff):
    """Calendar-day tensor in float16, aligned by ``users.row``.

    Counters are exported as unscaled ``log1p``; clipping and scaling are fit
    later, on each modelling split.
    """
    _validate_users(users)
    start = sequence_start(cutoff)

    uw = (
        _window_user_days(user_day, users, start, cutoff)
        .with_columns(_channel_exprs())
        .with_columns(_day_index_expr(start))
    )
    return _scatter_to_dense(uw, n_users=users.height)


def _validate_users(users):
    if users["uid"].n_unique() != users.height:
        raise ValueError("Sequence users must have unique uid values.")
    rows = np.sort(users["row"].to_numpy())
    if not np.array_equal(rows, np.arange(users.height)):
        raise ValueError("Sequence row values must be a permutation of 0..n_users-1.")


def _window_user_days(user_day, users, start, cutoff):
    """Keep only known users inside [start, cutoff] and attach their tensor row."""
    lo, hi = _date_int(start), _date_int(cutoff)
    return (
        user_day
        .filter(pl.col("date").is_between(lo, hi))
        # uid is unique in users, so a single inner join replaces semi + inner.
        .join(users.select("uid", "row"), on="uid", how="inner")
    )


def _channel_exprs():
    """One expression per listening channel: negatives → 0, then log1p."""
    return [
        pl.col(c).clip(lower_bound=0).log1p().cast(pl.Float32).alias(c)
        for c in LOG_COLS
    ]


def _day_index_expr(start):
    """Days elapsed since the sequence start, from a YYYYMMDD integer column."""
    return (
        (pl.col("date").cast(pl.String).str.to_date("%Y%m%d") - pl.lit(_date(start)))
        .dt.total_days()
        .cast(pl.Int32)
        .alias("day")
    )


def _scatter_to_dense(uw, n_users):
    """Write the long (user, day) table into a dense [user, day, channel] tensor."""
    rows = uw["row"].to_numpy()
    days = uw["day"].to_numpy()

    if days.size and (days.min() < 0 or days.max() >= SEQ_DAYS):
        raise ValueError("Day index falls outside the sequence window.")

    positions = rows.astype(np.int64) * SEQ_DAYS + days
    if np.unique(positions).size != positions.size:
        raise ValueError("Daily sequences require one row per user and calendar day.")

    values = uw.select(LOG_COLS).to_numpy().astype(np.float16)
    if not np.isfinite(values).all():
        raise ValueError("Daily sequence values are not finite after preprocessing.")

    seq = np.zeros((n_users, SEQ_DAYS, len(SEQ_CHANNELS)), dtype=np.float16)
    seq[rows, days, :N_LOG] = values
    seq[rows, days, -1] = 1  # active-day flag
    return seq


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def audit_log_sequences(seq, encoding="log1p", chunk_size=4096):
    """Validate sequence values in bounded memory and summarize calendar coverage."""
    if seq.ndim != 3 or seq.shape[1:] != (SEQ_DAYS, len(SEQ_CHANNELS)):
        raise ValueError("Unexpected daily sequence dimensions.")

    users_without_logs = 0
    users_with_logs_by_day = np.zeros(SEQ_DAYS, dtype=np.int64)

    for i in range(0, len(seq), chunk_size):
        active = _check_chunk(np.asarray(seq[i:i + chunk_size]), encoding)
        users_without_logs += int((~active.any(axis=1)).sum())
        users_with_logs_by_day += active.sum(axis=0, dtype=np.int64)

    n = len(seq)
    return {
        "n_users": n,
        "users_without_logs": users_without_logs,
        "share_without_logs": users_without_logs / max(n, 1),
        "mean_days_with_logs": int(users_with_logs_by_day.sum()) / max(n, 1),
        "users_with_logs_by_day": users_with_logs_by_day.tolist(),
    }


def _check_chunk(batch, encoding):
    """Validate one chunk and return its boolean active-day mask [user, day]."""
    if not np.isfinite(batch).all():
        raise ValueError("Non-finite sequence values.")

    flag = batch[:, :, -1]
    if not np.isin(flag, [0, 1]).all():
        raise ValueError("Sequence active flag must be binary.")
    active = flag == 1

    channels = batch[:, :, :-1]
    if np.any(channels[~active] != 0):
        raise ValueError("Days without logs must have zero listening channels.")
    if encoding == "log1p" and (channels < 0).any():
        raise ValueError("Unscaled log1p listening channels must be nonnegative.")
    return active