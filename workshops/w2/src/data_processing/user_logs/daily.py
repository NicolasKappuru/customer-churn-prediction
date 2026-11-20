"""user_logs.daily.

The single ``(uid, date)`` user-day table that feeds both the monthly
aggregates and the daily sequences.
"""

from pathlib import Path

import numpy as np
import polars as pl

from w2.src.data_processing.constants import LOG_COLS
from w2.src.data_processing.utils import _date_int


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _resolve_log_paths(parquet_dir, end) -> list[Path]:
    """Pick which parquet files cover the requested window."""
    parquet_dir = Path(parquet_dir)
    paths = [parquet_dir / "user_logs.parquet"]
    if _date_int(end) >= 20170301:
        march = parquet_dir / "user_logs_v2.parquet"
        if not march.exists():
            raise FileNotFoundError("The March cutoff requires user_logs_v2.parquet.")
        paths.append(march)
    return paths


def _load_logs(paths, user_index, start, end) -> pl.LazyFrame:
    """Scan, restrict to the date window and known users."""
    return (
        pl.concat([pl.scan_parquet(p) for p in paths])
        .filter(pl.col("date").is_between(_date_int(start), _date_int(end)))
        .join(user_index.lazy().select("msno", "uid"), on="msno", how="inner")
    )


# ---------------------------------------------------------------------------
# Clean + transform
# ---------------------------------------------------------------------------

def _clean_count(col: str) -> pl.Expr:
    """Per-row: cast and drop negatives (clipping). Aggregation happens later."""
    return pl.col(col).cast(pl.Int64).clip(lower_bound=0)


def _daily_sums() -> list[pl.Expr]:
    """Daily aggregation expressions: summed listening columns plus the raw count."""
    int_cols = [
        _clean_count(c).sum()
        .clip(upper_bound=np.iinfo(np.int32).max)
        .cast(pl.Int32)
        .alias(c)
        for c in LOG_COLS if c != "total_secs"
    ]
    secs = (
        pl.when(pl.col("total_secs").is_finite()).then(pl.col("total_secs")).otherwise(0)
        .clip(lower_bound=0)
        .sum().cast(pl.Float32)
        .alias("total_secs")
    )
    return [*int_cols, secs, pl.len().alias("rows")]


def build_user_day_logs(parquet_dir, user_index, start, end) -> pl.DataFrame:
    """One row per ``(uid, date)`` with the summed listening columns and the raw row count.

    ``start`` / ``end`` are ``YYYYMMDD`` ints. This is the single intermediate table that feeds
    both the monthly aggregates and the daily sequences.
    """
    paths = _resolve_log_paths(parquet_dir, end)
    return (
        _load_logs(paths, user_index, start, end)
        .group_by("uid", "date")
        .agg(_daily_sums())
        .collect(engine="streaming")
    )
