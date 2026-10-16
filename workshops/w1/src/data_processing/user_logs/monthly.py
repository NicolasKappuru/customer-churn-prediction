"""user_logs.monthly.

Wide monthly listening aggregates for each user's last / second / third active
month. Built from the ``(uid, date)`` table produced by ``daily.py``.
"""

import pandas as pd
import polars as pl

from w1.src.data_processing.constants import LOG_COLS, LOG_NAMES, SEQ_DAYS
from w1.src.data_processing.utils import _date, window_start


_PREFIX = {1: "last", 2: "last_second", 3: "last_third"}


def _build_templates():
    """Map each monthly metric to its wide output name for a rank prefix."""
    templates = {}
    for name in LOG_NAMES.values():
        templates[f"sum_{name}"] = "sum_{p}_month_" + name
        templates[f"mean_{name}"] = "mean_{p}_month_" + name
    templates.update({
        "count": "{p}_login",
        "last_date": "{p}_login_date_last",
        "sum_total_songs": "{p}_sum_total_songs",
        "ratio_100": "{p}_ratio_100",
        "ratio_25": "{p}_ratio_25",
    })
    return templates


def _eligible_months(cutoff):
    """The three calendar months ending at the cutoff, as ``YYYYMM`` ints."""
    last_period = pd.Timestamp(cutoff).to_period("M")
    return [int((last_period - offset).year * 100 + (last_period - offset).month)
            for offset in (2, 1, 0)]


def _monthly_sums(user_day, users):
    """Group user-day rows by ``(uid, month)``: sums, active-day count and last date.

    ``count`` is the number of daily rows (``pl.len()``), i.e. active days, so the monthly
    mean below is a per-day mean on the same scale as the daily-learned clipping limit.
    """
    return (
        user_day.join(users.select("uid"), on="uid", how="semi")
        .with_columns((pl.col("date") // 100).cast(pl.Int32).alias("month"))
        .group_by("uid", "month")
        .agg([pl.col(c).sum().alias(f"sum_{LOG_NAMES[c]}") for c in LOG_COLS]
             + [pl.len().cast(pl.Int64).alias("count"),
                pl.col("date").max().alias("last_date")])
    )


def _clip_means(md, clip):
    """Apply the daily p99.9 to the per-day mean, rebuild sums and add ratios.

    The p99.9 is learned on daily values, so it is applied on the per-day mean;
    the monthly sum is rebuilt from the clipped mean to keep the outlier control.
    """
    md = md.with_columns([
        (pl.col(f"sum_{LOG_NAMES[c]}").cast(pl.Float64) / pl.col("count"))
        .clip(upper_bound=clip[c]).alias(f"mean_{LOG_NAMES[c]}") for c in LOG_COLS])
    md = md.with_columns([
        (pl.col(f"mean_{n}") * pl.col("count")).round().cast(pl.Int64).alias(f"sum_{n}")
        for n in LOG_NAMES.values()])
    md = md.with_columns(
        pl.sum_horizontal([pl.col(f"sum_{n}") for n in ("25", "50", "75", "985", "100")])
          .alias("sum_total_songs"))
    songs = pl.when(pl.col("sum_total_songs") > 0).then(pl.col("sum_total_songs")).otherwise(None)
    return md.with_columns((pl.col("sum_100") / songs).alias("ratio_100"),
                           (pl.col("sum_25") / songs).alias("ratio_25"))


def _rank_active_months(md, months):
    """Keep the eligible months and rank them per user (1 = most recent)."""
    return (
        md.filter(pl.col("month").is_in(months))
        .with_columns(
            pl.col("month").rank("ordinal", descending=True).over("uid")
            .cast(pl.Int8).alias("rank"))
    )


def _widen_ranks(md, templates):
    """Pivot the last three active months into one wide row per ``uid``."""
    parts = [
        md.filter(pl.col("rank") == rank).select(
            [pl.col("uid")]
            + [pl.col(metric).alias(template.format(p=prefix))
               for metric, template in templates.items()])
        for rank, prefix in _PREFIX.items()
    ]
    log = parts[0]
    for part in parts[1:]:
        log = log.join(part, on="uid", how="full", coalesce=True)
    return log


def _add_relative_diffs(log):
    """Difference between the last and the second-to-last active month."""
    exprs = []
    for how in ("sum", "mean"):
        for metric in ("25", "50", "75", "985", "100", "unq", "total_secs"):
            exprs.append((pl.col(f"{how}_last_month_{metric}")
                          - pl.col(f"{how}_last_second_month_{metric}"))
                         .alias(f"{how}_diff_last_2_month_{metric}"))
    exprs.append((pl.col("last_login") - pl.col("last_second_login")).alias("login_difference"))
    return log.with_columns(exprs)


def build_log_features(user_day, users, cutoff, params):
    """Monthly log aggregates for the users' last / second / third active month (D-1)."""
    months = _eligible_months(cutoff)
    templates = _build_templates()

    md = _monthly_sums(user_day, users)
    md = _clip_means(md, params["log_clip_p999"])
    md = _rank_active_months(md, months)

    log = _widen_ranks(md, templates)
    log = log.drop([c for c in ("last_second_login_date_last", "last_third_login_date_last")
                    if c in log.columns])
    log = log.with_columns(pl.col("last_login_date_last").fill_null(0).cast(pl.Int64))
    log = log.fill_null(0)
    return _add_relative_diffs(log)


def relative_log_recency(frame, cutoff):
    """Replace the literal last login date with age relative to the prediction cutoff.

    Drops ``last_login_date_last`` and adds ``days_since_last_log`` (with an
    out-of-window sentinel derived from ``SEQ_DAYS`` when there is no date) and
    ``has_listening_logs``.
    """
    last_date = pl.col("last_login_date_last").cast(pl.String).str.to_date("%Y%m%d", strict=False)
    sentinel = max(SEQ_DAYS, (pd.Timestamp(cutoff) - window_start(cutoff)).days + 1) + 1
    return (frame.with_columns(
        pl.when(last_date.is_not_null())
          .then((pl.lit(_date(cutoff)) - last_date).dt.total_days())
          .otherwise(sentinel).cast(pl.Int32).alias("days_since_last_log"),
        last_date.is_not_null().cast(pl.Int8).alias("has_listening_logs"),
    ).drop("last_login_date_last"))
