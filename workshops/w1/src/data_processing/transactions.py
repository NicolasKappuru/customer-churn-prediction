"""Transactions domain.

Loading and deduplicating both transaction releases, then transforming the
per-record table into one row of aggregate features per user.
"""

from pathlib import Path

import polars as pl

from w1.src.data_processing.constants import _TXN_KEYS
from w1.src.data_processing.utils import _date, _date_int


# ---------------------------------------------------------------------------
# Load + clean (deduplication)
# ---------------------------------------------------------------------------

def scan_transactions(parquet_dir, user_index, max_date):
    """Lazy transactions (both files, deduplicated) up to ``max_date``, keyed by ``uid``.

    Loads ``transactions`` and ``transactions_v2``, keeps target users, drops
    records after ``max_date`` and removes exact duplicates using ``msno`` plus
    all :data:`_TXN_KEYS` fields (the first copy is kept). ``_ord`` records the
    stable source order and is used later as a tie break, not as intraday time.
    """
    parquet_dir = Path(parquet_dir)
    frames = [
        pl.scan_parquet(parquet_dir / "transactions.parquet"),
        pl.scan_parquet(parquet_dir / "transactions_v2.parquet"),
    ]

    return (
        pl.concat(frames)
        .with_row_index("_ord")
        .join(
            user_index.lazy().select("msno", "uid"),
            on="msno",
            how="inner",
            maintain_order="left",
        )
        .filter(pl.col("transaction_date") <= _date_int(max_date))
        .unique(
            subset=["msno", *_TXN_KEYS],
            keep="first",
            maintain_order=True,
        )
        .select("uid", "_ord", *_TXN_KEYS)
    )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def _prepare_transactions(trans, users, cutoff) -> pl.LazyFrame:
    """Filter transactions up to the cutoff and add per-record derived columns.

    Parses the date fields and derives ``duration``, ``plan_coverage_ratio``,
    ``more_than_30``, ``autorenew_not_cancel``, ``not_autorenew_cancel`` and
    ``discount`` used by the aggregation.
    """
    cutoff_int = _date_int(cutoff)
    lf = trans.lazy() if isinstance(trans, pl.DataFrame) else trans

    return (
        lf.join(users.lazy().select("uid"), on="uid", how="semi")
        .filter(pl.col("transaction_date") <= cutoff_int)
        .with_columns(
            pl.col("transaction_date").cast(pl.String).str.to_date("%Y%m%d").alias("td"),
            pl.col("membership_expire_date").cast(pl.String).str.to_date("%Y%m%d").alias("ed"),
        )
        .with_columns(
            (pl.col("ed") - pl.col("td")).dt.total_days().cast(pl.Int32).alias("duration")
        )
        .with_columns(
            (pl.col("duration") / pl.when(pl.col("payment_plan_days") > 0)
                .then(pl.col("payment_plan_days")).otherwise(None)
            ).alias("plan_coverage_ratio"),
            (pl.col("duration") > 30).cast(pl.Int8).alias("more_than_30"),
            ((pl.col("is_auto_renew") == 1) & (pl.col("is_cancel") == 0))
                .cast(pl.Int8).alias("autorenew_not_cancel"),
            ((pl.col("is_auto_renew") == 0) & (pl.col("is_cancel") == 1))
                .cast(pl.Int8).alias("not_autorenew_cancel"),
            (pl.col("plan_list_price") - pl.col("actual_amount_paid"))
                .cast(pl.Int16).alias("discount"),
        )
    )


def _last(col: str) -> pl.Expr:
    """Value of ``col`` in the most recent transaction (ties broken by ``_ord``)."""
    return pl.col(col).sort_by(["td", "_ord"]).last()


def _aggregate_per_user(t: pl.LazyFrame, cutoff) -> pl.LazyFrame:
    """Collapse the prepared transactions into one row per ``uid``.

    Aggregates counts, means, sums and last-record values, plus timing features
    relative to the cutoff (``days_to_expire``, ``days_since_last_transaction``).
    """
    cutoff_date = pl.lit(_date(cutoff))
    last_td, last_ed = _last("td"), _last("ed")

    return (
        t.group_by("uid")
        .agg(
            pl.col("is_cancel").max().alias("is_cancel_change_flag"),
            pl.col("is_auto_renew").max().alias("is_auto_renew_change_flag"),
            pl.col("payment_method_id").count().alias("transaction_count"),
            pl.col("actual_amount_paid").mean().alias("average_amount_paid"),
            pl.col("plan_list_price").mean().alias("average_amount_charged"),
            pl.col("payment_plan_days").mean().alias("average_plan"),
            pl.col("payment_method_id").n_unique().alias("n_payment_methods"),
            pl.col("payment_plan_days").n_unique().alias("n_plans"),
            pl.col("plan_list_price").n_unique().alias("n_list_prices"),
            pl.col("actual_amount_paid").n_unique().alias("n_amounts_paid"),
            pl.col("duration").mean(),
            pl.col("plan_coverage_ratio").mean(),
            pl.col("payment_plan_days").sum().alias("sum_plan_days"),
            pl.col("plan_list_price").sum().alias("sum_list_price"),
            pl.col("actual_amount_paid").sum().alias("sum_amount_paid"),
            pl.col("more_than_30").sum(),
            pl.col("autorenew_not_cancel").min(),
            pl.col("not_autorenew_cancel").min(),
            pl.col("payment_plan_days").max().alias("max_plan_days"),
            _last("payment_method_id").cast(pl.Int8).alias("last_payment_method_id"),
            _last("payment_plan_days").cast(pl.Int16).alias("last_plan_days"),
            _last("is_auto_renew").cast(pl.Int8).alias("last_is_auto_renew"),
            _last("is_cancel").cast(pl.Int8).alias("last_is_cancel"),
            _last("discount").alias("last_discount"),
            (last_ed - cutoff_date).dt.total_days().cast(pl.Int32).alias("days_to_expire"),
            (last_ed.dt.weekday() - 1).cast(pl.Int8).alias("expire_dayofweek"),
            (cutoff_date - last_td).dt.total_days().cast(pl.Int32)
                .alias("days_since_last_transaction"),
        )
        .with_columns(
            (pl.col("average_amount_paid") - pl.col("average_amount_charged"))
            .alias("difference")
        )
    )


def build_transaction_features(trans, users, cutoff) -> pl.DataFrame:
    """Aggregate transactions per user up to the cutoff (MOD-02).

    Public entry point: prepares the records and collects the per-user
    aggregation in streaming mode.
    """
    t = _prepare_transactions(trans, users, cutoff)
    return _aggregate_per_user(t, cutoff).collect(engine="streaming")
