from w2.src.data_processing.utils import _date, _date_int, sequence_start
from w2.src.data_processing.constants import _TXN_KEYS, LOG_COLS, TX_CHANNELS, LSTM_CHANNELS
import numpy as np
import polars as pl

def sequence_channel_specs():
    """Describe each channel in tensor order without transforming the data.

    ``mask`` identifies the channel that indicates whether information is available.
    ``fixed_limits`` documents the bounds applied when building the tensor.
    """
    flag_channels = {
        "active", "tx_state_known", "tx_last_is_cancel",
        "tx_last_is_auto_renew", "tx_expiry_known", "tx_state_ambiguous",
    }
    signed_channels = {"tx_paid_sum", "tx_days_to_expire"}
    event_channels = TX_CHANNELS[:5]
    event_units = {"tx_paid_sum": "NTD", "tx_plan_days_sum": "days"}
    day_limits = {
        "tx_days_to_expire": [-3650, 3650],
        "tx_days_since_last": [0, 3650],
    }

    specs = []
    for name in LSTM_CHANNELS:
        is_flag = name in flag_channels

        if is_flag:
            encoding = "flag"
        elif name in signed_channels:
            encoding = "signed_log1p"
        else:
            encoding = "log1p"

        if name in LOG_COLS:
            group = "listening"
            mask = "active"
            unit = "seconds" if name == "total_secs" else "count"
        elif name == "active":
            group = "listening"
            mask = None
            unit = "flag"
        elif name in event_channels:
            group = "event"
            mask = "tx_count"
            unit = event_units.get(name, "count")
        else:
            group = "state"
            unit = "flag" if is_flag else "days"
            if name == "tx_days_to_expire":
                mask = "tx_expiry_known"
            else:
                mask = "tx_state_known"

        specs.append({
            "name": name,
            "encoding": encoding,
            "group": group,
            "mask": mask,
            "units": unit,
            "fixed_limits": day_limits.get(name),
        })
    return specs

_REQUIRED_COLS = [
    "transaction_date", "payment_plan_days", "actual_amount_paid",
    "is_cancel", "is_auto_renew", "_ord",
]
 
 
def prepare_transaction_days(trans, users, cutoff):
    """Prepare daily events and the state preceding the 90-day window.

    1. Validate that each uid has a unique tensor position (``row``).
    2. Join transactions to those users and discard future dates.
    3. Check required fields and aggregate payments by user and day.
    4. Keep all observed days within the window and only the latest earlier
       day for each user, which provides their initial state.

    Return a sparse table and a report; the tensor's 90 days are not filled
    here. ``_ord`` breaks ties using source order, not transaction time.
    """
    _validate_users(users)
    txns = _load_transactions(trans, users, cutoff)
 
    preflight = _preflight_report(txns)
    if preflight["invalid_required"]:
        raise ValueError(f"Invalid transaction fields: {preflight}")
 
    daily = _aggregate_daily(txns)
    start = _date(sequence_start(cutoff))
    before, inside = _split_at_window_start(daily, start)
    days = _with_ordinals(pl.concat([before, inside]))
 
    preflight.update(_window_stats(days, inside))
    return days, preflight
 
 
# --------------------------------------------------------------------------
# Validation and loading
# --------------------------------------------------------------------------
 
def _validate_users(users):
    """Require unique uids and row positions 0..N-1 without gaps or duplicates."""
    is_unique = users["uid"].n_unique() == users.height
    is_permutation = np.array_equal(
        np.sort(users["row"].to_numpy()), np.arange(users.height)
    )
    if not (is_unique and is_permutation):
        raise ValueError("Users require unique uid and a row permutation.")
 
 
def _parse_yyyymmdd(col, alias):
    """Convert YYYYMMDD to Date, leaving invalid dates as null."""
    return (
        pl.col(col).cast(pl.String).str.to_date("%Y%m%d", strict=False).alias(alias)
    )
 
 
def _load_transactions(trans, users, cutoff):
    """Filter users and dates, attach ``row``, and remove exact duplicates.

    The inner join on uid keeps only payments belonging to the cohort and
    attaches each user's tensor position, ``row``. Since users.uid is unique,
    each transaction matches one user row, so the join does not multiply payments.
    Users without payments are absent here but retain their row in ``users``.

    This join does not define the time window. Only the upper bound (cutoff)
    is applied here; earlier history is needed to initialize the state.
    Unparseable dates are retained so that preflight validation can reject them.
    """
    lf = trans.lazy() if isinstance(trans, pl.DataFrame) else trans
    return (
        lf.join(users.select("uid", "row").lazy(), on="uid", how="inner")
        .unique(subset=["uid", *_TXN_KEYS], keep="first", maintain_order=True)
        .with_columns(
            _parse_yyyymmdd("transaction_date", "td"),
            _parse_yyyymmdd("membership_expire_date", "expiry"),
        )
        # Rows with unparseable dates are kept on purpose so the preflight flags them.
        .filter(
            (pl.col("transaction_date") <= _date_int(cutoff))
            | pl.col("td").is_null()
        )
    )
 
 
def _invalid_required_expr():
    """Flag payments with required nulls, invalid dates, plans, or binary flags."""
    has_null = pl.any_horizontal([pl.col(c).is_null() for c in _REQUIRED_COLS])
    return (
        has_null
        | pl.col("td").is_null()
        | (pl.col("payment_plan_days") < 0)
        | ~pl.col("is_cancel").is_in([0, 1])
        | ~pl.col("is_auto_renew").is_in([0, 1])
    )
 
 
def _preflight_report(txns):
    """Count issues after filtering/deduplication; invalid_required stops processing.

    Negative amounts and invalid expiry dates are reported but are not fatal:
    the tensor supports signed amounts and unknown expiry dates.
    """
    return txns.select(
        pl.len().alias("n_transactions"),
        _invalid_required_expr().sum().alias("invalid_required"),
        (pl.col("actual_amount_paid") < 0).sum().alias("negative_amounts"),
        pl.col("expiry").is_null().sum().alias("invalid_expiry"),
    ).collect().row(0, named=True)
 
 
# --------------------------------------------------------------------------
# Daily aggregation
# --------------------------------------------------------------------------
 
def _sum_i64(col, alias):
    """Sum a field within each daily group using 64-bit integers."""
    return pl.col(col).cast(pl.Int64).sum().alias(alias)
 
 
def _last_by_ord(col, alias):
    """Take the last value by _ord without assuming known payment times."""
    return pl.col(col).sort_by("_ord").last().alias(alias)
 
 
def _ambiguous_expr():
    """Flag days with conflicting cancellation, renewal, or expiry values."""
    return (
        pl.any_horizontal(
            [pl.col(c).n_unique() > 1 for c in ("is_cancel", "is_auto_renew", "expiry")]
        )
        .cast(pl.Int8)
        .alias("ambiguous")
    )
 
 
def _aggregate_daily(txns):
    """Build one row per (row, date), summing events and taking state by _ord.

    Distinct payments on the same day contribute to the sums and event count.
    Cancellation, renewal, and expiry come from that day's last source record;
    ambiguous indicates conflicting values. Days without events are not filled.
    """
    return (
        txns.group_by("row", "td")
        .agg(
            pl.len().alias("tx_count"),
            _sum_i64("actual_amount_paid", "tx_paid_sum"),
            _sum_i64("payment_plan_days", "tx_plan_days_sum"),
            _sum_i64("is_cancel", "tx_cancel_count"),
            _sum_i64("is_auto_renew", "tx_auto_renew_count"),
            _last_by_ord("is_cancel", "cancel"),
            _last_by_ord("is_auto_renew", "renew"),
            _last_by_ord("expiry", "expiry"),
            _ambiguous_expr(),
        )
        .collect(engine="streaming")
    )
 
 
# --------------------------------------------------------------------------
# Window selection and output
# --------------------------------------------------------------------------
 
def _split_at_window_start(daily, start):
    """Separate the initial state record from events within the window.

    ``before``: sort earlier history and keep only each user's latest day.
    That record initializes the state without creating an event on day 0.
    ``inside``: keep days >= start; loading already enforced dates <= cutoff.
    The event window is therefore [cutoff - 89 days, cutoff], both inclusive.
    """
    before = (
        daily.filter(pl.col("td") < start)
        .sort("row", "td")
        .group_by("row", maintain_order=True)
        .tail(1)
    )
    inside = daily.filter(pl.col("td") >= start)
    return before, inside
 
 
def _with_ordinals(days):
    """Add days since 1970-01-01 and sort by row and date.

    ``ordinal`` is not the 0..89 index: subtract the window start's ordinal
    to obtain that index. The earlier state record has a negative offset.
    """
    return days.with_columns(
        pl.col("td").cast(pl.Int32).alias("ordinal"),
        pl.col("expiry").cast(pl.Int32).alias("expiry_ordinal"),
    ).sort("row", "td")
 
 
def _window_stats(days, inside):
    """Summarize retained rows and days with multiple payments or ambiguous state."""
    return {
        "daily_rows_retained": days.height,
        "multiple_event_days": inside.filter(pl.col("tx_count") > 1).height,
        "ambiguous_event_days": inside.filter(pl.col("ambiguous") == 1).height,
    }
