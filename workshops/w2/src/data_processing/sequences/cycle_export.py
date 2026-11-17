"""Per-cycle sequences: each customer's 90-day window ends ``k`` days before their expiry.

The calendar export ends every window at the cohort cutoff (2017-01-31 or
2017-02-28). Here every customer gets their own ``as_of = anchor - k``:

- ``anchor``: the expiry known at the cohort's reference date (the last
  transaction day up to that date, last record by ``_ord``, exactly as the
  tensor's state channels read it). Only customers whose anchor falls inside
  the label month are in scope (1-28 days after 2017-01-31, 1-31 days after
  2017-02-28), as in ``eda/10_per_cycle_prediction.ipynb``.
- ``segment``: ``"manual"`` keeps customers whose state at the reference date
  is a payment without auto-renew (same rule as the calendar manual run);
  ``"all"`` keeps every customer in scope.
- eligibility (no look-ahead): when ``as_of`` is before the reference date the
  anchor itself comes from the future, so the customer is kept only if the
  expiry already known at ``as_of`` equals the anchor. When ``as_of`` is on or
  after the reference date everyone is kept; whatever happened in between (an
  early payment, a cancellation) is legitimate information.

How the tensor is built. The existing builders take one global cutoff. Instead
of rewriting them, every date of a customer (transaction, expiry, listening
day) is shifted by ``reference - as_of`` days, so their ``as_of`` lands on the
reference date, and the unchanged calendar builder is run with that cutoff.
All channels depend only on date differences, so the result is the window
that ends on ``as_of``; anything dated after ``as_of`` is dropped by the
builder's own cutoff filter. ``check_shift_equivalence`` verifies this against
a direct build with ``cutoff = as_of``.
"""

from datetime import timedelta

import numpy as np
import polars as pl

from w2.src.data_processing.constants import EXPORT_CHANNELS, LSTM_CHANNELS
from w2.src.data_processing.sequences.lstm_sequences import build_lstm_sequences
from w2.src.data_processing.sequences.log_sequences import build_log_sequences
from w2.src.data_processing.sequences.transactions_sequences import prepare_transaction_days
from w2.src.data_processing.utils import _date

SEGMENTS = ("all", "manual")


def cycle_export_name(k, segment):
    """Export folder name (the model side builds the same name in ``src/models/constants.py``)."""
    return f"export-sequential-v6-cycle-k{k}-{segment}"


# --------------------------------------------------------------------------
# Population: anchor, segment, as_of, eligibility
# --------------------------------------------------------------------------

def state_at_reference(transactions, users, reference):
    """Last transaction-day state of every user at ``reference``.

    Returns ``row, cancel, renew, anchor`` (``anchor`` = expiry of that day, a
    date or null). Users without transactions are absent.
    """
    days, _ = prepare_transaction_days(transactions, users, reference)
    return (days.group_by("row", maintain_order=True).last()
            .select("row", "cancel", "renew", pl.col("expiry").alias("anchor")))


def expiry_known_at(transactions, population):
    """Expiry of each user's last transaction day dated ``<= as_of`` (null if none)."""
    return (
        transactions.lazy()
        .join(population.lazy().select("uid", "as_of"), on="uid", how="inner")
        .with_columns(
            td=pl.col("transaction_date").cast(pl.String).str.to_date("%Y%m%d", strict=False),
            ed=pl.col("membership_expire_date").cast(pl.String).str.to_date("%Y%m%d", strict=False),
        )
        .filter(pl.col("td") <= pl.col("as_of"))
        .sort("uid", "td", "_ord")
        .group_by("uid")
        .agg(expiry_at_as_of=pl.col("ed").last())
        .collect(engine="streaming")
    )


def cycle_population(transactions, users, labelled, reference, max_days, k, segment):
    """In-scope, eligible customers of one cohort with their ``as_of``.

    ``users``: ``msno, uid, row`` (the cohort's label order); ``labelled``:
    ``msno, is_churn``. Returns ``(population, report)``; ``population`` has
    ``msno, uid, is_churn, anchor, as_of, offset, eligible`` in label order.
    """
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {SEGMENTS}")
    ref = _date(reference)
    state = state_at_reference(transactions, users, reference)
    frame = (users.join(state, on="row", how="left")
                  .join(labelled.select("msno", "is_churn"), on="msno", how="left")
                  .sort("row")
                  .with_columns(dte=(pl.col("anchor") - pl.lit(ref)).dt.total_days()))
    in_scope = pl.col("dte").is_between(1, max_days).fill_null(False)
    in_segment = (pl.lit(True) if segment == "all"
                  else ((pl.col("cancel") == 0) & (pl.col("renew") == 0)).fill_null(False))

    report = [{"step": "cohort", "customers": frame.height, "churn_rate": frame["is_churn"].mean()}]
    frame = frame.filter(in_segment)
    report.append({"step": f"segment = {segment} at {reference}", "customers": frame.height,
                   "churn_rate": frame["is_churn"].mean()})
    frame = frame.filter(in_scope)
    report.append({"step": f"anchor 1-{max_days} days after {reference}", "customers": frame.height,
                   "churn_rate": frame["is_churn"].mean()})

    frame = frame.with_columns(as_of=pl.col("anchor") - pl.duration(days=k))
    known = expiry_known_at(transactions, frame)
    frame = (frame.join(known, on="uid", how="left", maintain_order="left")
                  .with_columns(eligible=(pl.col("as_of") >= pl.lit(ref))
                                | (pl.col("expiry_at_as_of") == pl.col("anchor")).fill_null(False)))
    dropped = frame.filter(~pl.col("eligible"))
    frame = frame.filter("eligible")
    report.append({"step": f"eligible at as_of = anchor - {k}", "customers": frame.height,
                   "churn_rate": frame["is_churn"].mean()})
    report.append({"step": "  dropped (anchor not yet known at as_of)", "customers": dropped.height,
                   "churn_rate": dropped["is_churn"].mean() if dropped.height else None})

    population = frame.select(
        "msno", "uid", pl.col("is_churn").cast(pl.Int8), "anchor", "as_of", "eligible",
        offset=(pl.lit(ref) - pl.col("as_of")).dt.total_days().cast(pl.Int32),
    )
    return population, report


# --------------------------------------------------------------------------
# Date shift
# --------------------------------------------------------------------------

def _shift_int_dates(frame, columns):
    """Shift ``YYYYMMDD`` integer columns by the per-row ``offset`` (days); invalid dates stay as they are."""
    out = frame
    for col in columns:
        parsed = pl.col(col).cast(pl.String).str.to_date("%Y%m%d", strict=False)
        shifted = (parsed + pl.duration(days=pl.col("offset"))).dt.strftime("%Y%m%d").cast(pl.Int32)
        out = out.with_columns(
            pl.when(parsed.is_not_null()).then(shifted).otherwise(pl.col(col).cast(pl.Int32)).alias(col))
    return out


def shift_to_reference(transactions, user_day, population):
    """Transactions and user-day logs of ``population`` with every date moved by ``offset``."""
    offsets = population.select("uid", "offset")
    txn = _shift_int_dates(transactions.join(offsets, on="uid", how="inner"),
                           ["transaction_date", "membership_expire_date"]).drop("offset")
    logs = _shift_int_dates(user_day.join(offsets, on="uid", how="inner"), ["date"]).drop("offset")
    return txn, logs


def sequence_users(population):
    """``msno, uid, row`` with ``row = 0..n-1`` in population order (the tensor order)."""
    return population.select("msno", "uid").with_row_index("row").with_columns(pl.col("row").cast(pl.UInt32))


def build_labels(population):
    """``row, msno, is_churn`` aligned with :func:`sequence_users`."""
    return population.select("msno", pl.col("is_churn").cast(pl.Int8)).with_row_index("row") \
        .with_columns(pl.col("row").cast(pl.UInt32)).select("row", "msno", "is_churn")


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_shift_equivalence(transactions, user_day, population, reference, tensor, n_dates=3, max_users=300,
                            channels=EXPORT_CHANNELS):
    """The shifted build must equal a direct calendar build with ``cutoff = as_of``.

    For the ``n_dates`` most frequent ``as_of`` dates, up to ``max_users``
    customers sharing that date are rebuilt with the unshifted data and the
    unchanged builders, and compared with their rows of ``tensor`` (whose last
    axis holds ``channels``).
    Returns one row per checked date.
    """
    rows = sequence_users(population).join(population.select("uid", "as_of"), on="uid")
    common = rows.group_by("as_of").len().sort("len", descending=True).head(n_dates)["as_of"].to_list()
    out = []
    for as_of in common:
        sample = rows.filter(pl.col("as_of") == as_of).head(max_users)
        local = sample.select("uid", "msno", pl.int_range(sample.height, dtype=pl.UInt32).alias("row"))
        listening = build_log_sequences(user_day, local, as_of)
        direct, _ = build_lstm_sequences(transactions, local, as_of, listening)
        direct = direct[:, :, [LSTM_CHANNELS.index(c) for c in channels]]
        shifted = np.asarray(tensor[sample["row"].to_numpy()])
        out.append({"as_of": as_of, "customers": sample.height,
                    "identical": bool(np.array_equal(direct, shifted))})
        if not out[-1]["identical"]:
            raise AssertionError(f"Shifted tensor differs from the direct build at as_of={as_of}.")
    return out


def last_day_days_to_expire(tensor, channel_index, block=8192):
    """Decode ``tx_days_to_expire`` on the last day (``as_of``) for every row."""
    values = np.concatenate([np.asarray(tensor[i:i + block, -1, channel_index], dtype=np.float32)
                             for i in range(0, len(tensor), block)])
    return np.rint(np.sign(values) * np.expm1(np.abs(values))).astype(int)


def window_bounds(population, seq_days=90):
    """First listening day needed and last ``as_of`` of a population."""
    first = population["as_of"].min() - timedelta(days=seq_days - 1)
    return first, population["as_of"].max()
