"""Sanity checks for the transaction channels of the combined LSTM tensor.

A one-user fixture is run through ``build_lstm_sequences`` and compared with a
slow, day-by-day reference loop. A second run adds a future event to verify that
it never changes earlier days (no backward leakage).
"""

import numpy as np
import polars as pl

from w2.src.data_processing.constants import SEQ_CHANNELS, SEQ_DAYS, TX_CHANNELS
from w2.src.data_processing.sequences.lstm_sequences import build_lstm_sequences
from w2.src.data_processing.sequences.transactions_sequences import prepare_transaction_days
from w2.src.data_processing.utils import _date, sequence_start

_MAX_DAYS = 3650


def causal_fixture():
    """One user with a pre-window payment and an in-window payment."""
    users = pl.DataFrame({"uid": [1], "row": [0]})
    base = {"uid": 1, "payment_method_id": 1, "payment_plan_days": 30,
            "plan_list_price": 100, "actual_amount_paid": 100, "is_auto_renew": 1,
            "is_cancel": 0}
    transactions = pl.DataFrame([
        {**base, "_ord": 0, "transaction_date": 20161130, "membership_expire_date": 20161231},
        {**base, "_ord": 1, "transaction_date": 20161203, "membership_expire_date": 20170102},
    ])
    return transactions, users


def _signed_log1p(x):
    return np.sign(x) * np.log1p(abs(x))


def reference_transaction_channels(transactions, users, cutoff):
    """Slow day-by-day version of the 12 transaction channels for a single user."""
    records = prepare_transaction_days(transactions, users, cutoff)[0].to_dicts()
    first_ordinal = int(np.datetime64(_date(sequence_start(cutoff)), "D").astype(np.int64))
    channel = {name: i for i, name in enumerate(TX_CHANNELS)}
    reference = np.zeros((SEQ_DAYS, len(TX_CHANNELS)), dtype=np.float32)

    # State known before the window starts (latest pre-window record).
    latest = None
    for record in records:
        if record["ordinal"] < first_ordinal:
            latest = record

    for day in range(SEQ_DAYS):
        ordinal = first_ordinal + day
        today = next((r for r in records if r["ordinal"] == ordinal), None)
        if today is not None:
            for name in TX_CHANNELS[:5]:
                reference[day, channel[name]] = _signed_log1p(today[name])
            latest = today
        if latest is None:
            continue
        reference[day, channel["tx_state_known"]] = 1
        reference[day, channel["tx_last_is_cancel"]] = latest["cancel"]
        reference[day, channel["tx_last_is_auto_renew"]] = latest["renew"]
        elapsed = min(max(ordinal - latest["ordinal"], 0), _MAX_DAYS)
        reference[day, channel["tx_days_since_last"]] = np.log1p(elapsed)
        if latest["expiry_ordinal"] is not None:
            remaining = np.clip(latest["expiry_ordinal"] - ordinal, -_MAX_DAYS, _MAX_DAYS)
            reference[day, channel["tx_days_to_expire"]] = _signed_log1p(remaining)
            reference[day, channel["tx_expiry_known"]] = 1
        reference[day, channel["tx_state_ambiguous"]] = latest["ambiguous"]
    return reference


def check_transaction_causality(cutoff, future_date=20170201):
    """Run both checks on the fixture and return its combined (1, 90, 20) tensor."""
    transactions, users = causal_fixture()
    listening = np.zeros((1, SEQ_DAYS, len(SEQ_CHANNELS)), dtype=np.float16)
    combined, _ = build_lstm_sequences(transactions, users, cutoff, listening)

    # 1) The vectorised builder matches the slow reference on every transaction channel.
    reference = reference_transaction_channels(transactions, users, cutoff)
    np.testing.assert_array_equal(reference.astype(np.float16),
                                  combined[0, :, len(SEQ_CHANNELS):])

    # 2) A later event leaves every earlier day unchanged.
    future = transactions.vstack(transactions.tail(1).with_columns(
        pl.lit(future_date, dtype=pl.Int64).alias("transaction_date"),
        pl.lit(2, dtype=pl.Int64).alias("_ord"),
    ))
    with_future, _ = build_lstm_sequences(future, users, cutoff, listening)
    future_day = (np.datetime64(_date(str(future_date)), "D")
                  - np.datetime64(_date(sequence_start(cutoff)), "D")).astype(int)
    if not np.array_equal(combined[0, :future_day], with_future[0, :future_day]):
        raise AssertionError("A future transaction changed earlier days.")
    return combined
