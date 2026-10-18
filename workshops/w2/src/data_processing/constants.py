"""Data-contract constants shared across the preprocessing modules.

These describe the dataset itself (listening channels, transaction identity,
sequence channels), so they live next to the code that enforces them rather
than in the notebook.
"""

LOG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]

SEQ_DAYS = 90

# Transaction identity: exact duplicates are removed on msno + these fields.
_TXN_KEYS = ["payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
             "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel"]

TX_CHANNELS = [
    "tx_count", "tx_paid_sum", "tx_plan_days_sum", "tx_cancel_count",
    "tx_auto_renew_count", "tx_state_known", "tx_last_is_cancel",
    "tx_last_is_auto_renew", "tx_days_since_last", "tx_days_to_expire",
    "tx_expiry_known", "tx_state_ambiguous",
]

SEQ_CHANNELS = LOG_COLS + ["active"]
LSTM_CHANNELS = SEQ_CHANNELS + TX_CHANNELS