"""Data-contract constants shared across the preprocessing modules.

These describe the dataset itself (listening channels, transaction identity,
categorical features), so they live next to the code that enforces them rather
than in the notebook.
"""

LOG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]
LOG_NAMES = {
    "num_25": "25", "num_50": "50", "num_75": "75", "num_985": "985",
    "num_100": "100", "num_unq": "unq", "total_secs": "total_secs",
}

CATEGORICAL = ["city", "registered_via", "last_payment_method_id"]

SEQ_DAYS = 90

# Transaction identity: exact duplicates are removed on msno + these fields.
_TXN_KEYS = ["payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
             "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel"]
