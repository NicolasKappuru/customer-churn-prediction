"""CSV -> parquet conversion contract.

``SCHEMA`` maps each archive to its column dtypes and ``_COLUMNS`` to the
columns to keep; both describe the raw files, not a particular run.
"""

import polars as pl

from w1.src.data_processing.constants import LOG_COLS, _TXN_KEYS


# Explicit schemas for the conversion. Smallest types that fit the observed ranges
# (CP-01): city 1-22, registered_via -1-19, bd -7168-2016, plan_days 0-450,
# prices 0-2000, payment_method_id 2-41.
SCHEMA = {
    "train": {"msno": pl.String, "is_churn": pl.Int8},
    "train_v2": {"msno": pl.String, "is_churn": pl.Int8},
    "sample_submission_v2": {"msno": pl.String, "is_churn": pl.Int8},
    "members_v3": {"msno": pl.String, "city": pl.Int8, "bd": pl.Int16,
                   "registered_via": pl.Int8, "registration_init_time": pl.Int32,
                   "gender": pl.String},
    "transactions": {"msno": pl.String, "payment_method_id": pl.Int8, "payment_plan_days": pl.Int16,
                     "plan_list_price": pl.Int16, "actual_amount_paid": pl.Int16,
                     "is_auto_renew": pl.Int8, "transaction_date": pl.Int32,
                     "membership_expire_date": pl.Int32, "is_cancel": pl.Int8},
    "transactions_v2": {"msno": pl.String, "payment_method_id": pl.Int8,
                        "payment_plan_days": pl.Int16, "plan_list_price": pl.Int16,
                        "actual_amount_paid": pl.Int16, "is_auto_renew": pl.Int8,
                        "transaction_date": pl.Int32, "membership_expire_date": pl.Int32,
                        "is_cancel": pl.Int8},
    "user_logs": {"msno": pl.String, "date": pl.Int32, "num_25": pl.Int32, "num_50": pl.Int32,
                  "num_75": pl.Int32, "num_985": pl.Int32, "num_100": pl.Int32,
                  "num_unq": pl.Int32, "total_secs": pl.Float64},
    "user_logs_v2": {"msno": pl.String, "date": pl.Int32, "num_25": pl.Int32, "num_50": pl.Int32,
                     "num_75": pl.Int32, "num_985": pl.Int32, "num_100": pl.Int32,
                     "num_unq": pl.Int32, "total_secs": pl.Float64},
}

_COLUMNS = {
    "train": ["msno", "is_churn"],
    "train_v2": ["msno", "is_churn"],
    "sample_submission_v2": ["msno"],
    "members_v3": ["msno", "city", "registered_via", "bd", "registration_init_time", "gender"],
    "transactions": ["msno"] + _TXN_KEYS,
    "transactions_v2": ["msno"] + _TXN_KEYS,
    "user_logs": ["msno", "date"] + LOG_COLS,
    "user_logs_v2": ["msno", "date"] + LOG_COLS,
}
