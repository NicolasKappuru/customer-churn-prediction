"""Preprocessing pipeline orchestration.

Wires the per-domain builders together: builds the full feature set for one
cohort and attaches the labels.
"""

import polars as pl

from w1.src.data_processing.members import build_member_features
from w1.src.data_processing.transactions import build_transaction_features
from w1.src.data_processing.user_logs.monthly import build_log_features, relative_log_recency
from w1.src.data_processing.utils import _shrink


# No transactions means zero observed counts/totals. Means, dates, categories
# and last-record attributes remain null when unavailable.
_TRANSACTION_ZERO_COLUMNS = (
    "transaction_count", "n_payment_methods", "n_plans", "n_list_prices",
    "n_amounts_paid", "sum_plan_days", "sum_list_price", "sum_amount_paid",
    "more_than_30",
)


def _impute_missing_features(frame, params, log_columns):
    """Apply domain defaults without turning unknown attributes into real zeros.

    Missing member ages/sources use the fitted training values. Other missing
    member attributes and transaction statistics stay null for the downstream
    model preprocessor. Listening aggregates use zero for absent activity;
    listening recency and its presence flag are handled separately.
    """
    return frame.with_columns(
        pl.col("has_member_record").fill_null(0),
        pl.col("has_transactions").fill_null(0),
        pl.col("bd").fill_null(params["age_mean"]),
        pl.col("bd_missing").fill_null(1),
        pl.col("registered_via").fill_null(params["registered_via_mode"]),
        *[pl.col(name).fill_null(0) for name in _TRANSACTION_ZERO_COLUMNS],
        *[pl.col(name).fill_null(0) for name in log_columns],
    )


def build_features(members, trans, user_day, users, cutoff, params):
    """Build the full feature set for one set, in ``users`` (row) order."""
    member_feature = build_member_features(members, users, params, cutoff).with_columns(
        pl.lit(1, dtype=pl.Int8).alias("has_member_record")
    )
    transaction_feature = build_transaction_features(trans, users, cutoff).with_columns(
        pl.lit(1, dtype=pl.Int8).alias("has_transactions")
    )
    log_feature = build_log_features(user_day, users, cutoff, params)

    combined = (
        users.select("uid", "msno", "row")
        .join(member_feature, on="uid", how="left", maintain_order="left")
        .join(transaction_feature, on="uid", how="left", maintain_order="left")
        .join(log_feature, on="uid", how="left", maintain_order="left")
    )
    combined = relative_log_recency(combined, cutoff)
    log_columns = [
        name for name in log_feature.columns
        if name not in ("uid", "last_login_date_last")
    ]
    combined = _impute_missing_features(combined, params, log_columns)

    features = [c for c in combined.columns if c not in ("uid", "msno", "row")]
    combined = _shrink(combined).sort("row")
    return combined.select(["msno"] + features)


def attach_labels(feature_tables, labelled_users, test_users) -> dict:
    """Attach the labels to each cohort table and check they share one schema."""
    out = {}
    for split, feats in feature_tables.items():
        keys = (labelled_users[split].select("msno", "is_churn")
                if split in labelled_users else test_users.select("msno"))
        out[split] = keys.join(feats, on="msno", how="left", maintain_order="left")
    schemas = {s: f.drop("is_churn", strict=False).schema for s, f in out.items()}
    assert all(s == schemas["train"] for s in schemas.values()), "cohort tables differ"
    return out
