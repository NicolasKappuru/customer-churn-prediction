"""Customer segment of a cohort, selected before the tensor is built.

``manual``: the last transaction day up to the cutoff is a payment without
auto-renew and without cancellation. The state is read with
``state_at_reference`` (``prepare_transaction_days``, last record of the last
day), the same source the tensor's state channels use, so it equals the rule
on the last day of the window: ``tx_state_known = 1``, ``tx_last_is_cancel = 0``,
``tx_last_is_auto_renew = 0``. Users without any transaction are not in it.

Selecting the users first means only the segment is built (about 11 % of each
cohort), instead of building every user and filtering the tensor afterwards.
"""

import polars as pl

from w2.src.data_processing.sequences.cycle_export import state_at_reference

SEGMENTS = ("manual",)


def select_segment(transactions, users, labelled, cutoff, segment="manual"):
    """Restrict one cohort to ``segment`` at ``cutoff`` and renumber its rows.

    ``users``: ``msno, uid, row`` (``row`` = position in the cohort file);
    ``labelled``: ``row, msno, is_churn`` in the same order.
    Returns ``(users, labelled, report)``: both frames keep the original order
    and get ``row = 0..n-1``; ``report`` counts the users before and after.
    """
    if segment not in SEGMENTS:
        raise ValueError(f"segment must be one of {SEGMENTS}")
    state = state_at_reference(transactions, users, cutoff)
    in_segment = (state.filter((pl.col("cancel") == 0) & (pl.col("renew") == 0))
                       .select("row"))
    kept = users.join(in_segment, on="row", how="semi").sort("row")

    new_users = (kept.select("msno", "uid").with_row_index("row")
                     .with_columns(pl.col("row").cast(pl.UInt32))
                     .select("msno", "uid", "row"))
    new_labelled = (labelled.join(kept.select("row"), on="row", how="semi").sort("row")
                            .drop("row").with_row_index("row")
                            .with_columns(pl.col("row").cast(pl.UInt32))
                            .select("row", "msno", "is_churn"))
    if not new_labelled["msno"].equals(new_users["msno"]):
        raise ValueError("Segment users and labels are not aligned.")

    report = {
        "cutoff": cutoff,
        "segment": segment,
        "cohort_users": users.height,
        "cohort_churn_rate": float(labelled["is_churn"].mean()),
        "segment_users": new_users.height,
        "segment_share": new_users.height / max(users.height, 1),
        "segment_churn_rate": float(new_labelled["is_churn"].mean()),
    }
    return new_users, new_labelled, report
