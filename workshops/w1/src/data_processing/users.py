"""Users domain.

Loading the February and March labelled cohorts, and attaching the shared ``uid``
to each cohort.
"""

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_target_users(parquet_dir, LABEL_FILES):
    """Return labelled cohort frames and their shared user index.

    ``LABEL_FILES`` maps exported cohort names to raw labelled parquet stems.
    The March cohort is the evaluation test. Row positions preserve each
    source file's order; a user may occur in both months.
    """
    parquet_dir = Path(parquet_dir)
    labelled = {
        name: (
            pl.read_parquet(
                parquet_dir / f"{file}.parquet",
                columns=["msno", "is_churn"],
            )
            .with_row_index("row")
        )
        for name, file in LABEL_FILES.items()
    }

    union = (
        pl.concat([frame.select("msno") for frame in labelled.values()])
        .unique(maintain_order=True)
    )
    user_index = (
        union.with_row_index("uid")
        .with_columns(pl.col("uid").cast(pl.UInt32))
    )

    for name, frame in labelled.items():
        user_index = user_index.join(
            frame.select("msno", pl.col("row").alias(f"{name}_row")),
            on="msno",
            how="left",
            maintain_order="left",
        )

    return labelled, user_index


def users_with_codes(target_users, user_index):
    """Attach the shared ``uid`` to one cohort's users.

    Returns ``target_users`` as ``msno`` + ``uid`` + ``row`` in the original order.
    """
    return (
        target_users.select("msno", "row")
        .join(
            user_index.select("msno", "uid"),
            on="msno",
            how="left",
            maintain_order="left",
        )
    )
