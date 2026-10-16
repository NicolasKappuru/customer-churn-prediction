"""Users domain.

Loading the labelled cohorts and test users, and attaching the shared ``uid``
to each cohort.
"""

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_target_users(parquet_dir, LABEL_FILES):
    """Load the labelled cohorts and test users and build the ``msno`` -> ``uid`` union index.

    Parameters
    ----------
    parquet_dir : str | Path
        Directory with the converted parquet files.
    LABEL_FILES : dict
        Mapping ``cohort name -> labelled parquet stem`` (e.g. ``{"train_feb": "train"}``).

    Returns
    -------
    tuple
        ``(labelled, test, user_index)`` where ``labelled`` maps cohort name -> frame with
        ``msno``, ``is_churn`` and ``row``. ``uid`` is assigned over the union of all cohorts;
        ``<cohort>_row`` holds the position of the user in that cohort's source file (null when
        absent), because the competition files overlap.
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

    test = (
        pl.read_parquet(
            parquet_dir / "sample_submission_v2.parquet",
            columns=["msno"],
        )
        .with_row_index("row")
    )

    cohorts = {**labelled, "test": test}
    union = (
        pl.concat([frame.select("msno") for frame in cohorts.values()])
        .unique(maintain_order=True)
    )
    user_index = (
        union.with_row_index("uid")
        .with_columns(pl.col("uid").cast(pl.UInt32))
    )

    for name, frame in cohorts.items():
        user_index = user_index.join(
            frame.select("msno", pl.col("row").alias(f"{name}_row")),
            on="msno",
            how="left",
            maintain_order="left",
        )

    return labelled, test, user_index


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
