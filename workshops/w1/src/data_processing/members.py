"""Members domain.

Loading the member table, learning its imputation values from the training
cohort, and cleaning/transforming it into static user features.
"""

from pathlib import Path

import polars as pl

from w1.src.data_processing.utils import _date


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def scan_members(parquet_dir, user_index):
    """Lazy members table filtered to target users, keyed by ``uid``.

    Keeps only the columns used as static features (city, registered_via, bd,
    registration date); the join to ``user_index`` maps ``msno`` -> ``uid``.
    """
    return (
        pl.scan_parquet(Path(parquet_dir) / "members_v3.parquet")
        .join(
            user_index.lazy().select("msno", "uid"),
            on="msno",
            how="inner",
        )
        .select(
            "uid",
            "city",
            "registered_via",
            "bd",
            "registration_init_time",
        )
    )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_member_params(members, train_users):
    """Learn member imputation values from the training cohort only.

    Returns ``age_mean`` (mean valid age, 1..100) and ``registered_via_mode``
    (mode of the non-negative ``registered_via`` values).
    """
    members_train = members.join(train_users.select("uid"), on="uid", how="semi")

    age_mean = float(
        members_train.filter((pl.col("bd") > 0) & (pl.col("bd") <= 100))
                     .select(pl.col("bd").mean()).item())
    via_mode = int(
        members_train.filter(pl.col("registered_via") >= 0)
                     .select(pl.col("registered_via").mode().first()).item())

    return {"age_mean": age_mean, "registered_via_mode": via_mode}


# ---------------------------------------------------------------------------
# Clean + transform
# ---------------------------------------------------------------------------

def _clean_members(members, params):
    """Clean members: impute invalid age/source values (MOD-01 cleaning step).

    ``registered_via < 0`` gets the train mode (``params["registered_via_mode"]``);
    null ages and ages outside 1..100 get the train mean (``params["age_mean"]``) and are
    flagged with ``bd_missing=1``.
    """
    valid_age = pl.col("bd").is_between(1, 100).fill_null(False)
    return members.select(
        pl.col("uid"),
        pl.col("city"),
        (
            pl.when(pl.col("registered_via") >= 0)
            .then(pl.col("registered_via"))
            .otherwise(params["registered_via_mode"])
            .cast(pl.Int16)
            .alias("registered_via")
        ),
        (
            pl.when(valid_age)
            .then(pl.col("bd"))
            .otherwise(params["age_mean"])
            .cast(pl.Float32)
            .alias("bd")
        ),
        (
            (~valid_age)
            .cast(pl.Int8)
            .alias("bd_missing")
        ),
        pl.col("registration_init_time"),
    )


def build_member_features(members, users, params, cutoff):
    """Clean members and build the static member features (MOD-01).

    Transform: registration date is parsed and yields ``tenure_days`` at the
    cutoff plus ``registration_year`` / ``registration_month``.
    """
    registration_date = (
        pl.col("registration_init_time")
        .cast(pl.String)
        .str.to_date("%Y%m%d")
    )

    return (
        _clean_members(members.join(users.select("uid"), on="uid", how="semi"), params)
        .select(
            pl.col("uid"),
            pl.col("city"),
            pl.col("registered_via"),
            pl.col("bd"),
            pl.col("bd_missing"),
            (
                (pl.lit(_date(cutoff)) - registration_date)
                .dt.total_days()
                .cast(pl.Int32)
                .alias("tenure_days")
            ),
            registration_date.dt.year().cast(pl.Int16).alias("registration_year"),
            registration_date.dt.month().cast(pl.Int8).alias("registration_month"),
        )
    )
