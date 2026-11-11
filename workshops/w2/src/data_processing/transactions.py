"""Transactions domain: load and deduplicate both transaction releases."""

from pathlib import Path

import polars as pl

from w2.src.data_processing.constants import _TXN_KEYS
from w2.src.data_processing.utils import _date_int


# ---------------------------------------------------------------------------
# Load + clean (deduplication)
# ---------------------------------------------------------------------------

def scan_transactions(parquet_dir, user_index, max_date):
    """Lazy transactions (both files, deduplicated) up to ``max_date``, keyed by ``uid``.

    Loads ``transactions`` and ``transactions_v2``, keeps target users, drops
    records after ``max_date`` and removes exact duplicates using ``msno`` plus
    all :data:`_TXN_KEYS` fields (the first copy is kept). ``_ord`` records the
    stable source order and is used later as a tie break, not as intraday time.
    """
    parquet_dir = Path(parquet_dir)
    frames = [
        pl.scan_parquet(parquet_dir / "transactions.parquet"),
        pl.scan_parquet(parquet_dir / "transactions_v2.parquet"),
    ]

    return (
        pl.concat(frames)
        .with_row_index("_ord")
        .join(
            user_index.lazy().select("msno", "uid"),
            on="msno",
            how="inner",
            maintain_order="left",
        )
        .filter(pl.col("transaction_date") <= _date_int(max_date))
        .unique(
            subset=["msno", *_TXN_KEYS],
            keep="first",
            maintain_order=True,
        )
        .select("uid", "_ord", *_TXN_KEYS)
    )
