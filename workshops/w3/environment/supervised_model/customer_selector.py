"""Load a random churned KKBox customer from the raw parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = PROJECT_ROOT / "workshops" / "data" / "parquet"
_CUTOFF_INT = 20170131


class CustomerSelector:
    """Pick a random churned KKBox user and return their raw source records.

    Loads member data and transactions from parquet files.  User logs are
    omitted because user_logs.parquet is very large (6+ GB); the preprocessor
    handles their absence gracefully.  If you have filtered log data for a
    specific msno, add it under the key 'user_logs' in the returned dict.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, sample_size: int = 500) -> None:
        data_dir = Path(data_dir)

        train = pd.read_parquet(data_dir / "train.parquet")
        churned_msno = train.loc[train["is_churn"] == 1, "msno"].tolist()

        members = pd.read_parquet(
            data_dir / "members_v3.parquet",
            columns=["msno", "city", "registered_via", "bd", "registration_init_time"],
        )
        member_map = members.set_index("msno").to_dict("index")

        available = [m for m in churned_msno if m in member_map]
        sample = (
            pd.Series(available)
            .sample(min(sample_size, len(available)), replace=False)
            .tolist()
        )

        txn = pd.read_parquet(data_dir / "transactions_v2.parquet")
        txn = txn[txn["transaction_date"] <= _CUTOFF_INT]
        txn_map = txn.groupby("msno").apply(
            lambda g: g.to_dict("records"), include_groups=False
        ).to_dict()

        self._pool = []
        for msno in sample:
            row = dict(member_map[msno])
            row["transactions"] = txn_map.get(msno, [])
            row["user_logs"] = []
            self._pool.append(row)

        print(f"CustomerSelector ready: {len(self._pool)} churned users")

    def get_random_customer(self) -> dict:
        """Return one customer dict with member, transaction and log fields."""
        return pd.Series(self._pool).sample(1).iloc[0]
