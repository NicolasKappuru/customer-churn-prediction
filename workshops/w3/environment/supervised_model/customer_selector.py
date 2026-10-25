"""Load a random churned KKBox customer from the raw parquet files."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = PROJECT_ROOT / "workshops" / "data" / "parquet"
DEFAULT_POOL_DIR = PROJECT_ROOT / "workshops" / "data" / "pool_users"
_CUTOFF_INT = 20170131

_LOG_DATE_MIN: int = 20161101  # start of the 3-month window used by the preprocessor
_LOG_DATE_MAX: int = 20170131
_LOG_FIELDS: tuple[str, ...] = (
    "msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"
)


def build_log_map(data_dir: Path, msnos: set) -> dict:
    """Return {msno: [log_dict, ...]} for the 3-month window before cutoff.

    Reads user_logs.parquet (and user_logs_v2.parquet if present), keeps only
    rows within the eligible date window and for the requested msnos.
    """
    files = [data_dir / "user_logs.parquet"]
    if (data_dir / "user_logs_v2.parquet").exists():
        files.append(data_dir / "user_logs_v2.parquet")

    frames = []
    for path in files:
        df = pd.read_parquet(path, columns=list(_LOG_FIELDS))
        frames.append(
            df[df["msno"].isin(msnos) & (df["date"] >= _LOG_DATE_MIN) & (df["date"] <= _LOG_DATE_MAX)]
        )

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("msno")
        .apply(lambda g: g.to_dict("records"), include_groups=False)
        .to_dict()
    )


class CustomerSelector:
    """Pick a random churned KKBox user and return their raw source records.

    Loads member data and transactions from parquet files.  User logs are
    omitted because user_logs.parquet is very large (6+ GB); the preprocessor
    handles their absence gracefully.  If you have filtered log data for a
    specific msno, add it under the key 'user_logs' in the returned dict.

    The customer pool is serialised to a pickle file alongside the parquet
    data after the first build so subsequent runs skip the expensive I/O.
    Delete the cache file to force a fresh sample from the raw data.
    """

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR, sample_size: int = 1500) -> None:
        data_dir = Path(data_dir)
        pool_dir = DEFAULT_POOL_DIR
        pool_dir.mkdir(parents=True, exist_ok=True)
        cache_path = pool_dir / f".rl_pool_{sample_size}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self._pool: list[dict] = pickle.load(f)
            print(f"CustomerSelector ready: {len(self._pool)} churned users (cached)")
            return

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

        # Training combined both files and deduplicated — replicate that here.
        _TXN_KEYS = [
            "payment_method_id", "payment_plan_days", "plan_list_price",
            "actual_amount_paid", "is_auto_renew", "transaction_date",
            "membership_expire_date", "is_cancel",
        ]
        txn = pd.concat([
            pd.read_parquet(data_dir / "transactions.parquet"),
            pd.read_parquet(data_dir / "transactions_v2.parquet"),
        ], ignore_index=True)
        txn = txn[txn["transaction_date"] <= _CUTOFF_INT]
        txn = txn.drop_duplicates(subset=["msno", *_TXN_KEYS], keep="first")
        txn_map = txn.groupby("msno").apply(
            lambda g: g.to_dict("records"), include_groups=False
        ).to_dict()

        print("Loading user logs (one-time, may take a few minutes)...")
        log_map = build_log_map(data_dir, set(sample))
        print(f"  Listening logs loaded for {len(log_map)} / {len(sample)} customers")

        self._pool = []
        for msno in sample:
            row = dict(member_map[msno])
            row["transactions"] = txn_map.get(msno, [])
            row["user_logs"] = log_map.get(msno, [])
            self._pool.append(row)

        with open(cache_path, "wb") as f:
            pickle.dump(self._pool, f)
        print(f"CustomerSelector ready: {len(self._pool)} churned users")

    def get_random_customer(self) -> dict:
        """Return one customer dict with member, transaction and log fields."""
        return pd.Series(self._pool).sample(1).iloc[0]
