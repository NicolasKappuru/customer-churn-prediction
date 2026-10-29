"""Shared test-pool builder that excludes training/validation customers by fingerprint."""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
W3_ROOT      = PROJECT_ROOT / "workshops" / "w3"
POOL_DIR     = PROJECT_ROOT / "workshops" / "data" / "pool_users"
DATA_DIR     = PROJECT_ROOT / "workshops" / "data" / "parquet"

for _path in (str(W3_ROOT), str(W3_ROOT / "environment")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from environment.supervised_model.customer_selector import build_log_map

_CUTOFF_INT = 20170131
_TXN_KEYS = (
    "payment_method_id", "payment_plan_days", "plan_list_price",
    "actual_amount_paid", "is_auto_renew", "transaction_date",
    "membership_expire_date", "is_cancel",
)
_MEMBER_KEYS = ("city", "registered_via", "bd", "registration_init_time")


def _to_python(v: object) -> object:
    """Convert numpy scalars to native Python types for stable JSON serialization."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def customer_fingerprint(customer: dict) -> str:
    """SHA-256 hash of member fields + normalised sorted transactions.

    Excludes user_logs and msno: the training pool does not store msno, and
    logs differ across preprocessing runs and can vary with action simulation.
    """
    member = {k: _to_python(customer.get(k)) for k in _MEMBER_KEYS}
    txns = [
        {k: _to_python(t.get(k)) for k in _TXN_KEYS}
        for t in customer.get("transactions", [])
    ]
    txns_sorted = sorted(txns, key=lambda t: [str(t.get(k, "")) for k in _TXN_KEYS])
    payload = {"member": member, "transactions": txns_sorted}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_training_fingerprints() -> frozenset:
    """Return fingerprints for the 1 500-customer training + validation pool."""
    cache = POOL_DIR / ".rl_pool_1500.pkl"
    if not cache.exists():
        raise FileNotFoundError(
            f"Training pool cache not found: {cache}\n"
            "Run a training script once to build it."
        )
    with open(cache, "rb") as f:
        pool: list[dict] = pickle.load(f)
    return frozenset(customer_fingerprint(c) for c in pool)


def load_or_create_test_pools(
    n_pools: int = 5,
    n_churn: int = 90,
    n_non_churn: int = 10,
    base_seed: int = 0,
) -> list[list[dict]]:
    """Load or build mixed test pools, excluding training/validation customers.

    Files are saved as  pool_mixed_{n}c_{m}nc_{i}_ex.pkl  ('_ex' = excluded)
    so they never collide with the original unfiltered pools.
    Pools are disjoint from each other: customers are pre-selected in one
    pass and sliced, so the same customer never appears in two pools.
    """
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        POOL_DIR / f"pool_mixed_{n_churn}c_{n_non_churn}nc_{i}_ex.pkl"
        for i in range(n_pools)
    ]
    if all(p.exists() for p in paths):
        pools: list[list[dict]] = []
        for p in paths:
            with open(p, "rb") as f:
                pools.append(pickle.load(f))
        print(
            f"Loaded {n_pools} test pools "
            f"({n_churn} churn + {n_non_churn} non-churn, fingerprint-excluded) "
            f"from {POOL_DIR}"
        )
        return pools

    # ── Build from scratch ──────────────────────────────────────────────────
    print("Building test pools with fingerprint exclusion (one-time)…")
    training_fps = _load_training_fingerprints()
    print(f"  Training pool fingerprints: {len(training_fps)} customers")

    train = pd.read_parquet(DATA_DIR / "train.parquet")
    churned_msno     = train.loc[train["is_churn"] == 1, "msno"].tolist()
    non_churned_msno = train.loc[train["is_churn"] == 0, "msno"].tolist()

    members = pd.read_parquet(
        DATA_DIR / "members_v3.parquet",
        columns=["msno", "city", "registered_via", "bd", "registration_init_time"],
    )
    member_map = members.set_index("msno").to_dict("index")

    available_churn     = [m for m in churned_msno     if m in member_map]
    available_non_churn = [m for m in non_churned_msno if m in member_map]
    print(
        f"  Candidates before exclusion: "
        f"{len(available_churn)} churned, {len(available_non_churn)} non-churned"
    )

    txn = pd.concat([
        pd.read_parquet(DATA_DIR / "transactions.parquet"),
        pd.read_parquet(DATA_DIR / "transactions_v2.parquet"),
    ], ignore_index=True)
    txn = txn[txn["transaction_date"] <= _CUTOFF_INT]
    txn = txn.drop_duplicates(subset=["msno", *_TXN_KEYS], keep="first")
    txn_map = (
        txn.groupby("msno")
        .apply(lambda g: g.to_dict("records"), include_groups=False)
        .to_dict()
    )

    # ── Fingerprint exclusion (member + transactions only, no logs needed) ──
    def _keep(msno: str) -> bool:
        row = dict(member_map[msno])
        row["transactions"] = txn_map.get(msno, [])
        return customer_fingerprint(row) not in training_fps

    clean_churn     = [m for m in available_churn     if _keep(m)]
    clean_non_churn = [m for m in available_non_churn if _keep(m)]

    excl_churn     = len(available_churn)     - len(clean_churn)
    excl_non_churn = len(available_non_churn) - len(clean_non_churn)
    print(
        f"  Excluded (training overlap): "
        f"{excl_churn} churned, {excl_non_churn} non-churned"
    )
    print(
        f"  Remaining after exclusion: "
        f"{len(clean_churn)} churned, {len(clean_non_churn)} non-churned"
    )

    # ── Pre-select all customers for all pools (disjoint) ──────────────────
    n_c_needed  = n_churn     * n_pools
    n_nc_needed = n_non_churn * n_pools
    if len(clean_churn) < n_c_needed:
        raise ValueError(
            f"Not enough churned candidates after exclusion: "
            f"need {n_c_needed}, have {len(clean_churn)}"
        )
    if len(clean_non_churn) < n_nc_needed:
        raise ValueError(
            f"Not enough non-churned candidates after exclusion: "
            f"need {n_nc_needed}, have {len(clean_non_churn)}"
        )

    rng = np.random.RandomState(base_seed)
    selected_churn     = [clean_churn[i]     for i in rng.permutation(len(clean_churn))    [:n_c_needed]]
    selected_non_churn = [clean_non_churn[i] for i in rng.permutation(len(clean_non_churn))[:n_nc_needed]]

    # ── Load logs only for the selected customers (efficient) ──────────────
    all_selected = set(selected_churn) | set(selected_non_churn)
    print(f"  Loading user logs for {len(all_selected)} selected customers…")
    log_map = build_log_map(DATA_DIR, all_selected)
    print(f"  Logs loaded for {len(log_map)} customers")

    # ── Build, shuffle and save each pool ──────────────────────────────────
    pools = []
    for i in range(n_pools):
        churn_batch = selected_churn    [i * n_churn     : (i + 1) * n_churn]
        nc_batch    = selected_non_churn[i * n_non_churn : (i + 1) * n_non_churn]

        pool: list[dict] = []
        for msno in churn_batch + nc_batch:
            row = dict(member_map[msno])
            row["msno"]         = msno
            row["transactions"] = txn_map.get(msno, [])
            row["user_logs"]    = log_map.get(msno, [])
            pool.append(row)

        random.shuffle(pool)
        with open(paths[i], "wb") as pf:
            pickle.dump(pool, pf)
        pools.append(pool)
        print(
            f"  Pool {i}: {len(churn_batch)} churn + {len(nc_batch)} non-churn "
            f"= {len(pool)} customers saved"
        )

    return pools


def use_pool(env: object, pool: list[dict]) -> None:
    """Assign a test pool to the environment and clear the observation cache."""
    env.customer_selector._pool = pool  # type: ignore[attr-defined]
    env._obs_cache               = None  # type: ignore[attr-defined]
