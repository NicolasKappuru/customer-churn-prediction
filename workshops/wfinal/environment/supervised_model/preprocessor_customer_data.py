"""Prepare raw KKBox customer records for the trained churn model (M3 XGBoost).

The 59 model features come from three data sources; only raw fields are passed
in — all engineered features are computed here, replicating the logic in
workshops/w1/src/data_processing/.

RAW INPUTS (what the caller provides)
======================================
Member fields (top-level keys in the customer dict):
  city                   int   – city code from members_v3
  registered_via         int   – registration channel code
  bd                     int   – age (0 or missing = unknown)
  registration_init_time int   – registration date as YYYYMMDD int

Transaction records (key "transactions", list of dicts, may be empty):
  payment_method_id      int
  payment_plan_days      int
  plan_list_price        float
  actual_amount_paid     float
  is_auto_renew          0/1
  is_cancel              0/1
  transaction_date       int   – YYYYMMDD
  membership_expire_date int   – YYYYMMDD

User log records (key "user_logs", list of dicts, may be empty):
  date      int   – YYYYMMDD
  num_25    int   – songs listened to 0–25 %
  num_50    int   – songs listened to 25–50 %
  num_75    int   – songs listened to 50–75 %
  num_985   int   – songs listened to 75–98.5 %
  num_100   int   – songs listened to completion
  num_unq   int   – unique songs
  total_secs float – total listening seconds

ENGINEERED FEATURES (computed here, never passed by the caller)
=================================================================
From member:
  bd_missing, tenure_days, registration_year, registration_month, has_member_record

From transactions (aggregates over all records up to cutoff):
  is_cancel_change_flag, is_auto_renew_change_flag, transaction_count,
  average_amount_paid/charged/plan, n_payment_methods/plans/list_prices/amounts_paid,
  duration, plan_coverage_ratio, sum_plan_days/list_price/amount_paid,
  more_than_30, autorenew_not_cancel, not_autorenew_cancel, max_plan_days,
  last_payment_method_id/plan_days/is_auto_renew/is_cancel/discount,
  days_to_expire, expire_dayofweek, days_since_last_transaction, difference,
  has_transactions

From user logs (monthly aggregates for the 3 calendar months ending at cutoff):
  sum_last_month_unq, sum_last_month_total_secs, mean_last_month_total_secs,
  last_login, last_sum_total_songs, last_ratio_100, last_ratio_25,
  sum_last_second_month_total_secs, last_second_login, last_second_sum_total_songs,
  last_second_ratio_100/25, last_third_sum_total_songs, last_third_ratio_100/25,
  login_difference, trend_songs/secs/ratio100/ratio25,
  days_since_last_log, has_listening_logs
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "workshops"
    / "w1"
    / "artifacts"
    / "sherlock_w1"
    / "preprocessing"
    / "development.json"
)
DEFAULT_CUTOFF = "2017-01-31"

_LOG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]

class PreprocessorCustomerData:
    """Compute all 59 M3 features from raw member + transaction + log records.

    Usage
    -----
    preprocessor = PreprocessorCustomerData()
    features = preprocessor.preprocess_customer({
        "city": 1, "registered_via": 7, "bd": 25,
        "registration_init_time": 20150301,
        "transactions": [...],
        "user_logs": [...],
    })
    """

    def __init__(
        self,
        preprocessing_path: str | Path = DEFAULT_PREPROCESSING_PATH,
        cutoff: str = DEFAULT_CUTOFF,
    ) -> None:
        self.preprocessing_path = Path(preprocessing_path).expanduser()
        if not self.preprocessing_path.is_absolute():
            self.preprocessing_path = PROJECT_ROOT / self.preprocessing_path
        if not self.preprocessing_path.is_file():
            raise FileNotFoundError(
                f"Preprocessing state not found: {self.preprocessing_path}"
            )

        with self.preprocessing_path.open() as fh:
            state = json.load(fh)

        required_keys = {"features", "categorical", "source_preprocess_params"}
        missing = required_keys.difference(state)
        if missing:
            raise ValueError(
                f"Preprocessing state is missing required keys: {', '.join(sorted(missing))}"
            )

        self.feature_names: tuple[str, ...] = tuple(state["features"])
        self.source_params: dict = state["source_preprocess_params"]
        self.cutoff: pd.Timestamp = pd.Timestamp(cutoff)

        # Sentinel for users with no listening activity
        window_start = (self.cutoff.to_period("M") - 2).start_time
        self._log_sentinel = int(
            max(90, (self.cutoff - window_start).days + 1) + 1
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess_customer(self, customer: Mapping[str, Any]) -> pd.DataFrame:
        """Return all 59 M3 features for one customer as a one-row DataFrame.

        Parameters
        ----------
        customer : dict with keys:
            city, registered_via, bd, registration_init_time   (member)
            transactions : list[dict]   (raw transaction records, optional)
            user_logs    : list[dict]   (raw daily log records, optional)
        """
        if not isinstance(customer, Mapping):
            raise TypeError("customer must be a mapping.")

        features: dict[str, Any] = {}
        features.update(self._member_features(customer))
        features.update(self._transaction_features(customer.get("transactions") or []))
        features.update(self._log_features(customer.get("user_logs") or []))

        df = pd.DataFrame(
            [{name: features[name] for name in self.feature_names}]
        )
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        if not np.isfinite(df.to_numpy(dtype=np.float32)).all():
            raise ValueError("Computed features contain non-finite values.")
        return df

    # ------------------------------------------------------------------
    # Member features
    # ------------------------------------------------------------------

    def _member_features(self, customer: Mapping[str, Any]) -> dict:
        city = int(customer.get("city", 1))

        # registered_via: impute negative values with training mode
        rv = int(customer.get("registered_via") or 0)
        registered_via = rv if rv >= 0 else int(self.source_params["registered_via_mode"])

        # age: valid range 1–100; out-of-range → training mean + flag
        bd_raw = customer.get("bd")
        try:
            bd_val = float(bd_raw)
        except (TypeError, ValueError):
            bd_val = 0.0
        bd_valid = 1.0 <= bd_val <= 100.0
        bd = bd_val if bd_valid else float(self.source_params["age_mean"])
        bd_missing = 0 if bd_valid else 1

        # registration date
        rit = customer.get("registration_init_time")
        if rit is not None:
            reg_ts = pd.to_datetime(str(int(rit)), format="%Y%m%d")
            tenure_days = max(0, (self.cutoff - reg_ts).days)
            registration_year = int(reg_ts.year)
            registration_month = int(reg_ts.month)
        else:
            tenure_days, registration_year, registration_month = 0, 2013, 1

        return {
            "city": city,
            "registered_via": registered_via,
            "bd": bd,
            "bd_missing": bd_missing,
            "tenure_days": tenure_days,
            "registration_year": registration_year,
            "registration_month": registration_month,
            "has_member_record": 1,
        }

    # ------------------------------------------------------------------
    # Transaction features
    # ------------------------------------------------------------------

    def _transaction_features(self, transactions: list) -> dict:
        zeros = {
            "is_cancel_change_flag": 0, "is_auto_renew_change_flag": 0,
            "transaction_count": 0,
            "average_amount_paid": 0.0, "average_amount_charged": 0.0,
            "average_plan": 0.0,
            "n_payment_methods": 0, "n_plans": 0,
            "n_list_prices": 0, "n_amounts_paid": 0,
            "duration": 0.0, "plan_coverage_ratio": 0.0,
            "sum_plan_days": 0, "sum_list_price": 0.0, "sum_amount_paid": 0.0,
            "more_than_30": 0, "autorenew_not_cancel": 0, "not_autorenew_cancel": 0,
            "max_plan_days": 0,
            "last_payment_method_id": 0, "last_plan_days": 0,
            "last_is_auto_renew": 0, "last_is_cancel": 0, "last_discount": 0.0,
            "days_to_expire": 0, "expire_dayofweek": 0,
            "days_since_last_transaction": 0,
            "difference": 0.0, "has_transactions": 0,
        }
        if not transactions:
            return zeros

        rows = []
        for txn in transactions:
            td = pd.to_datetime(str(int(txn["transaction_date"])), format="%Y%m%d")
            ed = pd.to_datetime(str(int(txn["membership_expire_date"])), format="%Y%m%d")
            ppd = int(txn.get("payment_plan_days", 0))
            llp = float(txn.get("plan_list_price", 0))
            aap = float(txn.get("actual_amount_paid", 0))
            is_ar = int(txn.get("is_auto_renew", 0))
            is_c = int(txn.get("is_cancel", 0))
            dur = int((ed - td).days)
            rows.append({
                "td": td, "ed": ed,
                "payment_method_id": int(txn.get("payment_method_id", 0)),
                "payment_plan_days": ppd,
                "plan_list_price": llp,
                "actual_amount_paid": aap,
                "is_auto_renew": is_ar,
                "is_cancel": is_c,
                "duration": dur,
                "plan_coverage_ratio": dur / ppd if ppd > 0 else None,
                "more_than_30": int(dur > 30),
                "autorenew_not_cancel": int(is_ar == 1 and is_c == 0),
                "not_autorenew_cancel": int(is_ar == 0 and is_c == 1),
                "discount": llp - aap,
            })

        # Sort by transaction date; last = most recent
        rows.sort(key=lambda r: r["td"])
        last = rows[-1]
        n = len(rows)

        avg_paid = sum(r["actual_amount_paid"] for r in rows) / n
        avg_charged = sum(r["plan_list_price"] for r in rows) / n
        valid_pcr = [r["plan_coverage_ratio"] for r in rows if r["plan_coverage_ratio"] is not None]

        return {
            "is_cancel_change_flag": max(r["is_cancel"] for r in rows),
            "is_auto_renew_change_flag": max(r["is_auto_renew"] for r in rows),
            "transaction_count": n,
            "average_amount_paid": avg_paid,
            "average_amount_charged": avg_charged,
            "average_plan": sum(r["payment_plan_days"] for r in rows) / n,
            "n_payment_methods": len({r["payment_method_id"] for r in rows}),
            "n_plans": len({r["payment_plan_days"] for r in rows}),
            "n_list_prices": len({r["plan_list_price"] for r in rows}),
            "n_amounts_paid": len({r["actual_amount_paid"] for r in rows}),
            "duration": sum(r["duration"] for r in rows) / n,
            "plan_coverage_ratio": sum(valid_pcr) / len(valid_pcr) if valid_pcr else 0.0,
            "sum_plan_days": sum(r["payment_plan_days"] for r in rows),
            "sum_list_price": sum(r["plan_list_price"] for r in rows),
            "sum_amount_paid": sum(r["actual_amount_paid"] for r in rows),
            "more_than_30": sum(r["more_than_30"] for r in rows),
            "autorenew_not_cancel": min(r["autorenew_not_cancel"] for r in rows),
            "not_autorenew_cancel": min(r["not_autorenew_cancel"] for r in rows),
            "max_plan_days": max(r["payment_plan_days"] for r in rows),
            "last_payment_method_id": last["payment_method_id"],
            "last_plan_days": last["payment_plan_days"],
            "last_is_auto_renew": last["is_auto_renew"],
            "last_is_cancel": last["is_cancel"],
            "last_discount": last["discount"],
            "days_to_expire": int((last["ed"] - self.cutoff).days),
            "expire_dayofweek": int(last["ed"].dayofweek),
            "days_since_last_transaction": int((self.cutoff - last["td"]).days),
            "difference": avg_paid - avg_charged,
            "has_transactions": 1,
        }

    # ------------------------------------------------------------------
    # User log features
    # ------------------------------------------------------------------

    def _log_features(self, user_logs: list) -> dict:
        zeros = {
            "sum_last_month_unq": 0, "sum_last_month_total_secs": 0.0,
            "mean_last_month_total_secs": 0.0,
            "last_login": 0, "last_sum_total_songs": 0,
            "last_ratio_100": 0.0, "last_ratio_25": 0.0,
            "sum_last_second_month_total_secs": 0.0,
            "last_second_login": 0, "last_second_sum_total_songs": 0,
            "last_second_ratio_100": 0.0, "last_second_ratio_25": 0.0,
            "last_third_sum_total_songs": 0,
            "last_third_ratio_100": 0.0, "last_third_ratio_25": 0.0,
            "login_difference": 0, "trend_songs": 0.0, "trend_secs": 0.0,
            "trend_ratio100": 0.0, "trend_ratio25": 0.0,
            "days_since_last_log": self._log_sentinel, "has_listening_logs": 0,
        }
        if not user_logs:
            return zeros

        # Eligible months: 3 calendar months ending at cutoff.
        # rank 1 = most recent (e.g. 201701), rank 2 = previous, rank 3 = oldest.
        last_period = self.cutoff.to_period("M")
        eligible: dict[int, int] = {
            int((last_period - offset).year * 100 + (last_period - offset).month): rank
            for offset, rank in [(0, 1), (1, 2), (2, 3)]
        }

        clip = self.source_params["log_clip_p999"]

        # Accumulate sums + count per eligible month
        monthly: dict[int, dict] = {}
        for log in user_logs:
            date_int = int(log["date"])
            month = date_int // 100
            if month not in eligible:
                continue
            if month not in monthly:
                monthly[month] = {f"sum_{c}": 0.0 for c in _LOG_COLS}
                monthly[month]["count"] = 0
                monthly[month]["last_date"] = 0
            m = monthly[month]
            for c in _LOG_COLS:
                m[f"sum_{c}"] += float(log.get(c) or 0)
            m["count"] += 1
            m["last_date"] = max(m["last_date"], date_int)

        # Build per-rank feature dict, clipping means then rebuilding sums
        rank_data: dict[int, dict | None] = {1: None, 2: None, 3: None}
        for month, rank in eligible.items():
            if month not in monthly:
                continue
            m = monthly[month]
            count = m["count"]
            clipped_means = {
                c: min(m[f"sum_{c}"] / count, clip.get(c, float("inf")))
                for c in _LOG_COLS
            }
            rebuilt_sums = {
                c: round(clipped_means[c] * count) for c in _LOG_COLS
            }
            total_songs = sum(rebuilt_sums[c] for c in ["num_25", "num_50", "num_75", "num_985", "num_100"])
            ratio_100 = rebuilt_sums["num_100"] / total_songs if total_songs > 0 else 0.0
            ratio_25 = rebuilt_sums["num_25"] / total_songs if total_songs > 0 else 0.0
            rank_data[rank] = {
                "sum_unq": rebuilt_sums["num_unq"],
                "sum_total_secs": rebuilt_sums["total_secs"],
                "mean_total_secs": clipped_means["total_secs"],
                "login": count,
                "sum_total_songs": total_songs,
                "ratio_100": ratio_100,
                "ratio_25": ratio_25,
                "last_date": m["last_date"],
            }

        def _get(rank: int, key: str, default: float = 0.0) -> Any:
            d = rank_data[rank]
            return d[key] if d is not None else default

        # days_since_last_log from rank-1's last login date
        last_date_int = _get(1, "last_date", 0)
        if last_date_int:
            last_date_ts = pd.to_datetime(str(int(last_date_int)), format="%Y%m%d")
            days_since_last_log = int((self.cutoff - last_date_ts).days)
            has_listening_logs = 1
        else:
            days_since_last_log = self._log_sentinel
            has_listening_logs = 0

        return {
            "sum_last_month_unq": _get(1, "sum_unq"),
            "sum_last_month_total_secs": _get(1, "sum_total_secs"),
            "mean_last_month_total_secs": _get(1, "mean_total_secs"),
            "last_login": int(_get(1, "login")),
            "last_sum_total_songs": int(_get(1, "sum_total_songs")),
            "last_ratio_100": _get(1, "ratio_100"),
            "last_ratio_25": _get(1, "ratio_25"),
            "sum_last_second_month_total_secs": _get(2, "sum_total_secs"),
            "last_second_login": int(_get(2, "login")),
            "last_second_sum_total_songs": int(_get(2, "sum_total_songs")),
            "last_second_ratio_100": _get(2, "ratio_100"),
            "last_second_ratio_25": _get(2, "ratio_25"),
            "last_third_sum_total_songs": int(_get(3, "sum_total_songs")),
            "last_third_ratio_100": _get(3, "ratio_100"),
            "last_third_ratio_25": _get(3, "ratio_25"),
            "login_difference": int(_get(1, "login")) - int(_get(2, "login")),
            "trend_songs": _get(1, "sum_total_songs") / (_get(2, "sum_total_songs") + 1),
            "trend_secs": _get(1, "sum_total_secs") / (_get(2, "sum_total_secs") + 1),
            "trend_ratio100": _get(1, "ratio_100") - _get(2, "ratio_100"),
            "trend_ratio25": _get(1, "ratio_25") - _get(2, "ratio_25"),
            "days_since_last_log": days_since_last_log,
            "has_listening_logs": has_listening_logs,
        }
