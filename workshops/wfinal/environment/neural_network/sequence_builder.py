"""Build 90-day LSTM input sequences from raw KKBox customer data."""

from __future__ import annotations

import numpy as np
import pandas as pd

_SEQ_DAYS = 90
_DEFAULT_CUTOFF = "2017-01-31"
_MAX_DAYS = 3650

_LOG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]
_CHANNELS = [
    "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs",
    "active",
    "tx_count", "tx_paid_sum", "tx_plan_days_sum", "tx_cancel_count",
    "tx_auto_renew_count", "tx_state_known", "tx_last_is_cancel",
    "tx_last_is_auto_renew", "tx_days_since_last", "tx_days_to_expire",
    "tx_expiry_known", "tx_state_ambiguous",
]
_CH = {name: i for i, name in enumerate(_CHANNELS)}


def _signed_log1p(x: float) -> float:
    return float(np.sign(x) * np.log1p(abs(x)))


def _parse_date(date_int: int) -> pd.Timestamp:
    return pd.Timestamp(str(int(date_int)))


class SequenceBuilder:
    """Convert a raw customer dict into a (1, 90, 20) float32 array.

    Produces the same encoding as the w2 export-sequential-v5 .npy files:
    - Listening channels (0–7): log1p-transformed counts + active flag.
    - Transaction event channels (8–12): signed_log1p aggregates, only on
      days with transactions.
    - Transaction state channels (13–19): forward-filled from the most
      recent known transaction (including pre-window history).

    The NeuralNetworkModel scaler is applied separately during inference.

    Parameters
    ----------
    cutoff : str
        Observation cutoff date (YYYY-MM-DD).  The 90-day window ends here.
        Matches the development-set cutoff used to train the model (2017-01-31).
    n_days : int
        Length of the sequence window (default 90).
    """

    def __init__(self, cutoff: str = _DEFAULT_CUTOFF, n_days: int = _SEQ_DAYS) -> None:
        self.cutoff = pd.Timestamp(cutoff)
        self.n_days = n_days
        self.window_start = self.cutoff - pd.Timedelta(days=n_days - 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, customer: dict) -> np.ndarray:
        """Return a (1, 90, 20) float32 array from a raw customer dict.

        The dict should have the same structure as CustomerSelector output:
            {
                'transactions': [{'transaction_date': YYYYMMDD, ...}, ...],
                'user_logs':    [{'date': YYYYMMDD, 'num_25': ..., ...}, ...],
            }
        Missing keys default to empty lists (all-zero sequence).
        """
        seq = np.zeros((1, self.n_days, 20), dtype=np.float32)
        self._fill_listening(seq, customer.get("user_logs", []))
        self._fill_transactions(seq, customer.get("transactions", []))
        return seq

    # ------------------------------------------------------------------
    # Listening channels (0–7)
    # ------------------------------------------------------------------

    def _fill_listening(self, seq: np.ndarray, user_logs: list) -> None:
        for log in user_logs:
            d = (_parse_date(log["date"]) - self.window_start).days
            if not (0 <= d < self.n_days):
                continue
            for col in _LOG_COLS:
                seq[0, d, _CH[col]] = float(np.log1p(max(0.0, float(log.get(col, 0)))))
            total = sum(float(log.get(k, 0)) for k in (
                "num_25", "num_50", "num_75", "num_985", "num_100", "total_secs"
            ))
            seq[0, d, _CH["active"]] = 1.0 if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Transaction channels (8–19)
    # ------------------------------------------------------------------

    def _fill_transactions(self, seq: np.ndarray, transactions: list) -> None:
        if not transactions:
            return
        txs = sorted(transactions, key=lambda t: t["transaction_date"])

        # Daily aggregates for in-window events
        daily: dict[int, dict] = {}
        # Last transaction before the window (initialises state on day 0)
        pre_window_last: dict | None = None

        for tx in txs:
            ts = _parse_date(tx["transaction_date"])
            expiry_ts = (
                _parse_date(tx["membership_expire_date"])
                if tx.get("membership_expire_date") else None
            )

            if ts < self.window_start:
                pre_window_last = {
                    "ts": ts,
                    "cancel": int(tx.get("is_cancel", 0)),
                    "renew": int(tx.get("is_auto_renew", 0)),
                    "expiry_ts": expiry_ts,
                    "ambiguous": 0,
                }
                continue
            if ts > self.cutoff:
                continue

            d = (ts - self.window_start).days
            if d not in daily:
                daily[d] = {
                    "tx_count": 0, "tx_paid_sum": 0.0, "tx_plan_days_sum": 0.0,
                    "tx_cancel_count": 0, "tx_auto_renew_count": 0,
                    "last_cancel": 0, "last_renew": 0,
                    "expiry_ts": None, "ambiguous": 0,
                }

            ev = daily[d]
            prev = (ev["last_cancel"], ev["last_renew"], ev["expiry_ts"])
            ev["tx_count"] += 1
            ev["tx_paid_sum"] += float(tx.get("actual_amount_paid", 0))
            ev["tx_plan_days_sum"] += float(tx.get("payment_plan_days", 0))
            ev["tx_cancel_count"] += int(tx.get("is_cancel", 0))
            ev["tx_auto_renew_count"] += int(tx.get("is_auto_renew", 0))
            ev["last_cancel"] = int(tx.get("is_cancel", 0))
            ev["last_renew"] = int(tx.get("is_auto_renew", 0))
            ev["expiry_ts"] = expiry_ts
            if ev["tx_count"] > 1 and prev != (ev["last_cancel"], ev["last_renew"], ev["expiry_ts"]):
                ev["ambiguous"] = 1

        # Forward-fill state channels day by day
        state_known = False
        last_cancel = 0
        last_renew = 0
        last_expiry: pd.Timestamp | None = None
        last_ambiguous = 0
        last_event_day_offset: int | None = None  # relative to window_start

        if pre_window_last is not None:
            state_known = True
            last_cancel = pre_window_last["cancel"]
            last_renew = pre_window_last["renew"]
            last_expiry = pre_window_last["expiry_ts"]
            last_ambiguous = pre_window_last["ambiguous"]
            # Negative offset: transaction was before the window
            last_event_day_offset = -(self.window_start - pre_window_last["ts"]).days

        for d in range(self.n_days):
            current_date = self.window_start + pd.Timedelta(days=d)

            if d in daily:
                ev = daily[d]
                # Event-day channels
                seq[0, d, _CH["tx_count"]] = _signed_log1p(ev["tx_count"])
                seq[0, d, _CH["tx_paid_sum"]] = _signed_log1p(ev["tx_paid_sum"])
                seq[0, d, _CH["tx_plan_days_sum"]] = _signed_log1p(ev["tx_plan_days_sum"])
                seq[0, d, _CH["tx_cancel_count"]] = _signed_log1p(ev["tx_cancel_count"])
                seq[0, d, _CH["tx_auto_renew_count"]] = _signed_log1p(ev["tx_auto_renew_count"])
                # Update forwarded state
                state_known = True
                last_cancel = ev["last_cancel"]
                last_renew = ev["last_renew"]
                last_expiry = ev["expiry_ts"]
                last_ambiguous = ev["ambiguous"]
                last_event_day_offset = d

            if not state_known:
                continue

            seq[0, d, _CH["tx_state_known"]] = 1.0
            seq[0, d, _CH["tx_last_is_cancel"]] = float(last_cancel)
            seq[0, d, _CH["tx_last_is_auto_renew"]] = float(last_renew)
            seq[0, d, _CH["tx_state_ambiguous"]] = float(last_ambiguous)

            days_since = int(np.clip(d - last_event_day_offset, 0, _MAX_DAYS))
            seq[0, d, _CH["tx_days_since_last"]] = float(np.log1p(days_since))

            if last_expiry is not None:
                days_to_exp = float(
                    np.clip((last_expiry - current_date).days, -_MAX_DAYS, _MAX_DAYS)
                )
                seq[0, d, _CH["tx_days_to_expire"]] = _signed_log1p(days_to_exp)
                seq[0, d, _CH["tx_expiry_known"]] = 1.0
