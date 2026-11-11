"""Customer-side utility model for the KKBox retention environment."""
from __future__ import annotations
import numpy as np
import pandas as pd

# Feature column names as produced by PreprocessorCustomerData (59 XGBoost features)
# Weights are design assumptions — do not change without updating tests.
_V_W  = (0.4, 0.3, 0.1, 0.2)   # last_login, sum_last_month_total_secs, sum_last_month_unq, exp decay
_S_W  = (0.4, 0.4, 0.2)         # discount flag, avg price, n_plans
_L_W  = (0.5, 0.3, 0.2)         # tenure_days, last_is_auto_renew, autorenew_not_cancel
_R_W  = (0.6, 0.4)              # last_is_cancel, is_auto_renew_change_flag

NOISE_STD      = 0.1
FATIGUE_PER_TURN = 0.05

class CustomerUtility:
    """Compute customer acceptance utility from preprocessed features."""

    def __init__(self) -> None:
        self._pct_data: dict = {}

    def fit(self, feature_rows: pd.DataFrame) -> None:
        """Compute pool-level percentile reference distributions."""
        self._pct_data = {}
        for col in feature_rows.columns:
            vals = feature_rows[col].dropna().values
            self._pct_data[col] = np.sort(vals) if len(vals) > 0 else np.array([0.0])

    def _pct(self, col: str, x: float) -> float:
        """Return percentile rank of x in the pool distribution for col (0..1)."""
        ref = self._pct_data.get(col)
        if ref is None or len(ref) == 0:
            return 0.5
        return float(np.searchsorted(ref, x, side="right") / len(ref))

    def indices(self, row: pd.Series) -> dict:
        """Return V, S, L, R indices in [0, 1]."""
        def g(col, default=0.0):
            v = row.get(col, default)
            return float(v) if v is not None else default

        # V — engagement / recency
        last_login = self._pct("last_login", g("last_login"))
        secs       = self._pct("sum_last_month_total_secs", g("sum_last_month_total_secs"))
        unq        = self._pct("sum_last_month_unq", g("sum_last_month_unq"))
        days       = g("days_since_last_log", 999)
        exp_decay  = float(np.exp(-days / 30))
        V = _V_W[0]*last_login + _V_W[1]*secs + _V_W[2]*unq + _V_W[3]*exp_decay

        # S — price sensitivity
        diff_flag  = float(g("difference") < 0 or g("last_discount") > 0)
        avg_paid   = 1.0 - self._pct("average_amount_paid", g("average_amount_paid"))
        n_plans    = min(1.0, (g("n_plans", 1) - 1) / 2)
        S = _S_W[0]*diff_flag + _S_W[1]*avg_paid + _S_W[2]*n_plans

        # L — loyalty
        tenure     = self._pct("tenure_days", g("tenure_days"))
        auto_renew = float(g("last_is_auto_renew"))
        ar_nc      = float(g("autorenew_not_cancel", 0))
        L = _L_W[0]*tenure + _L_W[1]*auto_renew + _L_W[2]*ar_nc

        # R — cancellation risk
        is_cancel  = float(g("last_is_cancel"))
        ar_change  = float(g("is_auto_renew_change_flag", 0))
        R = _R_W[0]*is_cancel + _R_W[1]*ar_change

        return {"V": V, "S": S, "L": L, "R": R}

    def utility(self, action: int, row: pd.Series, turn: int) -> float:
        """Deterministic utility of accepting action on this turn (1-indexed)."""
        idx = self.indices(row)
        V, S, L, R = idx["V"], idx["S"], idx["L"], idx["R"]

        def g(col, default=0.0):
            v = row.get(col, default)
            return float(v) if v is not None else default

        if action == 0:   # Reminder
            u = 0.4*V + 0.2*L - 0.3*(1 - V)
        elif action == 1: # Loyalty
            u = V - 0.87*(0.5 + 0.5*S) + 0.2*L
        elif action == 2: # Re-engagement
            unq_pct = self._pct("sum_last_month_unq", g("sum_last_month_unq"))
            u = 0.3*unq_pct + 0.1
        elif action == 3: # Downgrade 99
            u = V - 0.66*(0.5 + 0.5*S) - 0.15*V
        elif action == 4: # Recover cancel
            u = V - 0.87*(0.5 + 0.5*S) - 0.4*R
        elif action == 5: # Win-back
            # use last_second_login as proxy if sum_last_month_unq not useful
            listen_pct = self._pct("last_second_login", g("last_second_login"))
            u = 0.3*listen_pct - 0.1
        else:
            u = 0.0

        u -= FATIGUE_PER_TURN * (turn - 1)
        return float(u)

    def sample_utility(self, action: int, row: pd.Series, turn: int, rng: np.random.Generator) -> float:
        """Utility with Gaussian noise."""
        return self.utility(action, row, turn) + float(rng.normal(0, NOISE_STD))
