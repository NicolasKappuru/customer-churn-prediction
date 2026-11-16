"""Test evaluation of the trajectory models against naive baselines.

Everything is computed in the **raw export encoding** (listening in log1p,
amounts converted back to NTD / days), never in the scaled space, so the
numbers can be read directly. Test rows are streamed in blocks: the March
tensor does not fit in memory as float32.

Groups (each compared with a predictor that learns nothing):

- listening amounts: MAE / RMSE, skill vs persistence, by horizon, by channel,
  on active vs inactive days and on users with logs;
- ``active`` and ``pay``: AUC / AP per day, plus weekly payment;
- paid amount and plan days: MAE on real payment days vs last paid value.

Simulated weeks (sampling events and rebuilding the state, option B) are not
evaluated here: the simulator is not part of this package.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..feature_preparation.scaling.helpers import channel_index
from .config import (
    EVENT_AMOUNT_CHANNELS,
    HORIZON,
    INPUT_DAYS,
    LISTEN_AMOUNT_CHANNELS,
    SEED,
)
from .decode import _inv_slog, decode_last_state, expected_listening, unscale_amounts

LISTEN_BASELINES = ("persistence", "mean_7d", "previous_week")
MODEL_COLORS = {"lstm": "#1f77b4", "gru": "#ff7f0e"}
BASELINE_COLORS = {"persistence": "#7f7f7f", "mean_7d": "#2ca02c",
                   "previous_week": "#9467bd"}


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------

@dataclass
class _ListenAcc:
    abs_h: np.ndarray = field(default_factory=lambda: np.zeros(HORIZON))
    sq_h: np.ndarray = field(default_factory=lambda: np.zeros(HORIZON))
    abs_ch: np.ndarray = field(default_factory=lambda: np.zeros(len(LISTEN_AMOUNT_CHANNELS)))
    abs_active: float = 0.0
    abs_inactive: float = 0.0
    abs_users_logs: float = 0.0
    per_user: list = field(default_factory=list)

    def add(self, pred, real, active, has_logs):
        err = pred - real                       # (B, 7, 7)
        absd = np.abs(err)
        self.abs_h += absd.sum(axis=(0, 2))
        self.sq_h += (err ** 2).sum(axis=(0, 2))
        self.abs_ch += absd.sum(axis=(0, 1))
        day_abs = absd.sum(axis=2)              # (B, 7)
        self.abs_active += day_abs[active].sum()
        self.abs_inactive += day_abs[~active].sum()
        self.abs_users_logs += day_abs[has_logs].sum()
        self.per_user.append(absd.mean(axis=(1, 2)))


def _listen_baselines(raw):
    """Naive forecasts of days 83..89 from the raw input days (log1p)."""
    listen = [i for i in range(len(LISTEN_AMOUNT_CHANNELS))]
    last = raw[:, INPUT_DAYS - 1, listen]
    week = raw[:, INPUT_DAYS - HORIZON:INPUT_DAYS, listen]
    return {
        "persistence": np.repeat(last[:, None, :], HORIZON, axis=1),
        "mean_7d": np.repeat(week.mean(axis=1, keepdims=True), HORIZON, axis=1),
        "previous_week": week,
    }


def _expiry_rule_score(state):
    """Soft 'pays when the membership expires' score, ``(B, 7)``.

    Any customer with a known membership is expected to pay on its expiry day
    (``exp(-|days_to_expire - h|)``).
    """
    days = np.arange(1, HORIZON + 1)
    dist = np.abs(state["to_expire"][:, None] - days[None, :])
    eligible = state["expiry_known"][:, None]
    return np.where(eligible, np.exp(-dist.astype(np.float64)), 0.0)


def _last_paid(raw, index):
    """Last paid amount (NTD) and plan days inside the input window; NaN if none."""
    pay = raw[:, :INPUT_DAYS, index["tx_count"]] > 0
    has = pay.any(axis=1)
    last = INPUT_DAYS - 1 - np.argmax(pay[:, ::-1], axis=1)
    rows = np.arange(len(raw))
    paid = _inv_slog(raw[rows, last, index["tx_paid_sum"]])
    plan = np.expm1(raw[rows, last, index["tx_plan_days_sum"]])
    return np.where(has, paid, np.nan), np.where(has, plan, np.nan)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_trajectories(preds, data, rows, *, scaler_state, block_rows=50_000):
    """Evaluate every model in ``preds`` (``name -> predict_trajectories`` dict).

    ``rows`` are the test rows in the same order used for prediction and
    ``scaler_state`` is the scaler used for that prediction (``development``).
    Returns a dict of DataFrames plus per-user errors for the bootstrap.
    """
    index = channel_index(data.channel_specs)
    names = list(preds)
    rows = np.asarray(rows)
    n = len(rows)

    listen_acc = {k: _ListenAcc() for k in (*names, *LISTEN_BASELINES)}
    scores = {"active": {k: [] for k in (*names, "persistence", "mean_7d")},
              "pay": {k: [] for k in (*names, "expiry_rule")},
              "pay_week": {k: [] for k in (*names, "expiry_rule")}}
    y_active, y_pay, y_pay_week = [], [], []
    amount_err = {k: {"paid": [], "plan": []} for k in (*names, "last_paid")}
    n_active_days = n_inactive_days = n_logs_days = 0
    to_expire = []

    for start in range(0, n, block_rows):
        pos = slice(start, min(start + block_rows, n))
        raw = data.raw(rows[pos])
        real = raw[:, INPUT_DAYS:, :]
        real_listen = real[:, :, :len(LISTEN_AMOUNT_CHANNELS)]
        active = real[:, :, index["active"]] > 0
        pay = real[:, :, index["tx_count"]] > 0
        has_logs = np.repeat(
            (raw[:, :INPUT_DAYS, index["active"]] > 0).any(axis=1)[:, None], HORIZON, axis=1
        )
        n_active_days += active.sum()
        n_inactive_days += (~active).sum()
        n_logs_days += has_logs.sum()
        last_day = raw[:, INPUT_DAYS - 1, :]
        state0 = decode_last_state(last_day, index)
        to_expire.append(np.where(state0["expiry_known"],
                                  state0["to_expire"], np.iinfo(np.int32).min))

        # ---- listening amounts
        for name, forecast in _listen_baselines(raw).items():
            listen_acc[name].add(forecast, real_listen, active, has_logs)
        block_pred = {name: {k: v[pos] for k, v in preds[name].items()} for name in names}
        for name in names:
            listen_acc[name].add(expected_listening(block_pred[name], scaler_state),
                                 real_listen, active, has_logs)

        # ---- active / pay events
        y_active.append(active.ravel())
        y_pay.append(pay.ravel())
        y_pay_week.append(pay.any(axis=1))
        input_active = raw[:, :INPUT_DAYS, index["active"]] > 0
        scores["active"]["persistence"].append(
            np.repeat(input_active[:, -1:], HORIZON, axis=1).ravel().astype(float))
        scores["active"]["mean_7d"].append(
            np.repeat(input_active[:, -HORIZON:].mean(axis=1, keepdims=True),
                      HORIZON, axis=1).ravel())
        rule = _expiry_rule_score(state0)
        scores["pay"]["expiry_rule"].append(rule.ravel())
        scores["pay_week"]["expiry_rule"].append(rule.max(axis=1))
        for name in names:
            p = block_pred[name]
            scores["active"][name].append(p["p_active"].ravel())
            scores["pay"][name].append(p["p_pay"].ravel())
            scores["pay_week"][name].append(1 - np.prod(1 - p["p_pay"], axis=1))

        # ---- amounts on real payment days
        real_paid = _inv_slog(real[:, :, index["tx_paid_sum"]])[pay]
        real_plan = np.expm1(real[:, :, index["tx_plan_days_sum"]])[pay]
        last_paid, last_plan = _last_paid(raw, index)
        fallback = (np.nanmedian(last_paid) if np.isfinite(last_paid).any() else 0.0,
                    np.nanmedian(last_plan) if np.isfinite(last_plan).any() else 0.0)
        lp = np.repeat(np.nan_to_num(last_paid, nan=fallback[0])[:, None], HORIZON, axis=1)
        lpl = np.repeat(np.nan_to_num(last_plan, nan=fallback[1])[:, None], HORIZON, axis=1)
        amount_err["last_paid"]["paid"].append(np.abs(lp[pay] - real_paid))
        amount_err["last_paid"]["plan"].append(np.abs(lpl[pay] - real_plan))
        for name in names:
            amounts = unscale_amounts(block_pred[name]["amount"], scaler_state)
            amount_err[name]["paid"].append(np.abs(_inv_slog(amounts[..., 0])[pay] - real_paid))
            amount_err[name]["plan"].append(np.abs(np.expm1(amounts[..., 1])[pay] - real_plan))

    # ---------------------------------------------------------------- tables
    n_cells = n * HORIZON * len(LISTEN_AMOUNT_CHANNELS)
    persistence_mae = listen_acc["persistence"].abs_h.sum() / n_cells
    listening, horizon, channel, per_user = [], {}, {}, {}
    for key, acc in listen_acc.items():
        mae = acc.abs_h.sum() / n_cells
        listening.append({
            "predictor": key.upper() if key in names else key,
            "MAE": mae,
            "RMSE": np.sqrt(acc.sq_h.sum() / n_cells),
            "skill_vs_persistence": 1 - mae / persistence_mae,
            "MAE_active_days": acc.abs_active / max(n_active_days * 7, 1),
            "MAE_inactive_days": acc.abs_inactive / max(n_inactive_days * 7, 1),
            "MAE_users_with_logs": acc.abs_users_logs / max(n_logs_days * 7, 1),
        })
        horizon[key] = acc.abs_h / (n * len(LISTEN_AMOUNT_CHANNELS))
        channel[key] = acc.abs_ch / (n * HORIZON)
        per_user[key] = np.concatenate(acc.per_user)

    events = []
    targets = {"active": np.concatenate(y_active), "pay": np.concatenate(y_pay),
               "pay_week": np.concatenate(y_pay_week)}
    for target, by_pred in scores.items():
        y = targets[target]
        for key, parts in by_pred.items():
            s = np.concatenate(parts)
            row = {"target": target, "predictor": key.upper() if key in names else key,
                   "prevalence": y.mean()}
            if 0 < y.sum() < len(y):
                row["AUC"] = roc_auc_score(y, s)
                row["AP"] = average_precision_score(y, s)
            if key in names:
                row["Brier"] = np.mean((s - y) ** 2)
                row["mean_predicted"] = s.mean()
            events.append(row)

    amounts = []
    for key, parts in amount_err.items():
        paid, plan = np.concatenate(parts["paid"]), np.concatenate(parts["plan"])
        amounts.append({"predictor": key.upper() if key in names else key,
                        "n_payment_days": len(paid),
                        "MAE_paid_NTD": paid.mean() if len(paid) else np.nan,
                        "MAE_plan_days": plan.mean() if len(plan) else np.nan})

    def _label(key):
        return key.upper() if key in names else key

    return {
        "listening": pd.DataFrame(listening).set_index("predictor"),
        "horizon": pd.DataFrame({_label(k): v for k, v in horizon.items()},
                                index=[f"h{h}" for h in range(1, HORIZON + 1)]).T,
        "channel": pd.DataFrame({_label(k): v for k, v in channel.items()},
                                index=list(LISTEN_AMOUNT_CHANNELS)).T,
        "events": pd.DataFrame(events).set_index(["target", "predictor"]),
        "amounts": pd.DataFrame(amounts).set_index("predictor"),
        "per_user_listen_mae": per_user,
        "model_names": names,
        # Raw scores for the event figures: labels and scores are flattened
        # (user, day) for `active` / `pay` and per user for `pay_week`.
        "event_labels": targets,
        "event_scores": {t: {_label(k): np.concatenate(v) for k, v in by_pred.items()}
                         for t, by_pred in scores.items()},
        # Days to expiry on the last input day (day 82); INT32_MIN if unknown.
        "to_expire": np.concatenate(to_expire),
    }


# ---------------------------------------------------------------------------
# LSTM vs GRU
# ---------------------------------------------------------------------------

def paired_bootstrap_mean(err_a, err_b, *, n_boot=1000, seed=SEED, alpha=0.05):
    """95% CI of ``mean(err_a) - mean(err_b)`` resampling the same users for both."""
    err_a, err_b = np.asarray(err_a, float), np.asarray(err_b, float)
    diff = err_a - err_b
    n = len(diff)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        draws[b] = w @ diff / n
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_a": err_a.mean(), "mean_b": err_b.mean(), "diff": diff.mean(),
            "ci_low": low, "ci_high": high, "significant": bool(low > 0 or high < 0)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _color(key):
    return MODEL_COLORS.get(key.lower(), BASELINE_COLORS.get(key, "#333333"))


def plot_horizon(report, save_path=None):
    """Listening MAE per horizon day; a flat model curve means a near-constant forecast."""
    fig, ax = plt.subplots(figsize=(7, 4))
    table = report["horizon"]
    for key, values in table.iterrows():
        is_model = key.lower() in MODEL_COLORS
        ax.plot(range(1, HORIZON + 1), values.to_numpy(), marker="o",
                lw=2.2 if is_model else 1.2, ls="-" if is_model else "--",
                color=_color(key), label=key)
    ax.set_xlabel("horizon day h (day 83 + h - 1)")
    ax.set_ylabel("MAE, log1p listening")
    ax.set_title("Listening error by horizon (test)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def pick_example_users(data, rows, *, n_per_group=2, max_scan=200_000, seed=SEED):
    """Positions (into ``rows``) of active, fading, inactive and paying users."""
    rng = np.random.default_rng(seed)
    index = channel_index(data.channel_specs)
    scan = np.sort(rng.choice(len(rows), min(max_scan, len(rows)), replace=False))
    raw = data.raw(np.asarray(rows)[scan])
    active = raw[:, :, index["active"]] > 0
    pays = (raw[:, INPUT_DAYS:, index["tx_count"]] > 0).any(axis=1)
    early = active[:, INPUT_DAYS - 30:INPUT_DAYS - 7].mean(axis=1)
    late = active[:, INPUT_DAYS:].mean(axis=1)
    groups = {
        "active": scan[(early > 0.7) & (late > 0.7)],
        "fading": scan[(early > 0.6) & (late < 0.3)],
        "inactive": scan[active[:, :INPUT_DAYS].sum(axis=1) == 0],
        "paying": scan[pays],           # pays in the target week (for the payment panels)
    }
    return {g: pos[:n_per_group] for g, pos in groups.items()}


def plot_example_trajectories(preds, data, rows, examples, *, scaler_state,
                              channel="total_secs", context_days=28, save_path=None):
    """Real input + real future vs each model's expected future for a few users."""
    index = channel_index(data.channel_specs)
    k = LISTEN_AMOUNT_CHANNELS.index(channel)
    items = [(g, p) for g, positions in examples.items() for p in positions]
    if not items:
        return None
    fig, axes = plt.subplots(len(items), 1, figsize=(9, 2.2 * len(items)), sharex=True)
    axes = np.atleast_1d(axes)
    days = np.arange(INPUT_DAYS - context_days, INPUT_DAYS + HORIZON)
    for ax, (group, p) in zip(axes, items):
        raw = data.raw(np.asarray(rows)[[p]])[0]
        ax.plot(days, raw[days, index[channel]], color="black", lw=1.5, label="real")
        for name, pred in preds.items():
            one = {key: value[[p]] for key, value in pred.items()}
            ax.plot(np.arange(INPUT_DAYS, INPUT_DAYS + HORIZON),
                    expected_listening(one, scaler_state)[0, :, k],
                    color=_color(name), lw=2, marker="o", ms=3, label=name.upper())
        ax.axvline(INPUT_DAYS - 0.5, color="grey", ls=":")
        ax.set_ylabel(f"log1p\n{channel}")
        ax.set_title(f"{group} user (test position {p})", fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[-1].set_xlabel("day of the window (dotted line: start of the forecast)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Event figures (active / pay): what the model actually predicts well
# ---------------------------------------------------------------------------

EVENT_TITLES = {"active": "listens on day h (active)",
                "pay": "pays on day h",
                "pay_week": "pays at least once in the 7 days"}


def _save(fig, save_path):
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_event_curves(report, save_path=None):
    """ROC and precision-recall curves of ``active``, ``pay`` and weekly ``pay`` vs the baselines."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    targets = list(EVENT_TITLES)
    fig, axes = plt.subplots(2, len(targets), figsize=(4.2 * len(targets), 7.6))
    for j, target in enumerate(targets):
        y = report["event_labels"][target]
        for key, s in report["event_scores"][target].items():
            is_model = key.lower() in MODEL_COLORS
            style = dict(color=_color(key), lw=2 if is_model else 1.2, ls="-" if is_model else "--")
            fpr, tpr, _ = roc_curve(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            axes[0, j].plot(fpr, tpr, label=f"{key} ({roc_auc_score(y, s):.3f})", **style)
            axes[1, j].plot(rec, prec, label=f"{key} ({average_precision_score(y, s):.3f})", **style)
        axes[0, j].plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8)
        axes[1, j].axhline(y.mean(), color="#bbbbbb", lw=0.8)
        axes[0, j].set_title(f"{EVENT_TITLES[target]}\nROC (AUC)", fontsize=10, loc="left")
        axes[1, j].set_title(f"precision-recall (AP), prevalence {y.mean():.3f}", fontsize=10, loc="left")
        axes[0, j].set_xlabel("false positive rate"); axes[0, j].set_ylabel("true positive rate")
        axes[1, j].set_xlabel("recall"); axes[1, j].set_ylabel("precision")
        for ax in axes[:, j]:
            ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8, loc="lower right")
        axes[1, j].legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, save_path)


def plot_event_calibration(report, n_bins=10, save_path=None):
    """Reliability of the predicted probabilities (quantile bins), one panel per event."""
    targets = list(EVENT_TITLES)
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.9))
    for ax, target in zip(axes, targets):
        y = report["event_labels"][target].astype(float)
        hi = 0.0
        for key, s in report["event_scores"][target].items():
            if key.lower() not in MODEL_COLORS:
                continue
            edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
            bins = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
            count = np.bincount(bins, minlength=len(edges) - 1)
            keep = count > 0
            mean_p = np.bincount(bins, s, len(edges) - 1)[keep] / count[keep]
            mean_y = np.bincount(bins, y, len(edges) - 1)[keep] / count[keep]
            ax.plot(mean_p, mean_y, marker="o", color=_color(key), lw=2, label=key)
            hi = max(hi, mean_p.max(), mean_y.max())
        hi = min(1.0, hi * 1.05)
        ax.plot([0, hi], [0, hi], color="#bbbbbb", lw=0.8, ls="--", label="perfect")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(EVENT_TITLES[target], fontsize=10, loc="left")
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
    return _save(fig, save_path)


def plot_pay_by_expiry(report, window=(-10, 10), save_path=None):
    """Daily payment rate aligned to each customer's expiry: real vs predicted ``p_pay``.

    The x axis is the forecast day relative to the expiry known on day 82
    (0 = expiry day). Customers with an unknown expiry are left out.
    """
    dte = report["to_expire"]
    known = dte != np.iinfo(np.int32).min
    rel = (np.arange(1, HORIZON + 1)[None, :] - dte[:, None])           # (N, 7)
    mask = np.repeat(known[:, None], HORIZON, axis=1) & (rel >= window[0]) & (rel <= window[1])
    rel, y = rel[mask], report["event_labels"]["pay"].reshape(-1, HORIZON)[mask].astype(float)
    xs = np.arange(window[0], window[1] + 1)
    idx = rel - window[0]
    n = np.bincount(idx, minlength=len(xs))
    ok = n > 0

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(xs[ok], (np.bincount(idx, y, len(xs)) / np.maximum(n, 1))[ok] * 100,
            color="black", lw=2, marker="o", ms=3, label="real")
    for key, s in report["event_scores"]["pay"].items():
        s = s.reshape(-1, HORIZON)[mask]
        is_model = key.lower() in MODEL_COLORS
        if not is_model:
            continue
        ax.plot(xs[ok], (np.bincount(idx, s, len(xs)) / np.maximum(n, 1))[ok] * 100,
                color=_color(key), lw=2, ls="--", label=f"{key} (mean p_pay)")
    ax.axvline(0, color="grey", ls=":")
    ax.set_xticks(xs[::2])
    ax.set_xlabel("forecast day relative to the expiry known on day 82 (0 = expiry day)")
    ax.set_ylabel("% paying that day")
    ax.set_title("Payments aligned to the expiry (test)", loc="left")
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9)
    return _save(fig, save_path)


def plot_example_events(preds, data, rows, examples, *, context_days=28, save_path=None):
    """Same users as :func:`plot_example_trajectories`, but the events.

    Left: real active days (squares) vs each model's ``p_active``. Right: real
    payment days (bars) vs ``p_pay``. Both are probabilities, so a good forecast
    is high where the square / bar is and low elsewhere.
    """
    index = channel_index(data.channel_specs)
    items = [(g, p) for g, positions in examples.items() for p in positions]
    if not items:
        return None
    fig, axes = plt.subplots(len(items), 2, figsize=(12, 1.9 * len(items)), sharex=True, sharey=True,
                             squeeze=False)
    days = np.arange(INPUT_DAYS - context_days, INPUT_DAYS + HORIZON)
    future = np.arange(INPUT_DAYS, INPUT_DAYS + HORIZON)
    for (ax_a, ax_p), (group, p) in zip(axes, items):
        raw = data.raw(np.asarray(rows)[[p]])[0]
        act = raw[days, index["active"]] > 0
        pay = raw[days, index["tx_count"]] > 0
        ax_a.scatter(days[act], np.ones(act.sum()), marker="s", s=14, color="black", label="real active day")
        ax_p.bar(days[pay], np.ones(pay.sum()), width=0.6, color="#d62728", alpha=0.5, label="real payment day")
        for name, pred in preds.items():
            ax_a.plot(future, pred["p_active"][p], color=_color(name), lw=2, marker="o", ms=3,
                      label=f"{name.upper()} p(active)")
            ax_p.plot(future, pred["p_pay"][p], color=_color(name), lw=2, marker="o", ms=3,
                      label=f"{name.upper()} p(pay)")
        for ax in (ax_a, ax_p):
            ax.axvline(INPUT_DAYS - 0.5, color="grey", ls=":")
            ax.grid(alpha=0.3)
        ax_a.set_ylim(-0.05, 1.1); ax_a.set_ylabel("probability")
        ax_a.set_title(f"{group} user (test position {p}): listening", fontsize=9, loc="left")
        ax_p.set_title(f"{group} user (test position {p}): payment", fontsize=9, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=3, loc="center left")
    for ax in axes[-1]:
        ax.set_xlabel("day of the window (dotted line: start of the forecast)")
    return _save(fig, save_path)
