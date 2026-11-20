"""Shared helpers for the EDA series in ``workshops/eda``.

Every notebook works on the same population: the February cohort (``train.parquet``)
restricted to the ``fit`` part of the development split, with raw records filtered
to dates on or before the feature cutoff (2017-01-31). ``stop``, ``tune`` and the
March cohort are never looked at with their labels here.

The split is reproduced with the same salted SHA-1 hash and per-class ranking as
``w1/src/models/feature_preparation/split.py`` so the EDA population is exactly the
rows the models are fitted on.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

EDA_DIR = Path(__file__).resolve().parent
WORKSHOPS_DIR = EDA_DIR.parent
PARQUET_DIR = WORKSHOPS_DIR / "data" / "parquet"
FIG_DIR = EDA_DIR / "figures"
CACHE_DIR = EDA_DIR / "cache"
DECISIONS_PATH = EDA_DIR / "DECISIONS.md"

CUTOFF = date(2017, 1, 31)              # last day of history for the February cohort
LABEL_MONTH = "2017-02"                 # month whose renewals define is_churn
WINDOW_DAYS = 90                        # listening window used by w1 and w2
WINDOW_START = CUTOFF - timedelta(days=WINDOW_DAYS - 1)   # 2016-11-03
CUTOFF_INT = int(CUTOFF.strftime("%Y%m%d"))
WINDOW_START_INT = int(WINDOW_START.strftime("%Y%m%d"))

# Same values as w1/src/models/constants.py (the split must match the models).
SPLIT_HASH_SALT = "salt110:"
SPLIT_SHARES = {"fit": 0.75, "stop": 0.125, "tune": 0.125}
SEED = 110

LOG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]
TXN_COLS = ["payment_method_id", "payment_plan_days", "plan_list_price", "actual_amount_paid",
            "is_auto_renew", "transaction_date", "membership_expire_date", "is_cancel"]

def set_cutoff(cutoff: date) -> None:
    """Switch every helper to another feature cutoff (e.g. 2017-02-28 for the March cohort)."""
    global CUTOFF, WINDOW_START, CUTOFF_INT, WINDOW_START_INT
    CUTOFF = cutoff
    WINDOW_START = cutoff - timedelta(days=WINDOW_DAYS - 1)
    CUTOFF_INT = int(CUTOFF.strftime("%Y%m%d"))
    WINDOW_START_INT = int(WINDOW_START.strftime("%Y%m%d"))


for _d in (FIG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _cut(users: pl.DataFrame) -> pl.Expr:
    """The feature cutoff: per customer when ``users`` has an ``as_of`` date column
    (notebook 10, prediction k days before each customer's expiry), else the global ``CUTOFF``."""
    return pl.col("as_of") if "as_of" in users.columns else pl.lit(CUTOFF)


def _keys(users: pl.DataFrame) -> list[str]:
    return ["msno", "as_of"] if "as_of" in users.columns else ["msno"]


def filter_as_of(t: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    """Keep the transactions dated on or before each customer's ``as_of``."""
    return (t.join(users.select("msno", "as_of"), on="msno", how="inner")
            .filter(pl.col("td") <= pl.col("as_of")).drop("as_of"))


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def _hash_unit(msno: str) -> float:
    h = hashlib.sha1((SPLIT_HASH_SALT + str(msno)).encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def _assign_split(hash_unit: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-class hash rank cut at the cumulative shares (same rule as the pipeline)."""
    names = list(SPLIT_SHARES)
    cum = np.cumsum([SPLIT_SHARES[n] for n in names])
    out = np.empty(len(y), dtype=object)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        ranks = np.argsort(np.argsort(hash_unit[idx])) / max(len(idx), 1)
        part = np.clip(np.searchsorted(cum, ranks, side="right"), 0, len(names) - 1)
        out[idx] = np.asarray(names, dtype=object)[part]
    return out


def load_cohort(refresh: bool = False) -> pl.DataFrame:
    """February cohort with its split label: ``row, msno, is_churn, split``."""
    path = CACHE_DIR / "cohort_feb.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    cohort = pl.read_parquet(PARQUET_DIR / "train.parquet", columns=["msno", "is_churn"]).with_row_index("row")
    hu = np.fromiter((_hash_unit(m) for m in cohort["msno"]), dtype=np.float64, count=cohort.height)
    split = _assign_split(hu, cohort["is_churn"].to_numpy())
    cohort = cohort.with_columns(pl.Series("split", split.astype(str)))
    cohort.write_parquet(path)
    return cohort


def load_fit_population() -> pl.DataFrame:
    """The EDA population: ``msno, is_churn`` for the ``fit`` part only."""
    return load_cohort().filter(pl.col("split") == "fit").select("msno", "is_churn")


# ---------------------------------------------------------------------------
# Raw tables, restricted to the population and to the cutoff
# ---------------------------------------------------------------------------

def scan_members(users: pl.DataFrame) -> pl.LazyFrame:
    """Raw ``members_v3`` rows of ``users`` (no cleaning)."""
    return pl.scan_parquet(PARQUET_DIR / "members_v3.parquet").join(
        users.lazy().select("msno"), on="msno", how="semi")


def scan_transactions(users: pl.DataFrame, as_of: bool = True) -> pl.LazyFrame:
    """Raw transactions of ``users`` from both releases (no deduplication).

    ``source`` is 1 or 2 (which file) and ``_ord`` the stable source order, used as a
    tie break. With ``as_of`` only records with ``transaction_date <= CUTOFF`` remain.
    """
    frames = [
        pl.scan_parquet(PARQUET_DIR / f).with_columns(pl.lit(i, dtype=pl.Int8).alias("source"))
        for i, f in ((1, "transactions.parquet"), (2, "transactions_v2.parquet"))
    ]
    lf = (pl.concat(frames).with_row_index("_ord")
          .join(users.lazy().select("msno"), on="msno", how="semi"))
    if as_of:
        lf = lf.filter(pl.col("transaction_date") <= CUTOFF_INT)
    return lf


def scan_user_logs(users: pl.DataFrame, start: int | None = None,
                   end: int | None = None) -> pl.LazyFrame:
    """Raw daily log rows of ``users`` between ``start`` and ``end`` (default: the current window)."""
    start = WINDOW_START_INT if start is None else start
    end = CUTOFF_INT if end is None else end
    frames = [pl.scan_parquet(PARQUET_DIR / f) for f in ("user_logs.parquet", "user_logs_v2.parquet")]
    return (pl.concat(frames)
            .filter(pl.col("date").is_between(start, end))
            .join(users.lazy().select("msno"), on="msno", how="semi"))


def clean_transactions(users: pl.DataFrame) -> pl.DataFrame:
    """Transactions of ``users`` as of the cutoff, cleaned with the rules of notebook 02.

    - exact duplicates (msno + 8 fields) dropped, first copy kept;
    - ``payment_plan_days`` and ``plan_list_price`` set to null when both are 0
      (missing fields, Mar-Oct 2015), flagged by ``plan_fields_missing``;
    - ``ed`` (expiry date) null when the year is before 2000 or when a non-cancel
      record expires before its transaction date, flagged by ``expiry_invalid``;
    - ``td`` transaction date; ``price_type`` in full / discounted / free / unknown.
    """
    raw = scan_transactions(users).collect(engine="streaming")
    t = raw.unique(subset=["msno"] + TXN_COLS, keep="first", maintain_order=True)
    missing = (pl.col("payment_plan_days") == 0) & (pl.col("plan_list_price") == 0)
    t = t.with_columns(
        plan_fields_missing=missing,
        payment_plan_days=pl.when(missing).then(None).otherwise(pl.col("payment_plan_days")),
        plan_list_price=pl.when(missing).then(None).otherwise(pl.col("plan_list_price")),
        td=to_date("transaction_date"),
        ed=to_date("membership_expire_date"),
    )
    invalid = (pl.col("ed").dt.year() < 2000) | ((pl.col("is_cancel") == 0) & (pl.col("ed") < pl.col("td")))
    t = t.with_columns(expiry_invalid=invalid.fill_null(True))
    return t.with_columns(
        ed=pl.when(pl.col("expiry_invalid")).then(None).otherwise(pl.col("ed")),
        price_type=pl.when(pl.col("plan_fields_missing")).then(pl.lit("unknown"))
        .when(pl.col("actual_amount_paid") == 0).then(pl.lit("free"))
        .when(pl.col("actual_amount_paid") < pl.col("plan_list_price")).then(pl.lit("discounted"))
        .otherwise(pl.lit("full")),
    )


STATE_ORDER = ["auto-renew", "manual", "cancel + payment", "cancel", "no txn"]
EXPIRY_ORDER = ["lapsed", "february", "march", "april+", "unknown", "no txn"]


def subscription_state(t: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    """Per-customer subscription state at the cutoff (rules of notebooks 02-03).

    Records are collapsed per day first (file order is not time order):

    - ``state``: last transaction day = payment (``auto-renew`` / ``manual`` from the
      day's latest-expiring payment), ``cancel``, ``cancel + payment``, or ``no txn``;
    - ``last_expire``: latest valid expiry of the last day; ``days_to_expire`` from the cutoff;
    - ``expiry_state``: lapsed / february / march / april+ / unknown / no txn;
    - ``days_since_last_txn``;
    - ``last_plan_days``, ``last_list_price``, ``last_paid``, ``last_price_type``,
      ``last_payment_method_id``: from the latest **payment** record (latest day,
      then latest expiry, then source order).
    """
    last_day = (t.filter(pl.col("td") == pl.col("td").max().over("msno"))
                .group_by("msno").agg(
                    last_txn_date=pl.col("td").first(),
                    day_has_cancel=(pl.col("is_cancel") == 1).any(),
                    day_has_payment=(pl.col("is_cancel") == 0).any(),
                    last_expire=pl.col("ed").max()))
    pay = (t.filter(pl.col("is_cancel") == 0).sort("td", "ed", "_ord", nulls_last=False)
           .group_by("msno").agg(
               last_plan_days=pl.col("payment_plan_days").last(),
               last_list_price=pl.col("plan_list_price").last(),
               last_paid=pl.col("actual_amount_paid").last(),
               last_price_type=pl.col("price_type").last(),
               last_payment_method_id=pl.col("payment_method_id").last(),
               last_pay_auto_renew=pl.col("is_auto_renew").last(),
               last_pay_date=pl.col("td").last()))
    s = users.join(last_day, on="msno", how="left").join(pay, on="msno", how="left")
    cut = _cut(users)
    dte = (pl.col("last_expire") - cut).dt.total_days()
    return s.with_columns(
        state=pl.when(pl.col("last_txn_date").is_null()).then(pl.lit("no txn"))
        .when(pl.col("day_has_cancel") & pl.col("day_has_payment")).then(pl.lit("cancel + payment"))
        .when(pl.col("day_has_cancel")).then(pl.lit("cancel"))
        .when(pl.col("last_pay_auto_renew") == 1).then(pl.lit("auto-renew"))
        .otherwise(pl.lit("manual")),
        days_to_expire=dte,
        expiry_state=pl.when(pl.col("last_txn_date").is_null()).then(pl.lit("no txn"))
        .when(dte.is_null()).then(pl.lit("unknown"))
        .when(dte < 0).then(pl.lit("lapsed"))
        .when(dte <= 28).then(pl.lit("february"))
        .when(dte <= 59).then(pl.lit("march"))
        .otherwise(pl.lit("april+")),
        days_since_last_txn=(cut - pl.col("last_txn_date")).dt.total_days(),
    )


STATE_BLOCK = ["state", "expiry_state", "plan_band", "price_level", "price_type", "pm_group", "dsl_band"]


def state_block(st: pl.DataFrame, min_method_customers: int = 1000, keep_methods=None) -> pl.DataFrame:
    """Categorical state features proposed in notebook 03 (adds the ``STATE_BLOCK`` columns).

    ``keep_methods`` fixes the payment methods kept as their own level (learned on fit);
    by default they are the methods with at least ``min_method_customers`` customers in ``st``.
    """
    if keep_methods is None:
        counts = st.group_by("last_payment_method_id").len()
        keep_methods = counts.filter((pl.col("len") >= min_method_customers)
                                     & pl.col("last_payment_method_id").is_not_null())["last_payment_method_id"].to_list()
    keep = list(keep_methods)
    plan, dsl = pl.col("last_plan_days"), pl.col("days_since_last_txn")
    return st.with_columns(
        plan_band=pl.when(plan.is_null()).then(pl.lit("unknown")).when(plan <= 7).then(pl.lit("7 or less"))
        .when(plan <= 31).then(pl.lit("30-31")).when(plan <= 200).then(pl.lit("32-200")).otherwise(pl.lit("201+")),
        price_level=pl.when(pl.col("last_list_price").is_in([99, 100, 129, 149, 150, 180]))
        .then(pl.col("last_list_price").cast(pl.String)).otherwise(pl.lit("other")),
        price_type=pl.col("last_price_type").fill_null("none"),
        pm_group=pl.when(pl.col("last_payment_method_id").is_null()).then(pl.lit("none"))
        .when(pl.col("last_payment_method_id").is_in(keep)).then(pl.col("last_payment_method_id").cast(pl.String))
        .otherwise(pl.lit("other")),
        dsl_band=pl.when(dsl.is_null()).then(pl.lit("none")).when(dsl <= 7).then(pl.lit("0-7"))
        .when(dsl <= 31).then(pl.lit("8-31")).when(dsl <= 90).then(pl.lit("32-90")).otherwise(pl.lit("91+")),
    )


HISTORY_BLOCK = ["payer_days", "n_payments", "n_lapses", "total_lapse_days"]


def history_block(t: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    """The four history features kept in notebook 04 (whole history, cleaned records)."""
    pay = t.filter(pl.col("is_cancel") == 0).group_by("msno").agg(
        n_payments=pl.len(), first_td=pl.col("td").min())
    days = (t.group_by("msno", "td").agg(day_expire=pl.col("ed").max(), has_payment=(pl.col("is_cancel") == 0).any())
             .sort("msno", "td").with_columns(prev_expire=pl.col("day_expire").shift(1).over("msno"))
             .with_columns(gap=(pl.col("td") - pl.col("prev_expire")).dt.total_days()))
    lapses = days.filter(pl.col("has_payment") & (pl.col("gap") > 0)).group_by("msno").agg(
        n_lapses=pl.len(), total_lapse_days=pl.col("gap").sum())
    return (users.select(_keys(users)).join(pay, on="msno", how="left").join(lapses, on="msno", how="left")
            .with_columns(pl.col("n_payments", "n_lapses", "total_lapse_days").fill_null(0),
                          payer_days=(_cut(users) - pl.col("first_td")).dt.total_days())
            .drop("first_td", "as_of", strict=False))


MEMBER_CAT = ["via", "profile_state", "age_band"]
MEMBER_NUM = ["bd", "tenure_days"]


def member_block(users: pl.DataFrame) -> pl.DataFrame:
    """Member features proposed in notebook 01: channel, profile state, age band (categorical);
    age with NaN for unknown and tenure with NaN for missing / post-cutoff registration (numeric)."""
    m = users.select(_keys(users)).join(scan_members(users).collect(), on="msno", how="left")
    valid = pl.col("bd").is_between(1, 100).fill_null(False)
    bd = pl.col("bd")
    tenure = (_cut(users) - to_date("registration_init_time")).dt.total_days()
    return m.select(
        "msno",
        via=pl.col("registered_via").cast(pl.String).fill_null("no member"),
        profile_state=pl.when(pl.col("city").is_null()).then(pl.lit("no member"))
        .when(~valid & pl.col("gender").is_null()).then(pl.lit("incomplete")).otherwise(pl.lit("complete")),
        age_band=pl.when(pl.col("city").is_null()).then(pl.lit("no member"))
        .when(~valid).then(pl.lit("unknown")).when(bd <= 18).then(pl.lit("<=18")).when(bd <= 21).then(pl.lit("19-21"))
        .when(bd <= 24).then(pl.lit("22-24")).when(bd <= 29).then(pl.lit("25-29")).when(bd <= 44).then(pl.lit("30-44"))
        .otherwise(pl.lit("45+")),
        bd=pl.when(valid).then(bd).otherwise(None).cast(pl.Float64),
        tenure_days=pl.when(tenure >= 0).then(tenure).otherwise(None).cast(pl.Float64),
    )


def clean_user_days(users: pl.DataFrame) -> pl.LazyFrame:
    """Daily logs of ``users`` in the 90-day window with the notebook-05 rule applied:
    days with ``total_secs > 86,400`` have all channels rescaled to 24 h. Adds ``day``
    (0 = window start ... 89 = cutoff) and ``plays`` (sum of the five play counters)."""
    scale = pl.when(pl.col("total_secs") > 86_400).then(86_400 / pl.col("total_secs")).otherwise(1.0)
    return (scan_user_logs(users)
            .with_columns([(pl.col(c).cast(pl.Float64) * scale).alias(c) for c in LOG_COLS])
            .with_columns(day=(to_date("date") - pl.lit(WINDOW_START)).dt.total_days().cast(pl.Int16),
                          plays=pl.sum_horizontal("num_25", "num_50", "num_75", "num_985", "num_100")))


def payment_cycles(t: pl.DataFrame) -> pl.DataFrame:
    """One row per billing cycle (notebooks 02 and 09; basis of the w2 cycle tensor).

    A cycle starts on a payment day (day with a non-cancel record, records collapsed per
    day) and ends the day before the next payment day, or at the cutoff for the current
    cycle. Columns: ``start``, ``end``, ``cycle_days``, ``expire`` (latest valid expiry of
    the start day), ``auto_renew``, ``plan_days``, ``paid``, ``price_type`` (latest-expiring
    payment of the day), ``lapse_before`` (days without subscription before the cycle; null
    for the first), ``cancel_in_cycle``, ``k_from_last`` (0 = current cycle).
    """
    pay = (t.filter(pl.col("is_cancel") == 0).sort("msno", "td", "ed", "_ord", nulls_last=False)
           .group_by("msno", "td").agg(
               expire=pl.col("ed").max(), auto_renew=pl.col("is_auto_renew").last(),
               plan_days=pl.col("payment_plan_days").last(), paid=pl.col("actual_amount_paid").last(),
               price_type=pl.col("price_type").last())
           .sort("msno", "td")
           .with_columns(next_start=pl.col("td").shift(-1).over("msno"),
                         prev_expire=pl.col("expire").shift(1).over("msno"))
           .rename({"td": "start"}))
    cyc = pay.with_columns(
        end=pl.coalesce(pl.col("next_start") - pl.duration(days=1), pl.lit(CUTOFF)),
        lapse_before=(pl.col("start") - pl.col("prev_expire")).dt.total_days().clip(lower_bound=0),
    ).with_columns(
        cycle_days=(pl.col("end") - pl.col("start")).dt.total_days() + 1,
        k_from_last=(pl.len().over("msno") - 1 - pl.int_range(pl.len()).over("msno")).cast(pl.Int32),
    ).drop("next_start", "prev_expire")
    cancels = (t.filter(pl.col("is_cancel") == 1).select("msno", pl.col("td").alias("cdate")).unique()
               .sort("cdate"))
    hit = (cancels.join_asof(cyc.select("msno", "start").sort("start"), left_on="cdate", right_on="start",
                             by="msno", strategy="backward")
           .drop_nulls("start").group_by("msno", "start").agg(cancel_in_cycle=pl.len()))
    return (cyc.join(hit, on=["msno", "start"], how="left")
            .with_columns(cancel_in_cycle=(pl.col("cancel_in_cycle").fill_null(0) > 0).cast(pl.Int8)))


LISTEN_BLOCK = ["has_logs", "days_since_last_log", "longest_gap", "active_days_30", "secs_30"]


def daily_logs(users: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Cleaned daily logs of ``users`` between ``start`` and ``end`` (24 h rescaling of
    notebook 05), reduced to ``msno, date, total_secs``. Input of the per-customer
    ``listening_block(users, days=...)``."""
    scale = pl.when(pl.col("total_secs") > 86_400).then(86_400 / pl.col("total_secs")).otherwise(1.0)
    lf = scan_user_logs(users, int(start.strftime("%Y%m%d")), int(end.strftime("%Y%m%d")))
    return lf.select("msno", date=to_date("date"),
                     total_secs=(pl.col("total_secs").cast(pl.Float64) * scale).cast(pl.Float32)
                     ).collect(engine="streaming")


def _window_days(days: pl.DataFrame, users: pl.DataFrame) -> pl.LazyFrame:
    """Rows of ``days`` inside each customer's 90-day window ending at ``as_of``;
    ``day`` = 0 (window start) ... 89 (``as_of``), as in ``clean_user_days``."""
    back = (pl.col("as_of") - pl.col("date")).dt.total_days()
    return (days.lazy().join(users.lazy().select("msno", "as_of"), on="msno", how="inner")
            .filter(back.is_between(0, WINDOW_DAYS - 1))
            .with_columns(day=(WINDOW_DAYS - 1 - back).cast(pl.Int16)).drop("as_of"))


def listening_block(users: pl.DataFrame, days: pl.DataFrame | None = None) -> pl.DataFrame:
    """The five listening features kept in notebook 06 (cleaned daily logs, current window).

    With ``days`` (from ``daily_logs``) and an ``as_of`` column in ``users``, each customer's
    window ends at their own ``as_of`` instead of the global cutoff."""
    ud = clean_user_days(users) if days is None else _window_days(days, users)
    last30 = ud.filter(pl.col("day") >= WINDOW_DAYS - 30).group_by("msno").agg(
        active_days_30=pl.len(), secs_30=pl.col("total_secs").sum())
    rec = (ud.group_by("msno").agg(days=pl.col("day").sort())
             .with_columns(last_day=pl.col("days").list.max(),
                           inner_gap=(pl.col("days").list.diff().list.max() - 1).fill_null(0))
             .select("msno",
                     days_since_last_log=(WINDOW_DAYS - 1 - pl.col("last_day")).cast(pl.Int32),
                     longest_gap=pl.max_horizontal(pl.col("inner_gap"), WINDOW_DAYS - 1 - pl.col("last_day"),
                                                   pl.col("days").list.min()).cast(pl.Int32)))
    out = (users.select("msno").lazy().join(last30, on="msno", how="left").join(rec, on="msno", how="left")
           .collect(engine="streaming"))
    return out.with_columns(
        has_logs=pl.col("days_since_last_log").is_not_null().cast(pl.Int8),
        active_days_30=pl.col("active_days_30").fill_null(0),
        secs_30=pl.col("secs_30").fill_null(0.0),
        days_since_last_log=pl.col("days_since_last_log").fill_null(WINDOW_DAYS),
        longest_gap=pl.col("longest_gap").fill_null(WINDOW_DAYS))


def to_date(col: str) -> pl.Expr:
    """``YYYYMMDD`` integer column -> ``pl.Date`` (invalid values become null)."""
    return pl.col(col).cast(pl.String).str.to_date("%Y%m%d", strict=False)


def cached(name: str, build, refresh: bool = False) -> pl.DataFrame:
    """Read ``cache/<name>.parquet`` or build it with ``build()`` and store it."""
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = build()
    df.write_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Churn-rate tables
# ---------------------------------------------------------------------------

def wilson(k, n, z: float = 1.96):
    """Wilson 95 % interval for ``k`` successes out of ``n``."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def churn_rate_by(df: pl.DataFrame, by, target: str = "is_churn", order=None) -> pd.DataFrame:
    """Customers, churners, churn rate, Wilson CI, share and lift per level of ``by``."""
    by = [by] if isinstance(by, str) else list(by)
    base = df[target].mean()
    t = (df.group_by(by).agg(pl.len().alias("n"), pl.col(target).sum().alias("churners"))
         .to_pandas())
    t["churn_rate"] = t["churners"] / t["n"]
    t["ci_low"], t["ci_high"] = wilson(t["churners"], t["n"])
    t["share"] = t["n"] / t["n"].sum()
    t["lift"] = t["churn_rate"] / base
    if order is not None:
        t[by[0]] = pd.Categorical(t[by[0]], categories=order, ordered=True)
    t = t.sort_values(by).reset_index(drop=True)
    if order is not None:
        t[by[0]] = t[by[0]].astype(str)
    return t


def quantile_bins(df: pl.DataFrame, col: str, q: int = 10, label: str | None = None) -> pl.DataFrame:
    """Add ``<col>_bin`` with quantile bins (ties merged); nulls get their own bin."""
    label = label or f"{col}_bin"
    s = df[col].cast(pl.Float64)
    edges = np.unique(np.nanquantile(s.drop_nulls().to_numpy(), np.linspace(0, 1, q + 1)))
    labels = [f"[{edges[i]:g}, {edges[i + 1]:g}{']' if i == len(edges) - 2 else ')'}"
              for i in range(len(edges) - 1)]
    idx = np.clip(np.searchsorted(edges, s.fill_null(np.nan).to_numpy(), side="right") - 1,
                  0, len(labels) - 1)
    out = np.where(s.is_null().to_numpy(), "missing", np.asarray(labels, dtype=object)[idx])
    return df.with_columns(pl.Series(label, out.astype(str)))


# ---------------------------------------------------------------------------
# Plotting (validated reference palette, light mode)
# ---------------------------------------------------------------------------

COLORS = {
    "series": "#2a78d6",      # churn rate bars / main series
    "churn": "#eb6834",       # churners when compared with stayers
    "stay": "#2a78d6",        # stayers
    "base": "#52514e",        # base-rate reference line
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["surface"], "axes.facecolor": COLORS["surface"],
    "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["muted"],
    "axes.titlecolor": COLORS["text"], "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.8,
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "font.size": 9, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def plot_churn_rate(table: pd.DataFrame, col: str, ax=None, title: str | None = None,
                    base_rate: float | None = None, rotate: int = 0, show_n: bool = True):
    """Bar per level: churn rate with Wilson CI, ``n`` under each label, base-rate line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 0.55 * len(table) + 1.5), 3.4))
    x = np.arange(len(table))
    rate = table["churn_rate"].to_numpy() * 100
    err = np.vstack([rate - table["ci_low"] * 100, table["ci_high"] * 100 - rate])
    ax.bar(x, rate, width=0.7, color=COLORS["series"], edgecolor=COLORS["surface"], linewidth=2)
    ax.errorbar(x, rate, yerr=err, fmt="none", ecolor=COLORS["text"], elinewidth=1, capsize=2)
    if base_rate is not None:
        ax.axhline(base_rate * 100, color=COLORS["base"], linestyle="--", linewidth=1)
        ax.annotate(f"base {base_rate:.2%}", xy=(1, base_rate * 100), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                    color=COLORS["muted"], fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["surface"], ec="none"))
    labels = table[col].astype(str).tolist()
    if show_n:
        labels = [f"{l}\nn={n:,}" for l, n in zip(labels, table["n"])]
    ax.set_xticks(x, labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel("churn rate (%)")
    ax.set_ylim(0, min(100, max(table["ci_high"].max() * 100, (base_rate or 0) * 100) * 1.1))
    if title:
        ax.set_title(title, loc="left")
    return ax


def save_fig(fig, name: str):
    """Save under ``eda/figures/<name>.png``."""
    fig.savefig(FIG_DIR / f"{name}.png")


# ---------------------------------------------------------------------------
# Decisions log
# ---------------------------------------------------------------------------

DECISION_KINDS = {"new feature", "transformation", "cleaning", "drop", "no action", "open"}


def write_decisions(notebook: str, title: str, rows: list[dict]) -> pd.DataFrame:
    """Replace this notebook's section in ``DECISIONS.md`` and return it as a table.

    Each row needs ``finding``, ``decision`` and ``kind`` (one of ``DECISION_KINDS``).
    """
    for r in rows:
        if r["kind"] not in DECISION_KINDS:
            raise ValueError(f"unknown kind {r['kind']!r}")
    table = pd.DataFrame(rows, columns=["finding", "decision", "kind"])
    body = "\n".join(f"| {r['finding']} | {r['decision']} | {r['kind']} |" for r in rows)
    section = (f"<!-- nb:{notebook} -->\n## {notebook} · {title}\n\n"
               f"| Finding | Decision | Kind |\n|---|---|---|\n{body}\n<!-- /nb:{notebook} -->")
    text = DECISIONS_PATH.read_text() if DECISIONS_PATH.exists() else (
        "# EDA decisions\n\nOne section per notebook, written by `write_decisions` in its last cell. "
        "This table is the contract for `w1/src/data_processing` and `w2/src/data_processing`.\n")
    pattern = re.compile(rf"<!-- nb:{re.escape(notebook)} -->.*?<!-- /nb:{re.escape(notebook)} -->", re.S)
    text = pattern.sub(section, text) if pattern.search(text) else text.rstrip() + "\n\n" + section + "\n"
    DECISIONS_PATH.write_text(text)
    return table
