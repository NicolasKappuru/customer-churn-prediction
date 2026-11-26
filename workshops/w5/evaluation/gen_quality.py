"""Quantitative generation-quality metrics for w2 sequences ``(n, 90, 14)``.

All inputs are in the w2 export encoding (``log1p`` / ``signed_log1p`` / flags).
Real and synthetic users of the same class are compared; a second real sample
of that class (``tune``, never seen by the GAN) gives the real-vs-real floor
every score should be read against.

1. ``marginal_table`` / ``aggregate_table``: per-channel day values and per-user
   aggregates (Wasserstein-1, KS, rates).
2. ``frechet_distance``: Fréchet distance between Gaussian fits of features,
   the FID recipe with the w2 churn LSTM as feature extractor (``lstm_features``).
3. ``discriminative_score``: post-hoc GRU real-vs-synthetic, ``|acc - 0.5|``
   (TimeGAN paper; 0 = indistinguishable).
4. ``predictive_score``: train-on-synthetic, test-on-real next-step MAE
   (TimeGAN paper; compare with train-on-real).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import linalg
from scipy.stats import ks_2samp, wasserstein_distance

# ---------------------------------------------------------------------------
# Scaling shared by the post-hoc networks (min-max fitted on real data only)
# ---------------------------------------------------------------------------

def minmax_fit(X):
    return X.min(axis=(0, 1)), X.max(axis=(0, 1))


def minmax_apply(X, lo, hi):
    return ((X - lo) / (hi - lo + 1e-7)).astype(np.float32)


def _subsample(X, n, rng):
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


# ---------------------------------------------------------------------------
# 1. Marginals and per-user aggregates
# ---------------------------------------------------------------------------

def marginal_table(real, synth, channels, max_values=200_000, seed=0):
    """Per channel: mean/std, share of non-zero days and Wasserstein-1.

    W1 is computed on the day values of each channel and divided by the real
    standard deviation, so channels with different scales are comparable.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c, name in enumerate(channels):
        r = real[:, :, c].ravel()
        s = synth[:, :, c].ravel()
        r_s, s_s = _subsample(r, max_values, rng), _subsample(s, max_values, rng)
        std = r.std() + 1e-7
        rows.append({
            "channel": name,
            "mean real": r.mean(), "mean synth": s.mean(),
            "std real": r.std(), "std synth": s.std(),
            "non-zero real": (r != 0).mean(), "non-zero synth": (s != 0).mean(),
            "W1 / std real": wasserstein_distance(r_s, s_s) / std,
        })
    return pd.DataFrame(rows).set_index("channel")


def user_aggregates(X, channels):
    """One row per user with business-readable summaries of the 90 days."""
    i = {name: k for k, name in enumerate(channels)}
    active = X[:, :, i["active"]]
    secs = np.expm1(np.clip(X[:, :, i["total_secs"]], 0, None))
    to_expire = X[:, -1, i["tx_days_to_expire"]]
    return pd.DataFrame({
        "active days": active.sum(axis=1),
        "active days (last 30)": active[:, -30:].sum(axis=1),
        "payment days": (X[:, :, i["tx_count"]] > 0).sum(axis=1),
        "log hours listened": np.log1p(secs.sum(axis=1) / 3600),
        "days to expire at cutoff": np.sign(to_expire) * np.expm1(np.abs(to_expire)),
        "active-day switches": np.abs(np.diff(active, axis=1)).sum(axis=1),
    })


def aggregate_table(real, synth, channels):
    """Mean of each aggregate and the two-sample KS statistic (0 = same distribution)."""
    a_r, a_s = user_aggregates(real, channels), user_aggregates(synth, channels)
    rows = []
    for col in a_r:
        ks = ks_2samp(a_r[col], a_s[col])
        rows.append({"aggregate": col, "mean real": a_r[col].mean(),
                     "mean synth": a_s[col].mean(), "KS": ks.statistic})
    return pd.DataFrame(rows).set_index("aggregate")


# ---------------------------------------------------------------------------
# 2. Fréchet distance on the w2 churn-LSTM features
# ---------------------------------------------------------------------------

@torch.no_grad()
def lstm_features(model, X_scaled, device, batch_size=2048):
    """Last hidden output ``h_T`` of the w2 recurrent classifier (before BN/head)."""
    model.to(device).eval()
    feats = []
    for start in range(0, len(X_scaled), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(X_scaled[start:start + batch_size])).to(device)
        out, _ = model.rnn(x)
        feats.append(out[:, -1, :].float().cpu().numpy())
    return np.concatenate(feats)


@torch.no_grad()
def churn_proba(model, X_scaled, device, batch_size=2048):
    model.to(device).eval()
    probs = []
    for start in range(0, len(X_scaled), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(X_scaled[start:start + batch_size])).to(device)
        probs.append(torch.sigmoid(model(x)).float().cpu().numpy().ravel())
    return np.concatenate(probs)


def frechet_distance(feat_a, feat_b):
    """``||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^{1/2})`` (FID formula)."""
    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    s_a = np.cov(feat_a, rowvar=False)
    s_b = np.cov(feat_b, rowvar=False)
    covmean = linalg.sqrtm(s_a @ s_b)
    if not np.isfinite(covmean).all():
        eye = np.eye(len(s_a)) * 1e-6
        covmean = linalg.sqrtm((s_a + eye) @ (s_b + eye))
    covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(s_a + s_b - 2 * covmean))


# ---------------------------------------------------------------------------
# 3-4. Post-hoc networks (TimeGAN paper)
# ---------------------------------------------------------------------------

class _PostHocGRU(nn.Module):
    def __init__(self, dim, hidden, out_dim):
        super().__init__()
        self.rnn = nn.GRU(dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)


def _seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def discriminative_score(real, synth, *, lo, hi, max_per_class=5000, iterations=2000,
                         batch_size=128, hidden=None, device=None, seed=0):
    """Train a GRU to tell real from synthetic; return accuracy and ``|acc - 0.5|``.

    Balanced classes, 80/20 split; the score is measured on the held-out 20%.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _seed(seed)
    rng = np.random.default_rng(seed)
    n = min(len(real), len(synth), max_per_class)
    X = np.concatenate([minmax_apply(_subsample(real, n, rng), lo, hi),
                        minmax_apply(_subsample(synth, n, rng), lo, hi)])
    y = np.r_[np.ones(n), np.zeros(n)].astype(np.float32)
    order = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = order[:cut], order[cut:]

    dim = X.shape[2]
    model = _PostHocGRU(dim, hidden or max(dim // 2, 4), 1).to(device)
    opt = torch.optim.Adam(model.parameters())
    X_t, y_t = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    model.train()
    for _ in range(iterations):
        idx = torch.from_numpy(rng.choice(tr, batch_size)).to(device)
        logit = model(X_t[idx])[:, -1, 0]
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = (model(X_t[te])[:, -1, 0] > 0).float()
        acc = float((pred == y_t[te]).float().mean())
    return {"accuracy": acc, "discriminative score": abs(acc - 0.5)}


def predictive_score(train_X, test_X, *, lo, hi, max_train=10000, iterations=5000,
                     batch_size=128, hidden=None, device=None, seed=0):
    """Train a GRU on ``train_X`` to predict day ``t+1`` (all channels) from days ``<= t``.

    Returns the mean absolute error on ``test_X`` (min-max scale of the real
    data). Train on synthetic, test on real (TSTR); TRTR is the reference.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _seed(seed)
    rng = np.random.default_rng(seed)
    tr = torch.from_numpy(minmax_apply(_subsample(train_X, max_train, rng), lo, hi)).to(device)
    te = torch.from_numpy(minmax_apply(test_X, lo, hi)).to(device)
    dim = tr.shape[2]
    model = _PostHocGRU(dim, hidden or max(dim // 2, 4), dim).to(device)
    opt = torch.optim.Adam(model.parameters())
    model.train()
    for _ in range(iterations):
        idx = torch.from_numpy(rng.integers(0, len(tr), batch_size)).to(device)
        x = tr[idx]
        loss = (torch.sigmoid(model(x[:, :-1])) - x[:, 1:]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        maes = [(torch.sigmoid(model(te[s:s + 2048, :-1])) - te[s:s + 2048, 1:]).abs().mean().item()
                * len(te[s:s + 2048]) for s in range(0, len(te), 2048)]
    return float(sum(maes) / len(te))


# ---------------------------------------------------------------------------
# 5. Domain checks for the trajectory model (payments, expiry, horizon)
# ---------------------------------------------------------------------------

def _decode_state(X, channels):
    i = {name: k for k, name in enumerate(channels)}
    known = X[:, :, i["tx_state_known"]] > 0
    since = np.expm1(np.clip(X[:, :, i["tx_days_since_last"]], 0, None))
    v = X[:, :, i["tx_days_to_expire"]]
    expiry = np.sign(v) * np.expm1(np.abs(v))
    pay = X[:, :, i["tx_count"]] > 0
    plan = np.expm1(np.clip(X[:, :, i["tx_plan_days_sum"]], 0, None))
    return known, since, expiry, pay, plan


def _close(a, b, rel=0.02, tol=1.5):
    """Equal up to the float16 rounding of the log encodings (1.5 days or 2 %)."""
    return np.abs(a - b) <= np.maximum(tol, rel * np.abs(b))


def payment_state_consistency(X, channels, horizon=7):
    """Share of days that follow the w2 payment-state rules (real users: ~95-100 %).

    The trajectory model learns *when* users pay from the state channels, so a
    synthetic user whose payments ignore its own expiry teaches the wrong rule.
    The last columns describe the forecast horizon (the last ``horizon`` days).
    """
    known, since, expiry, pay, plan = _decode_state(X, channels)
    carry = known[:, 1:] & known[:, :-1] & ~pay[:, 1:]
    pay_k = known[:, :-1] & pay[:, 1:]
    expected = np.maximum(expiry[:, :-1] - 1, 0) + plan[:, 1:]
    v = X[:, :, list(channels).index("tx_paid_sum")]
    paid = np.sign(v) * np.expm1(np.abs(v))
    return {
        "pay day: since_last = 0": float((since[pay] < 0.5).mean()) if pay.any() else np.nan,
        "no pay: since_last + 1": float(_close(since[:, 1:], since[:, :-1] + 1)[carry].mean()),
        "no pay: to_expire - 1": float(_close(expiry[:, 1:], expiry[:, :-1] - 1)[carry].mean()),
        "pay day: to_expire = max(prev-1,0) + plan": (
            float((np.abs(expiry[:, 1:] - expected) <= 2)[pay_k].mean()) if pay_k.any() else np.nan),
        "payment day rate": float(pay.mean()),
        "pay day: mean plan days": float(plan[pay].mean()) if pay.any() else np.nan,
        "pay day: mean paid (NTD)": float(paid[pay].mean()) if pay.any() else np.nan,
        f"pays in last {horizon} days": float(pay[:, -horizon:].any(axis=1).mean()),
        f"active rate last {horizon} days": float(
            (X[:, -horizon:, list(channels).index("active")] > 0).mean()),
    }


def pay_by_expiry(X, channels, window=(-10, 10)):
    """Daily payment rate against the day relative to the expiry (0 = expiry day).

    For day ``t`` the expiry known the day before gives ``rel = 1 - days_to_expire[t-1]``,
    the same alignment as ``plot_pay_by_expiry`` in w2. Returns a Series indexed by
    ``rel`` (percent) and the number of user-days behind each point.
    """
    known, _, expiry, pay, _ = _decode_state(X, channels)
    rel = np.rint(1 - expiry[:, :-1]).astype(np.int64)
    ok = known[:, :-1] & (rel >= window[0]) & (rel <= window[1])
    idx = rel[ok] - window[0]
    size = window[1] - window[0] + 1
    n = np.bincount(idx, minlength=size)
    hits = np.bincount(idx, pay[:, 1:][ok].astype(float), minlength=size)
    xs = np.arange(window[0], window[1] + 1)
    rate = pd.Series(np.where(n > 0, 100 * hits / np.maximum(n, 1), np.nan), index=xs, name="% paying")
    return rate, pd.Series(n, index=xs, name="user-days")


def payments_near_expiry(X, channels, days=3):
    """Share of payment days within ``days`` of the expiry known the day before.

    Real manual payers renew around their expiry, so this share is high; a
    generator that places payments at random days scores low. Robust to the
    number of synthetic user-days near the expiry (unlike comparing curves).
    """
    known, _, expiry, pay, _ = _decode_state(X, channels)
    p = known[:, :-1] & pay[:, 1:]
    rel = 1 - expiry[:, :-1][p]
    return float((np.abs(rel) <= days).mean()) if p.any() else np.nan


# ---------------------------------------------------------------------------
# 5. Mode-collapse signals (TIMEGAN_DIAGNOSIS.md, section 7)
# ---------------------------------------------------------------------------

def collapse_signals(fit_X, X, channels, *, reference=None, n=2000, seed=0):
    """Per-user extremes and nearest-neighbour diversity of ``X``.

    * ``0 active days`` / ``>= 85 active days``: all-or-nothing users.
    * ``active switches (median)``: days where ``active`` flips; 0 = flat sequences.
    * ``no payment``: share of users without any payment day.
    * ``NN within``: median distance of a user of ``X`` to its nearest *other*
      user of ``X`` (low = near-copies, mode collapse).
    * ``NN reference -> X``: median distance of a ``reference`` user (real
      ``tune``) to its nearest user of ``X`` (high = real user types missing).
      With ``X`` = ``reference`` it compares ``reference`` with ``fit``.

    Users are flattened to ``90 x 14`` vectors, min-max scaled with ``fit_X``;
    ``n`` users per set.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    i = {name: k for k, name in enumerate(channels)}
    lo, hi = minmax_fit(fit_X)

    def flat(A):
        return minmax_apply(_subsample(A, n, rng), lo, hi).reshape(min(len(A), n), -1)

    active = X[:, :, i["active"]] > 0.5
    days = active.sum(axis=1)
    out = {
        "0 active days": float((days == 0).mean()),
        ">= 85 active days": float((days >= 85).mean()),
        "active switches (median)": float(np.median(np.abs(np.diff(active.astype(int), axis=1)).sum(axis=1))),
        "no payment": float(((X[:, :, i["tx_count"]] > 0).sum(axis=1) == 0).mean()),
    }
    F = flat(X)
    d, _ = NearestNeighbors(n_neighbors=2).fit(F).kneighbors(F)
    out["NN within"] = float(np.median(d[:, 1]))
    if reference is not None:
        target = flat(fit_X) if reference is X else F
        d, _ = NearestNeighbors(n_neighbors=1).fit(target).kneighbors(flat(reference))
        out["NN reference -> X"] = float(np.median(d[:, 0]))
    return out
