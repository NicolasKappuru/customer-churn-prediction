"""Qualitative figures for the w5 GANs (matplotlib, saved as PNG).

Colors follow the entity, never the order: real users are always blue, each
GAN keeps its own color across every figure (``SOURCE_COLORS``).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SOURCE_COLORS = {
    "real": "#2a78d6",          # blue
    "timegan": "#eb6834",       # orange
    "wtimegan": "#1baf7a",      # aqua (v1, GRU critic)
    "wtimegan_tcn": "#d4a900",  # gold (v2, TCN critic)
}
# Display names used by the notebooks -> entity key. The 4 colors pass the
# colorblind / normal-vision checks on every pair (dataviz validator, light mode).
_ALIASES = (("real", "real"), ("wtimegan_tcn", "wtimegan_tcn"), ("wtimegan v2", "wtimegan_tcn"),
            ("wtimegan", "wtimegan"), ("timegan", "timegan"))
GRID = {"color": "#d9d9d9", "linewidth": 0.6}


def color(source):
    """Color of a source, from its key (``"wtimegan"``) or its display name
    (``"WTimeGAN v1 (GRU critic) · label 1"``, ``"real (tune)"``). Gray if unknown."""
    name = source.lower()
    key = next((k for prefix, k in _ALIASES if name.startswith(prefix)), None)
    return SOURCE_COLORS.get(key, "#6b6b6b")


def _style(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_gan_history(history, title, save_path=None):
    """One panel per training phase of TimeGAN (or any GAN with the same columns).

    The joint phase is split in two panels (adversarial losses vs reconstruction /
    supervised / moment losses) because their scales differ by two orders of magnitude.
    """
    adversarial = {"d_loss", "g_loss_u", "g_loss_u_e"}
    critic = {"w_distance", "gradient_penalty"}   # Wasserstein TimeGAN only
    weights = {"g_weight_norm"}                    # Wasserstein TimeGAN v2 only
    phases = [p for p in ("embedding", "supervised", "joint") if p in set(history["phase"])]
    phases += [p for p in history["phase"].unique() if p not in phases]
    panels = []
    for phase in phases:
        part = history[history["phase"] == phase]
        losses = [c for c in part.columns
                  if c not in ("phase", "iteration", "d_updated") and part[c].notna().any()]
        adv = [c for c in losses if c in adversarial]
        crit = [c for c in losses if c in critic]
        norm = [c for c in losses if c in weights]
        rest = [c for c in losses if c not in adversarial | critic | weights]
        if adv and rest:
            panels.append((f"{phase} · adversarial", part, adv))
            if crit:
                panels.append((f"{phase} · critic", part, crit))
            panels.append((f"{phase} · reconstruction", part, rest))
            if norm:
                panels.append((f"{phase} · generator weights", part, norm))
        else:
            panels.append((f"{phase} phase", part, losses))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    for ax, (panel_title, part, losses) in zip(axes[0], panels):
        for loss in losses:
            ax.plot(part["iteration"], part[loss], linewidth=2, label=loss)
        ax.set_title(panel_title)
        ax.set_xlabel("iteration")
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("loss")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_embedding_2d(features, title, save_path=None, seed=0, max_points=2000):
    """PCA and t-SNE of ``features`` (dict source -> (n, d)), as in the TimeGAN paper."""
    rng = np.random.default_rng(seed)
    names = list(features)
    sampled = {k: v[rng.choice(len(v), min(len(v), max_points), replace=False)] for k, v in features.items()}
    stacked = np.concatenate([sampled[k] for k in names])
    owner = np.concatenate([[k] * len(sampled[k]) for k in names])
    stacked = (stacked - stacked.mean(0)) / (stacked.std(0) + 1e-7)
    pca = PCA(2, random_state=seed).fit(stacked[owner == names[0]])
    coords = {"PCA": pca.transform(stacked),
              "t-SNE": TSNE(2, init="pca", perplexity=40, random_state=seed).fit_transform(stacked)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (method, xy) in zip(axes, coords.items()):
        for k in names:
            m = owner == k
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.35, color=color(k),
                       label=k, edgecolors="none")
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color("#bfbfbf")
    axes[0].legend(frameon=False, markerscale=2.5)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_daily_profiles(sources, channels, title, save_path=None):
    """Mean of each channel per day (line) with the 25-75 % band, one panel per channel."""
    n = len(channels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), sharex=True, squeeze=False)
    days = np.arange(next(iter(sources.values())).shape[1])
    for c, name in enumerate(channels):
        ax = axes[c // cols][c % cols]
        for source, X in sources.items():
            col = color(source)
            values = X[:, :, c]
            ax.plot(days, values.mean(0), color=col, linewidth=2, label=source)
            q25, q75 = np.percentile(values, [25, 75], axis=0)
            ax.fill_between(days, q25, q75, color=col, alpha=0.12, linewidth=0)
        ax.set_title(name, fontsize=10)
        _style(ax)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("day (0 = oldest, 89 = cutoff)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_user_slices(sources, channels, n_users=3, seed=0, save_path=None):
    """Heatmaps of individual users (rows = days, columns = channels), each channel
    scaled by the real users' range so real and synthetic panels are comparable."""
    rng = np.random.default_rng(seed)
    real = next(iter(sources.values()))
    lo, hi = real.min(axis=(0, 1)), real.max(axis=(0, 1))
    fig, axes = plt.subplots(len(sources), n_users, figsize=(3.2 * n_users, 3.4 * len(sources)),
                             squeeze=False)
    for r, (source, X) in enumerate(sources.items()):
        for k, i in enumerate(rng.choice(len(X), n_users, replace=False)):
            ax = axes[r][k]
            img = (X[i] - lo) / (hi - lo + 1e-7)
            ax.imshow(img, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{source} · user {i}", fontsize=9)
            ax.set_xticks(range(len(channels)))
            ax.set_xticklabels(channels if r == len(sources) - 1 else [], rotation=90, fontsize=7)
            ax.set_yticks([0, 30, 60, 89] if k == 0 else [])
            if k == 0:
                ax.set_ylabel("day")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_aggregate_hists(aggregates, save_path=None):
    """Histogram of each per-user aggregate (dict source -> DataFrame)."""
    cols = list(next(iter(aggregates.values())).columns)
    fig, axes = plt.subplots(2, int(np.ceil(len(cols) / 2)), figsize=(5 * int(np.ceil(len(cols) / 2)), 6.4),
                             squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        values = np.concatenate([a[col].to_numpy() for a in aggregates.values()])
        bins = np.histogram_bin_edges(values[np.isfinite(values)], bins=40)
        for source, a in aggregates.items():
            ax.hist(a[col], bins=bins, density=True, histtype="step", linewidth=2,
                    color=color(source), label=source)
        ax.set_title(col, fontsize=10)
        _style(ax)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes[0][0].legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("density")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_scenarios_roc_pr(runs, colors, save_path=None):
    """ROC and PR on March for every bridge scenario (one line per scenario)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, run in runs.items():
        y, p = run["y_test"], run["proba_test"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        style = "--" if "baseline" in name else "-"
        ax_roc.plot(fpr, tpr, style, color=colors[name], linewidth=2,
                    label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        ax_pr.plot(rec, prec, style, color=colors[name], linewidth=2,
                   label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax_roc.plot([0, 1], [0, 1], ":", color="#9e9e9e", linewidth=1)
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC · March")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-Recall · March")
    for ax in (ax_roc, ax_pr):
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_pay_by_expiry(curves, title, save_path=None):
    """Payment rate aligned to the expiry (dict source -> Series from ``pay_by_expiry``)."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    styles = [("rebuilt", "--", "s"), ("generated", ":", "^")]   # variant of one GAN -> line + marker
    for source, rate in curves.items():
        line, marker = next(((ls, m) for word, ls, m in styles if word in source), ("-", "o"))
        ok = rate.notna()
        ax.plot(rate.index[ok], rate[ok], linewidth=2, marker=marker, markersize=5,
                linestyle=line, color=color(source), label=source)
    ax.axvline(0, color="#9e9e9e", linestyle=":", linewidth=1)
    ax.set_xticks(range(int(min(r.index.min() for r in curves.values())),
                        int(max(r.index.max() for r in curves.values())) + 1, 2))
    ax.set_xlabel("day relative to the known expiry (0 = expiry day)")
    ax.set_ylabel("% of users paying that day")
    ax.set_title(title, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, save_path)
