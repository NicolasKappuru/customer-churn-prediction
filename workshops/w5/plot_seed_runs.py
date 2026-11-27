"""Figures for the Optuna + 5-seed experiments (run after ``run_experiments.py summary``).

    python workshops/w5/plot_seed_runs.py [--out docs/w5/figures/experiments]

1. ``frechet_seeds.png``: Fréchet distance of every seed, per GAN and class, with the
   mean and the single hand-set run of the notebooks for reference.
2. For each class, the best seed of each GAN (lowest Fréchet):
   ``embedding_label{k}_{gan}.png`` (PCA / t-SNE of w2-LSTM features, real tune vs synthetic)
   and ``daily_profiles_label{k}.png`` (real fit vs both GANs).
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "evaluation"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gen_plots as gp  # noqa: E402
import run_experiments as rx  # noqa: E402
from w2_sequences import MODEL_CHANNELS  # noqa: E402

GANS = ("timegan", "wtimegan_tcn")
SHORT = {"timegan": "TimeGAN", "wtimegan_tcn": "WTimeGAN v2"}
CLASS = {1: "churners", 0: "non-churners"}
# Single hand-set run of the notebooks (seed 110), Table tab:w5_quality of the report.
SINGLE_RUN = {("timegan", 1): 0.964, ("wtimegan_tcn", 1): 0.242,
              ("timegan", 0): 0.087, ("wtimegan_tcn", 0): 0.027}
INK, MUTED = "#262626", "#6b6b6b"


def plot_frechet_seeds(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, label in zip(axes, (1, 0)):
        for x, gan in enumerate(GANS):
            m = metrics[(metrics.gan == gan) & (metrics.label == label)].sort_values("seed")
            col = gp.color(gan)
            jitter = np.linspace(-0.12, 0.12, len(m))
            ax.scatter(x + jitter, m.frechet, s=64, color=col, edgecolors="white", linewidths=2, zorder=3,
                       label=f"{SHORT[gan]} · seed" if label == 1 else None)
            mean = m.frechet.mean()
            ax.hlines(mean, x - 0.25, x + 0.25, color=col, linewidth=2, zorder=2)
            ax.annotate(f"mean {mean:.2f}", (x + 0.27, mean), va="center", fontsize=9, color=INK)
            ax.scatter(x, SINGLE_RUN[gan, label], s=80, marker="D", facecolors="white", edgecolors=col,
                       linewidths=2, zorder=4, label="single hand-set run (seed 110)" if (label, x) == (1, 1) else None)
        ax.set_xticks(range(len(GANS)), [SHORT[g] for g in GANS])
        ax.set_xlim(-0.5, 1.9)
        ax.set_title(CLASS[label].capitalize(), fontsize=11, color=INK)
        gp._style(ax)
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Fréchet distance (lower is better)")
    axes[0].set_ylim(bottom=0)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.tight_layout()
    gp._save(fig, path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parents[1] / "docs" / "w5" / "figures" / "experiments")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(rx.METRICS_CSV)
    plot_frechet_seeds(metrics, args.out / "frechet_seeds.png")
    print("saved", args.out / "frechet_seeds.png")

    best = metrics.loc[metrics.groupby(["gan", "label"]).frechet.idxmin()]
    print(best[["gan", "label", "seed", "frechet"]].to_string(index=False))

    ctx = rx.Context([1, 0])
    for label in (1, 0):
        f_real, _ = ctx.lstm_view(ctx.real["tune", label])
        profiles = {"real (fit)": ctx.real["fit", label]}
        for gan in GANS:
            row = best[(best.gan == gan) & (best.label == label)].iloc[0]
            seed = int(row.seed)
            X = np.load(rx.SEEDS_DIR / f"seed{seed}" / gan / f"label{label}" / "synthetic.npy")
            _, _, X_sub = ctx.frechet(X, label)
            f_x, _ = ctx.lstm_view(X_sub)
            name = f"{SHORT[gan]} seed {seed}"
            gp.plot_embedding_2d({f"real (tune) · {CLASS[label]}": f_real, f"{name} · {CLASS[label]}": f_x},
                                 f"Real vs {name} (best of 5) · {CLASS[label]} · Fréchet {row.frechet:.3f}",
                                 seed=rx.EVAL_SEED, save_path=args.out / f"embedding_label{label}_{gan}.png")
            plt.close("all")
            profiles[name] = X
        gp.plot_daily_profiles(profiles, MODEL_CHANNELS, f"Daily profile per channel · best seeds · {CLASS[label]}",
                               save_path=args.out / f"daily_profiles_label{label}.png")
        plt.close("all")
        print("saved figures for label", label)


if __name__ == "__main__":
    main()
