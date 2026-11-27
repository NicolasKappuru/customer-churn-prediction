"""Hyper-parameter search (Optuna) and multi-seed runs for TimeGAN and WTimeGAN v2.

Three steps, meant to run unattended (see ``run_all.sh``):

    python workshops/w5/run_experiments.py optuna --gan timegan --trials 15
    python workshops/w5/run_experiments.py optuna --gan wtimegan_tcn --trials 15
    python workshops/w5/run_experiments.py seeds --gans timegan wtimegan_tcn --seeds 0 1 2 3 4
    python workshops/w5/run_experiments.py summary

``optuna``  small search on one class (run it once per class) with a reduced
            budget. Objective: Fréchet distance between w2-LSTM features of the
            synthetic users and of real ``fit`` users (the headline metric of
            the w5 notebooks; lower is better). The current configuration is
            enqueued as trial 0, so the search can only match or improve it.
``seeds``   full-budget training of each GAN with its best parameters, for every
            seed and class, followed by the quality metrics of the notebook
            (Fréchet, discriminative, predictive, w2-LSTM churn probability).
            Seeds are the outer loop, so an interrupted run still has balanced
            results; finished runs are reused (``train_or_load``) on restart.
``summary`` mean ± std over seeds per GAN and class.

Everything is written under ``artifacts/experiments/``.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("models", "evaluation"):
    sys.path.insert(0, str(HERE / sub))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import gen_quality as gq  # noqa: E402
import w2_bridge as bridge  # noqa: E402
from gan_runner import train_or_load  # noqa: E402
from TimeGAN import timegan  # noqa: E402
from w2_sequences import MODEL_CHANNELS, postprocess  # noqa: E402
from WTimeGAN import wtimegan  # noqa: E402

EXP_DIR = Path(os.environ.get("W5_EXP_DIR", HERE / "artifacts" / "experiments"))
OPTUNA_DIR = EXP_DIR / "optuna"
SEEDS_DIR = EXP_DIR / "seeds"
METRICS_CSV = EXP_DIR / "seed_metrics.csv"
EVAL_SEED = 110                     # fixed subsampling of the real users in every evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurations of the w5 notebooks (TimeGAN baseline and WTimeGAN v2).
BASE = {
    "timegan": {"hidden_dim": 56, "num_layer": 3, "batch_size": 128, "module": "gru", "lr": 1e-3},
    "wtimegan_tcn": {"hidden_dim": 56, "num_layer": 3, "batch_size": 128, "module": "gru",
                     "n_critic": 5, "lambda_gp": 10.0, "gan_betas": [0.5, 0.9],
                     "critic": "tcn", "critic_channels": 64, "g_lr": 1e-3, "d_lr": 1e-4},
}
TRAIN_FN = {"timegan": timegan, "wtimegan_tcn": wtimegan}
NAMES = {"timegan": "TimeGAN", "wtimegan_tcn": "WTimeGAN v2 (TCN critic)"}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def suggest(trial, gan):
    """Search space: same capacity knobs for both GANs plus their learning rates."""
    params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [28, 56, 84]),
        "num_layer": trial.suggest_int("num_layer", 2, 4),
    }
    if gan == "timegan":
        params["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    else:
        params["g_lr"] = trial.suggest_float("g_lr", 3e-4, 3e-3, log=True)
        params["d_lr"] = trial.suggest_float("d_lr", 3e-5, 3e-4, log=True)
        params["n_critic"] = trial.suggest_categorical("n_critic", [3, 5])
    return params


def budget(gan, iterations, joint_iterations):
    out = {"iterations": iterations}
    if gan == "wtimegan_tcn":
        out["joint_iterations"] = joint_iterations
    return out


# ---------------------------------------------------------------------------
# Shared evaluation context (w2 data + w2 LSTM as feature extractor)
# ---------------------------------------------------------------------------

class Context:
    def __init__(self, labels):
        self.data = bridge.load_w2_data()
        self.specs = [next(s for s in self.data.channel_specs if s["name"] == c) for c in MODEL_CHANNELS]
        self.real = {(part, label): self.data.raw_rows(bridge.real_rows(self.data, part, label))
                     for part in ("fit", "tune") for label in labels}
        self.lstm, ckpt = bridge.load_w2_classifier("lstm")
        self.threshold = float(ckpt["threshold"])
        self._ref = {}

    def lstm_view(self, X):
        scaled = bridge.scale(self.data, X, "development")
        return gq.lstm_features(self.lstm, scaled, DEVICE), gq.churn_proba(self.lstm, scaled, DEVICE)

    def reference(self, label):
        """Fit-user features (subsample of the tune size) and min-max of fit, cached."""
        if label not in self._ref:
            fit_X, tune_X = self.real["fit", label], self.real["tune", label]
            rng = np.random.default_rng(EVAL_SEED)
            fit_sub = fit_X[rng.choice(len(fit_X), len(tune_X), replace=False)]
            self._ref[label] = {"f_fit": self.lstm_view(fit_sub)[0], "minmax": gq.minmax_fit(fit_X)}
        return self._ref[label]

    def frechet(self, X, label):
        ref = self.reference(label)
        n_ref = len(self.real["tune", label])
        rng = np.random.default_rng(EVAL_SEED)
        X_sub = X[rng.choice(len(X), n_ref, replace=False)] if len(X) > n_ref else X
        f_x, p_x = self.lstm_view(X_sub)
        return gq.frechet_distance(f_x, ref["f_fit"]), p_x, X_sub

    def quality(self, X, label):
        fd, p_x, X_sub = self.frechet(X, label)
        lo, hi = self.reference(label)["minmax"]
        fit_X, tune_X = self.real["fit", label], self.real["tune", label]
        disc = gq.discriminative_score(fit_X, X_sub, lo=lo, hi=hi, device=DEVICE, seed=EVAL_SEED)
        pred = gq.predictive_score(X, tune_X, lo=lo, hi=hi, device=DEVICE, seed=EVAL_SEED)
        return {
            "frechet": fd,
            "discriminative": disc["discriminative score"],
            "predictive_mae": pred,
            "mean_churn_proba": float(p_x.mean()),
            "share_as_label": float(((p_x >= self.threshold) == label).mean()),
        }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def cmd_optuna(args):
    import optuna

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    ctx = Context([args.label])
    fit_X = ctx.real["fit", args.label]
    n_gen = len(ctx.real["tune", args.label])
    study = optuna.create_study(
        study_name=f"{args.gan}_label{args.label}", direction="minimize",
        storage=f"sqlite:///{OPTUNA_DIR / f'{args.gan}_label{args.label}.db'}",
        sampler=optuna.samplers.TPESampler(seed=args.seed), load_if_exists=True)
    if len(study.trials) == 0:
        searched = suggest(optuna.trial.FixedTrial(BASE[args.gan]), args.gan)
        study.enqueue_trial({k: BASE[args.gan][k] for k in searched})

    def objective(trial):
        params = {**BASE[args.gan], **suggest(trial, args.gan),
                  **budget(args.gan, args.iterations, args.joint_iterations),
                  "seed": args.seed, "n_samples": n_gen, "log_every": 10**9}
        start = time.perf_counter()
        generated = TRAIN_FN[args.gan](fit_X, params)
        raw = np.stack(generated).astype(np.float32)
        synth = postprocess(raw, ctx.specs, seed=args.seed, upper=fit_X.max(axis=(0, 1)))
        fd = ctx.frechet(synth, args.label)[0]
        minutes = (time.perf_counter() - start) / 60
        trial.set_user_attr("minutes", round(minutes, 2))
        log(f"{args.gan} trial {trial.number}: FD={fd:.4f} ({minutes:.1f} min) {trial.params}")
        return fd

    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(args.trials - done, 0)
    log(f"optuna {args.gan} label={args.label}: {done} done, running {remaining} more")
    study.optimize(objective, n_trials=remaining, catch=(Exception,))

    study.trials_dataframe().to_csv(OPTUNA_DIR / f"{args.gan}_label{args.label}_trials.csv", index=False)
    best = {**BASE[args.gan], **study.best_params}
    (OPTUNA_DIR / f"{args.gan}_label{args.label}_best_params.json").write_text(json.dumps(best, indent=2))
    log(f"best {args.gan}: FD={study.best_value:.4f} trial {study.best_trial.number} {best}")


def best_params(gan, label):
    """Each class is a separate unconditional GAN, so each has its own search."""
    path = OPTUNA_DIR / f"{gan}_label{label}_best_params.json"
    if path.is_file():
        return json.loads(path.read_text())
    log(f"WARNING: {path} missing, using the notebook configuration for {gan} label={label}")
    return dict(BASE[gan])


def cmd_seeds(args):
    ctx = Context(args.labels)
    done = set()
    if METRICS_CSV.is_file():
        prev = pd.read_csv(METRICS_CSV)
        done = set(zip(prev["gan"], prev["seed"], prev["label"]))
    failures = 0
    for seed in args.seeds:
        for gan in args.gans:
            for label in args.labels:
                params = {**best_params(gan, label), **budget(gan, args.iterations, args.joint_iterations),
                          "seed": seed}
                if (gan, seed, label) in done:
                    log(f"skip {gan} seed={seed} label={label} (already evaluated)")
                    continue
                try:
                    log(f"train {gan} seed={seed} label={label} {params}")
                    run = train_or_load(gan, TRAIN_FN[gan], ctx.real["fit", label], ctx.specs, params,
                                        SEEDS_DIR / f"seed{seed}", label=label)
                    log(f"evaluate {gan} seed={seed} label={label}")
                    row = {"gan": gan, "seed": seed, "label": label,
                           "train_minutes": run["run"]["train_seconds"] / 60,
                           **ctx.quality(run["synthetic"], label)}
                    pd.DataFrame([row]).to_csv(METRICS_CSV, mode="a", index=False,
                                               header=not METRICS_CSV.is_file())
                    log(f"done {gan} seed={seed} label={label}: "
                        + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)))
                except Exception:
                    failures += 1
                    log(f"FAILED {gan} seed={seed} label={label}\n{traceback.format_exc()}")
                finally:
                    torch.cuda.empty_cache()
    cmd_summary(args)
    if failures:
        sys.exit(f"{failures} run(s) failed, see the log")


def cmd_summary(_args):
    if not METRICS_CSV.is_file():
        log("no seed metrics yet")
        return
    df = pd.read_csv(METRICS_CSV)
    metrics = ["frechet", "discriminative", "predictive_mae", "mean_churn_proba",
               "share_as_label", "train_minutes"]
    summary = df.groupby(["label", "gan"])[metrics].agg(["mean", "std", "count"])
    summary.to_csv(EXP_DIR / "seed_summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.round(4))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("optuna")
    p.add_argument("--gan", choices=tuple(BASE), required=True)
    p.add_argument("--label", type=int, choices=(0, 1), default=1)
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--joint-iterations", type=int, default=600)
    p.add_argument("--seed", type=int, default=110)
    p.set_defaults(func=cmd_optuna)

    p = sub.add_parser("seeds")
    p.add_argument("--gans", nargs="+", choices=tuple(BASE), default=list(BASE))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--labels", nargs="+", type=int, choices=(0, 1), default=[1, 0])
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--joint-iterations", type=int, default=10000)
    p.set_defaults(func=cmd_seeds)

    p = sub.add_parser("summary")
    p.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
