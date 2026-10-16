"""Persistent Optuna studies shared by the tabular model families."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import optuna
from optuna.trial import TrialState

from ..constants import SPLIT_PROTOCOL


def make_contract(meta, split, smoke, env, extra=None):
    """Fingerprint data, split, libraries, and model settings for study reuse."""
    return {
        "dataset_version": meta["dataset_version"],
        **{
            key: meta[key]
            for key in ("hash_train_feb", "hash_test")
        },
        "cutoffs": {cohort: meta["cutoffs"][cohort] for cohort in ("train_feb", "test")},
        "split_protocol": SPLIT_PROTOCOL,
        "split_sizes": {part: int(len(rows)) for part, rows in split.items()},
        "smoke": smoke,
        **(extra or {}),
        "libs": {
            key: env[key]
            for key in ("xgboost", "optuna", "sklearn")
        },
    }


def contract_hash(contract):
    payload = json.dumps(contract, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def get_study(model_id, contract, storage, seed, n_trials, tag=""):
    name = f"{model_id}_{contract_hash(contract)}"
    if tag:
        name += f"_{tag}"

    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=max(1, min(3, n_trials // 2)),
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
        ),
    )
    # Only persist the contract when it changed, to avoid a storage write per resume.
    if study.user_attrs.get("contract") != contract:
        study.set_user_attr("contract", contract)
    return study


def remaining_trials(study, total):
    """Count attempted trials, including failed ones, against the budget."""
    done = sum(trial.state != TrialState.WAITING for trial in study.trials)
    return max(0, total - done)


def _enqueue_new(study, candidates):
    existing = {json.dumps(trial.params, sort_keys=True) for trial in study.trials}
    for params in candidates:
        key = json.dumps(params, sort_keys=True)
        if key not in existing:
            study.enqueue_trial(params, skip_if_exists=True)
            existing.add(key)


def count_states(study):
    """Count complete / pruned / failed trials in a single pass."""
    states = Counter(trial.state for trial in study.trials)
    return {
        "complete": states[TrialState.COMPLETE],
        "pruned": states[TrialState.PRUNED],
        "failed": states[TrialState.FAIL],
    }


def save_study_artifacts(study, model_id, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(
        out_dir / f"{model_id}_trials.csv", index=False
    )
    (out_dir / f"{model_id}_best_params.json").write_text(
        json.dumps(study.best_params, indent=2)
    )


def run_study(model_id, objective, study, total_trials, out_dir, enqueue=None):
    """Resume the budget, save artifacts, and return the selected study."""
    if enqueue:
        _enqueue_new(study, enqueue)

    remaining = remaining_trials(study, total_trials)
    print(f"{model_id}: budget={total_trials}, remaining={remaining}", flush=True)

    def objective_with_progress(trial):
        print(f"{model_id}: starting trial #{trial.number}", flush=True)
        return objective(trial)

    def report_trial(_study, trial):
        seconds = trial.duration.total_seconds() if trial.duration else 0.0
        print(
            f"{model_id}: trial #{trial.number} {trial.state.name}"
            f" | log-loss={trial.value} | {seconds:.1f}s",
            flush=True,
        )

    if remaining > 0:
        study.optimize(
            objective_with_progress,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=True,
            callbacks=[report_trial],
        )

    counts = count_states(study)
    if counts["complete"] == 0:
        raise RuntimeError(f"{model_id}: no completed trial; cannot select params.")

    save_study_artifacts(study, model_id, out_dir)
    print(f"{model_id}: best tune log-loss={study.best_value:.5f} | trials {counts}")
    return study, counts
