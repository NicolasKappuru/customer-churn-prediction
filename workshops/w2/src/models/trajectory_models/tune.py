"""Optuna search for the trajectory models.

Each trial trains on ``fit``, early-stops on ``stop`` and returns the weighted
composite loss on ``tune``. The loss weights are fixed before the search, so
Optuna only chooses the architecture and the optimiser. There is no threshold
search: option B samples from the predicted probabilities.
"""

import hashlib
import json

import optuna
import pandas as pd
import torch

from .config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIMS,
    HORIZON,
    INPUT_DAYS,
    LAYER_DIM_RANGE,
    LR_RANGE,
    N_TRIALS,
    PATIENCE,
    SEED,
    TARGET_NAMES,
    WEIGHT_DECAY_RANGE,
    model_dirs,
)
from .model import MODELS
from .train import evaluate_loss, set_seed, train_model


def _contract(data, batch_size, model_name, loss_weights):
    """Fingerprint data, split, targets and loss so a study is only reused when valid."""
    meta = data.meta
    return {
        "task": "trajectory",
        "model_name": model_name,
        "dataset_version": meta.get("dataset_version"),
        "hash_train_feb": meta.get("hash_train_feb"),
        "hash_test": meta.get("hash_test"),
        "split_sizes": {part: int(len(rows)) for part, rows in data.split.items()},
        "input_days": INPUT_DAYS,
        "horizon": HORIZON,
        "targets": list(TARGET_NAMES),
        "loss_weights": {k: round(float(v), 6) for k, v in loss_weights.items()},
        "batch_size": int(batch_size),
        "torch": torch.__version__,
    }


def _contract_hash(contract):
    return hashlib.sha1(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]


def _get_study(contract, seed, n_trials, model_name):
    optuna_dir = model_dirs(model_name)["optuna"]
    optuna_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(optuna_dir / f'{model_name}_study.sqlite3').as_posix()}"
    study = optuna.create_study(
        study_name=f"{model_name}_traj_{_contract_hash(contract)}",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=max(1, min(3, n_trials // 2))
        ),
    )
    if study.user_attrs.get("contract") != contract:
        study.set_user_attr("contract", contract)
    return study


def _suggest_params(trial, batch_size):
    return {
        "layer_dim": trial.suggest_int("layer_dim", *LAYER_DIM_RANGE),
        "hidden_dim": trial.suggest_categorical("hidden_dim", list(HIDDEN_DIMS)),
        "lr": trial.suggest_float("lr", *LR_RANGE, log=True),
        "weight_decay": trial.suggest_float("weight_decay", *WEIGHT_DECAY_RANGE, log=True),
        "batch_size": int(batch_size),
    }


def _release_gpu_cache(study, trial):
    """Return the trial's cached GPU blocks to the driver.

    Each trial has a different shape, so PyTorch's caching allocator keeps
    growing across trials and never hands memory back to other processes
    (another notebook on the same GPU then runs out of memory).
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_objective(data, model_name, loss_weights, *, epochs, patience,
                    batch_size, device, seed, amp, verbose):
    model_cls = MODELS[model_name]

    def objective(trial):
        params = _suggest_params(trial, batch_size)
        print(f"trial {trial.number}: starting | params={params}", flush=True)
        set_seed(seed + trial.number)
        model = model_cls(
            input_dim=data.input_dim,
            hidden_dim=params["hidden_dim"],
            layer_dim=params["layer_dim"],
        )
        result = train_model(
            model, data, data.rows("fit"), data.rows("stop"),
            loss_weights=loss_weights, epochs=epochs, lr=params["lr"],
            weight_decay=params["weight_decay"], batch_size=batch_size,
            patience=patience, device=device, phase="fit",
            seed=seed + trial.number, amp=amp, verbose=verbose,
            log_prefix=f"trial {trial.number} | ",
        )
        tune_loss = evaluate_loss(
            model, data, data.rows("tune"), loss_weights=loss_weights,
            phase="fit", device=device,
        )
        for key in ("best_epoch", "stopped_epoch", "best_stop_loss", "history"):
            trial.set_user_attr(key, result[key])
        trial.set_user_attr(
            "n_parameters", int(sum(p.numel() for p in model.parameters()))
        )
        print(f"trial {trial.number}: tune_loss={tune_loss:.5f}"
              f" | best_epoch={result['best_epoch']}", flush=True)
        return tune_loss

    return objective


def _save_study_artifacts(study, best_params, model_name):
    optuna_dir = model_dirs(model_name)["optuna"]
    study.trials_dataframe().to_csv(optuna_dir / f"{model_name}_trials.csv", index=False)
    (optuna_dir / f"{model_name}_best_params.json").write_text(
        json.dumps(best_params, indent=2)
    )
    pd.DataFrame(study.best_trial.user_attrs.get("history", [])).to_csv(
        optuna_dir / f"{model_name}_best_history.csv", index=False
    )


def tune_model(
    data,
    *,
    loss_weights,
    model_name="lstm",
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    device=None,
    seed=SEED,
    amp=True,
    verbose=False,
):
    """Search hyper-parameters for one family and return the best configuration."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model_name = model_name.lower()
    contract = _contract(data, batch_size, model_name, loss_weights)
    study = _get_study(contract, seed, n_trials, model_name)

    # Only finished trials use the budget; FAIL / stale RUNNING trials are retried.
    done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
               for t in study.trials)
    remaining = max(0, n_trials - done)
    print(f"Optuna budget={n_trials} | remaining={remaining}", flush=True)
    if remaining:
        study.optimize(
            _make_objective(
                data, model_name, loss_weights, epochs=epochs, patience=patience,
                batch_size=batch_size, device=device, seed=seed, amp=amp,
                verbose=verbose,
            ),
            n_trials=remaining,
            gc_after_trial=True,
            callbacks=[_release_gpu_cache],
            show_progress_bar=True,
        )
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f"The {model_name} study has no completed trial.")

    best = study.best_trial
    best_params = {**study.best_params, "batch_size": int(batch_size)}
    _save_study_artifacts(study, best_params, model_name)
    result = {
        "model_name": model_name,
        "study": study,
        "best_params": best_params,
        "best_value": float(study.best_value),
        "history": best.user_attrs.get("history", []),
        "best_epoch": int(best.user_attrs.get("best_epoch", 0)),
        "stopped_epoch": int(best.user_attrs.get("stopped_epoch", 0)),
        "n_parameters": int(best.user_attrs.get("n_parameters", 0)),
        "trial_number": int(best.number),
        "contract_hash": _contract_hash(contract),
    }
    print(f"best trial #{result['trial_number']}: tune_loss={result['best_value']:.5f}"
          f" | best_epoch={result['best_epoch']} | params={best_params}", flush=True)
    return result
