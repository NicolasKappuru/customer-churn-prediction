"""Optuna tuning for the w2 LSTM."""

import hashlib
import json

import optuna
import torch
from sklearn.metrics import log_loss

from ..evaluation.metrics import select_threshold
from .config import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIMS,
    INFER_BATCH_SIZE,
    LAYER_DIM_RANGE,
    LR_RANGE,
    N_TRIALS,
    OPTUNA_DIR,
    OUTPUT_DIM,
    PATIENCE,
    SEED,
    WEIGHT_DECAY_RANGE,
)
from .model import LSTMModel
from .train import predict_proba, set_seed, train_model


def _contract(data, batch_size):
    """Fingerprint the data, split and model input for study reuse."""
    meta = data.meta
    return {
        "dataset_version": meta["dataset_version"],
        "hash_train_feb": meta["hash_train_feb"],
        "hash_test": meta["hash_test"],
        "cutoffs": meta["cutoffs"],
        "split_sizes": {
            part: int(len(rows)) for part, rows in data.split.items()
        },
        "input_dim": data.input_dim,
        "batch_size": int(batch_size),
        "torch": torch.__version__,
    }


def _contract_hash(contract):
    payload = json.dumps(contract, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def _get_study(contract, seed, n_trials):
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(OPTUNA_DIR / 'lstm_study.sqlite3').as_posix()}"
    name = f"lstm_{_contract_hash(contract)}"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=max(1, min(3, n_trials // 2)),
        ),
    )
    if study.user_attrs.get("contract") != contract:
        study.set_user_attr("contract", contract)
    return study


def _suggest_params(trial, batch_size):
    return {
        "layer_dim": trial.suggest_int(
            "layer_dim", LAYER_DIM_RANGE[0], LAYER_DIM_RANGE[1]
        ),
        "hidden_dim": trial.suggest_categorical("hidden_dim", list(HIDDEN_DIMS)),
        "lr": trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", WEIGHT_DECAY_RANGE[0], WEIGHT_DECAY_RANGE[1], log=True
        ),
        "batch_size": int(batch_size),
    }


def _remaining_trials(study, total_trials):
    done = sum(
        trial.state != optuna.trial.TrialState.WAITING
        for trial in study.trials
    )
    return max(0, total_trials - done)


def _save_study_artifacts(study, best_params):
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(
        OPTUNA_DIR / "lstm_trials.csv", index=False
    )
    (OPTUNA_DIR / "lstm_best_params.json").write_text(
        json.dumps(best_params, indent=2)
    )


def _store_trial_attributes(trial, train_result, model, threshold):
    trial.set_user_attr("best_epoch", train_result["best_epoch"])
    trial.set_user_attr("stopped_epoch", train_result["stopped_epoch"])
    trial.set_user_attr("best_stop_loss", train_result["best_stop_loss"])
    trial.set_user_attr("threshold", threshold)
    trial.set_user_attr(
        "n_parameters",
        int(sum(parameter.numel() for parameter in model.parameters())),
    )


def _train_trial_model(
    trial,
    data,
    params,
    *,
    epochs,
    patience,
    device,
    seed,
    amp,
    verbose,
):
    set_seed(seed + trial.number)
    model = LSTMModel(
        input_dim=data.input_dim,
        hidden_dim=params["hidden_dim"],
        layer_dim=params["layer_dim"],
        output_dim=OUTPUT_DIM,
    )
    result = train_model(
        model,
        data,
        data.rows("fit"),
        data.rows("stop"),
        epochs=epochs,
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        batch_size=params["batch_size"],
        patience=patience,
        device=device,
        phase="fit",
        seed=seed + trial.number,
        amp=amp,
        verbose=verbose,
        log_prefix=f"trial {trial.number} | ",
    )
    return model, result


def _evaluate_trial(model, data, y_tune, device):
    proba_tune = predict_proba(
        model,
        data,
        data.rows("tune"),
        phase="fit",
        batch_size=INFER_BATCH_SIZE,
        device=device,
    )
    tune_loss = log_loss(y_tune, proba_tune, labels=[0, 1])
    return tune_loss, select_threshold(y_tune, proba_tune)


def _run_trial(
    trial,
    data,
    y_tune,
    *,
    epochs,
    patience,
    batch_size,
    device,
    seed,
    amp,
    verbose,
):
    """Train one Optuna trial and return its tune log-loss."""
    params = _suggest_params(trial, batch_size)
    print(
        f"trial {trial.number}: starting | params={params}",
        flush=True,
    )
    model, train_result = _train_trial_model(
        trial,
        data,
        params,
        epochs=epochs,
        patience=patience,
        device=device,
        seed=seed,
        amp=amp,
        verbose=verbose,
    )
    tune_loss, threshold = _evaluate_trial(model, data, y_tune, device)
    _store_trial_attributes(trial, train_result, model, threshold)
    print(
        f"trial {trial.number}: loss={tune_loss:.5f}"
        f" | best_epoch={train_result['best_epoch']}"
        f" | stopped={train_result['stopped_epoch']}"
        f" | params={params}",
        flush=True,
    )
    return tune_loss


def _make_objective(
    data, y_tune, epochs, patience, batch_size, device, seed, amp, verbose
):
    def objective(trial):
        return _run_trial(
            trial,
            data,
            y_tune,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            seed=seed,
            amp=amp,
            verbose=verbose,
        )

    return objective


def _optimize(study, objective, n_trials):
    remaining = _remaining_trials(study, n_trials)
    print(f"Optuna budget={n_trials} | remaining={remaining}", flush=True)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    if not any(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    ):
        raise RuntimeError("The LSTM study has no completed trial.")


def _tune_result(study, batch_size, contract_hash):
    best_params = {**study.best_params, "batch_size": int(batch_size)}
    best_trial = study.best_trial
    return {
        "study": study,
        "best_params": best_params,
        "best_value": float(study.best_value),
        "best_epoch": int(best_trial.user_attrs.get("best_epoch", 0)),
        "stopped_epoch": int(best_trial.user_attrs.get("stopped_epoch", 0)),
        "threshold": float(best_trial.user_attrs.get("threshold", 0.5)),
        "trial_number": int(best_trial.number),
        "contract_hash": contract_hash,
    }


def _print_best(result):
    print(
        f"best trial #{result['trial_number']}: loss={result['best_value']:.5f}"
        f" | threshold={result['threshold']:.2f}"
        f" | best_epoch={result['best_epoch']}"
        f" | params={result['best_params']}",
        flush=True,
    )


def tune_model(
    data,
    *,
    n_trials=N_TRIALS,
    epochs=EPOCHS,
    patience=PATIENCE,
    batch_size=BATCH_SIZE,
    device=None,
    seed=SEED,
    amp=True,
    verbose=False,
):
    """Search LSTM hyper-parameters with Optuna and return the best configuration."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    contract = _contract(data, batch_size)
    study = _get_study(contract, seed, n_trials)
    objective = _make_objective(
        data,
        data.labels("tune"),
        epochs,
        patience,
        batch_size,
        device,
        seed,
        amp,
        verbose,
    )
    _optimize(study, objective, n_trials)

    result = _tune_result(study, batch_size, _contract_hash(contract))
    _save_study_artifacts(study, result["best_params"])
    _print_best(result)
    return result
