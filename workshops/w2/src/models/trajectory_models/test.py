"""Final refit on all of February and one prediction of March."""

import time

import torch

from .config import (
    EPOCHS,
    HORIZON,
    INPUT_DAYS,
    SEED,
    TARGET_NAMES,
    model_dirs,
)
from .model import build_model
from .train import fit_fixed_epochs, predict_trajectories, resolve_device, set_seed


def _save_checkpoint(model, model_name, best_params, loss_weights, n_epochs):
    """Everything needed to rebuild the model outside this notebook (e.g. wfinal)."""
    checkpoint_dir = model_dirs(model_name)["checkpoints"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{model_name}_traj.pt"
    torch.save(
        {
            "model_name": model_name,
            "task": "trajectory",
            "state_dict": model.state_dict(),
            "params": best_params,
            "loss_weights": loss_weights,
            "epochs": n_epochs,
            "input_days": INPUT_DAYS,
            "horizon": HORIZON,
            "targets": list(TARGET_NAMES),
            "scaler_phase": "development",
        },
        path,
    )
    return path


def test_model(
    tune_result,
    data,
    *,
    loss_weights,
    epochs=EPOCHS,
    device=None,
    seed=SEED,
    amp=True,
    verbose=False,
    save_checkpoint=True,
):
    """Refit on ``development`` for the tuned ``best_epoch`` and predict ``test`` once."""
    model_name = tune_result["model_name"]
    params = tune_result["best_params"]
    device = resolve_device(device)
    set_seed(seed)
    model = build_model(model_name, params, data.input_dim)
    n_epochs = int(tune_result.get("best_epoch") or epochs)

    start = time.perf_counter()
    fit = fit_fixed_epochs(
        model, data, data.rows("development"), loss_weights=loss_weights,
        epochs=n_epochs, lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]), device=device,
        phase="development", seed=seed, amp=amp, verbose=verbose,
        log_prefix=f"{model_name} final | ",
    )
    train_seconds = time.perf_counter() - start

    pred = predict_trajectories(model, data, data.rows("test"),
                                phase="development", device=device)
    result = {
        "model_name": model_name,
        "model": model,
        "history": fit["history"],
        "pred": pred,
        "epochs": n_epochs,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / max(n_epochs, 1),
        "device": str(device),
    }
    if save_checkpoint:
        result["checkpoint"] = _save_checkpoint(
            model, model_name, params, loss_weights, n_epochs
        )
    return result
