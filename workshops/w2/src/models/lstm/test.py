"""Final refit and test inference for the w2 LSTM."""

import torch

from .config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    INFER_BATCH_SIZE,
    OUTPUT_DIM,
    SEED,
)
from .model import LSTMModel
from .train import fit_fixed_epochs, predict_proba, resolve_device, set_seed


def _build_model(best_params, input_dim):
    return LSTMModel(
        input_dim=input_dim,
        hidden_dim=int(best_params["hidden_dim"]),
        layer_dim=int(best_params["layer_dim"]),
        output_dim=OUTPUT_DIM,
    )


def _fit_final_model(
    model, data, best_params, n_epochs, batch_size, device, seed, amp, verbose
):
    return fit_fixed_epochs(
        model,
        data,
        data.rows("development"),
        epochs=n_epochs,
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        batch_size=int(best_params.get("batch_size", batch_size)),
        device=device,
        phase="development",
        seed=seed,
        amp=amp,
        verbose=verbose,
        log_prefix="final | ",
    )


def _save_checkpoint(model, best_params, threshold, n_epochs):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / "lstm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": best_params,
            "threshold": float(threshold),
            "epochs": n_epochs,
        },
        path,
    )
    return path


def _test_result(model, fit, data, proba_test, threshold, n_epochs, device):
    return {
        "model": model,
        "history": fit["history"],
        "y_test": data.labels("test"),
        "proba_test": proba_test,
        "pred_test": (proba_test >= float(threshold)).astype(int),
        "threshold": float(threshold),
        "epochs": n_epochs,
        "device": str(device),
    }


def _fit_and_predict(
    model,
    data,
    best_params,
    n_epochs,
    batch_size,
    device,
    seed,
    amp,
    verbose,
):
    fit = _fit_final_model(
        model,
        data,
        best_params,
        n_epochs,
        batch_size,
        device,
        seed,
        amp,
        verbose,
    )
    proba = predict_proba(
        model,
        data,
        data.rows("test"),
        phase="development",
        batch_size=INFER_BATCH_SIZE,
        device=device,
    )
    return fit, proba


def test_model(
    best_params,
    data,
    *,
    threshold=0.5,
    best_epoch=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=None,
    seed=SEED,
    amp=True,
    verbose=False,
    save_checkpoint=True,
):
    """Refit on development, score test once and return probabilities."""
    device = resolve_device(device)
    set_seed(seed)
    model = _build_model(best_params, data.input_dim)
    n_epochs = int(best_epoch) if best_epoch else int(epochs)

    fit, proba_test = _fit_and_predict(
        model,
        data,
        best_params,
        n_epochs,
        batch_size,
        device,
        seed,
        amp,
        verbose,
    )
    result = _test_result(
        model, fit, data, proba_test, threshold, n_epochs, device
    )

    if save_checkpoint:
        result["checkpoint"] = _save_checkpoint(
            model, best_params, threshold, n_epochs
        )
    return result
