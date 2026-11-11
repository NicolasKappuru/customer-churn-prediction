"""Combined listening + transaction scaler state."""

from ...constants import BLOCK_ROWS, QUANTILE_SAMPLE, SEED
from .listening import fit_listening_scaler
from .transaction import fit_transaction_scaler


def fit_sequence_preprocessor(
    view,
    rows,
    channel_specs,
    quantile_sample=QUANTILE_SAMPLE,
    block_rows=BLOCK_ROWS,
    seed=SEED,
):
    """Fit the listening and transaction scalers for one phase."""
    listening = fit_listening_scaler(
        view, rows, channel_specs, quantile_sample, block_rows, seed
    )
    transaction = fit_transaction_scaler(
        view, rows, channel_specs, quantile_sample, block_rows, seed
    )
    return {
        "listening": listening,
        "transaction": transaction,
        "fit_rows": int(len(rows)),
        "quantile_sample": int(quantile_sample),
        "block_rows": int(block_rows),
        "seed": int(seed),
    }
