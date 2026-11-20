import numpy as np
from dataclasses import dataclass
from pathlib import Path

from w2.src.data_processing.constants import TX_CHANNELS, SEQ_CHANNELS, LSTM_CHANNELS, SEQ_DAYS
from w2.src.data_processing.sequences.transactions_sequences import prepare_transaction_days
from w2.src.data_processing.utils import _date, sequence_start

_N_LISTENING = len(SEQ_CHANNELS)
_TX_COL = {name: _N_LISTENING + i for i, name in enumerate(TX_CHANNELS)}  # name -> channel index
_NUMERIC_TX = TX_CHANNELS[:5]  # per-day aggregates, written only on event days
_LAST_STATE_SOURCES = {        # forward-filled channel -> source column in the event table
    "tx_last_is_cancel": "cancel",
    "tx_last_is_auto_renew": "renew",
    "tx_state_ambiguous": "ambiguous",
}
_EVENT_COLS = ["row", "ordinal", "expiry_ordinal", *_NUMERIC_TX, "cancel", "renew", "ambiguous"]
_MAX_DAYS = 3650  # clip for day-count features (~10 years)
 
 
# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
 
def build_lstm_sequences(trans, users, cutoff, listening, output_path=None, chunk_size=4096):
    """Build bounded-memory (users, 90, 20) arrays, optionally directly to .npy."""
    n_users = users.height
    if listening.shape != (n_users, SEQ_DAYS, _N_LISTENING):
        raise ValueError("Listening shape does not match users.")
 
    days, preflight = prepare_transaction_days(trans, users, cutoff) # Here I created the data of transactions
    # Relies on `days` being sorted by (row, date), as prepare_transaction_days guarantees.
    events = {c: days[c].to_numpy() for c in _EVENT_COLS}
    first_day_ordinal = _epoch_day(_date(sequence_start(cutoff)))
 
    result = _allocate_output((n_users, SEQ_DAYS, len(LSTM_CHANNELS)), output_path)
    limited_since = limited_expiry = 0
 
    for start in range(0, n_users, chunk_size):
        end = min(start + chunk_size, n_users)
        block, n_since, n_expiry = _build_block(
            listening[start:end], events, start, end, first_day_ordinal
        )
        result[start:end] = block
        limited_since += n_since
        limited_expiry += n_expiry
 
    if isinstance(result, np.memmap):
        result.flush()
    preflight.update({
        "limited_days_since_last": limited_since,
        "limited_days_to_expire": limited_expiry,
    })
    return result, preflight
 
 
# --------------------------------------------------------------------------
# General helpers
# --------------------------------------------------------------------------
 
def _epoch_day(d):
    return int(np.datetime64(d, "D").astype(np.int64))
 
 
def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))
 
 
def _allocate_output(shape, output_path):
    if output_path is None:
        return np.zeros(shape, dtype=np.float16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=shape)
 
 
# --------------------------------------------------------------------------
# One block of users
# --------------------------------------------------------------------------
 
def _build_block(listening_block, events, block_start, block_end, first_day_ordinal):
    """Combined listening + transaction channels for users [block_start, block_end)."""
    n = block_end - block_start
    block = np.zeros((n, SEQ_DAYS, len(LSTM_CHANNELS)), dtype=np.float32)
    block[:, :, :_N_LISTENING] = listening_block
 
    located = _locate_block_events(events, block_start, block_end, first_day_ordinal)
    _write_event_day_channels(block, events, located)
    n_since, n_expiry = _write_state_channels(block, events, located, first_day_ordinal)
 
    if not np.isfinite(block).all():
        raise ValueError("Non-finite combined sequence values.")
    return block.astype(np.float16), n_since, n_expiry
 
 
@dataclass(frozen=True)
class _BlockEvents:
    records: slice                 # this block's rows in the event table
    user_rows: np.ndarray          # per record: user index relative to the block
    relative_days: np.ndarray      # per record: days since window start (<0 = pre-window)
    latest_index: np.ndarray       # (users, days): event-table index of last known event (0 if none)
    state_known: np.ndarray        # (users, days): bool, any event seen up to that day
 
 
def _locate_block_events(events, block_start, block_end, first_day_ordinal):
    """Find the block's events and forward-fill 'last event so far' for every user-day."""
    first, after_last = np.searchsorted(events["row"], [block_start, block_end])
    records = slice(first, after_last)
    user_rows = events["row"][records].astype(np.int64) - block_start
    relative_days = events["ordinal"][records].astype(np.int64) - first_day_ordinal
 
    # Scratch table: slot 0 holds the pre-window state, day d lives in slot d + 1.
    # Event indices grow with time, so a running max forward-fills the latest event.
    slots = np.full((block_end - block_start, SEQ_DAYS + 1), -1, dtype=np.int64)
    slots[user_rows, np.maximum(relative_days + 1, 0)] = np.arange(first, after_last)
    latest = np.maximum.accumulate(slots, axis=1)[:, 1:]
 
    return _BlockEvents(
        records=records,
        user_rows=user_rows,
        relative_days=relative_days,
        latest_index=np.maximum(latest, 0),
        state_known=latest >= 0,
    )
 
 
def _write_event_day_channels(block, events, located):
    """Signed-log1p of the daily aggregates, only on days that have an event."""
    inside = located.relative_days >= 0
    rows = located.user_rows[inside]
    days = located.relative_days[inside]
    for name in _NUMERIC_TX:
        values = events[name][located.records][inside].astype(np.float64)
        block[rows, days, _TX_COL[name]] = _signed_log1p(values)
 
 
def _write_state_channels(block, events, located, first_day_ordinal):
    """Forward-filled subscription state. Returns (n_clipped_since, n_clipped_expiry)."""
    known = located.state_known
    block[:, :, _TX_COL["tx_state_known"]] = known
 
    if not len(events["row"]):
        return 0, 0  # no transactions at all: state is never known, channels stay zero
 
    idx = located.latest_index
    for channel, source in _LAST_STATE_SOURCES.items():
        block[:, :, _TX_COL[channel]] = np.where(known, events[source][idx], 0)
 
    calendar = first_day_ordinal + np.arange(SEQ_DAYS)[None, :]
    days_since_last = calendar - events["ordinal"][idx]
    expiry = events["expiry_ordinal"][idx]
    expiry_known = known & np.isfinite(expiry)
    days_to_expire = np.where(expiry_known, expiry - calendar, 0)
 
    n_since = int(((days_since_last > _MAX_DAYS) & known).sum())
    n_expiry = int(((np.abs(days_to_expire) > _MAX_DAYS) & expiry_known).sum())
 
    block[:, :, _TX_COL["tx_days_since_last"]] = np.where(
        known, np.log1p(np.clip(days_since_last, 0, _MAX_DAYS)), 0
    )
    block[:, :, _TX_COL["tx_days_to_expire"]] = _signed_log1p(
        np.clip(days_to_expire, -_MAX_DAYS, _MAX_DAYS)
    )
    block[:, :, _TX_COL["tx_expiry_known"]] = expiry_known
    return n_since, n_expiry


def audit_lstm_sequences(seq, listening, chunk_size=4096):
    if seq.shape != (len(listening), SEQ_DAYS, len(LSTM_CHANNELS)):
        raise ValueError("Unexpected combined sequence shape.")
    stats = {"n_users": len(seq), "users_without_transactions": 0,
             "days_with_transactions": 0, "ambiguous_state_days": 0}
    listening_channels = len(SEQ_CHANNELS)
    channel = {name: listening_channels + index for index, name in enumerate(TX_CHANNELS)}
    flags = [channel[name] for name in ("tx_state_known", "tx_last_is_cancel",
                                       "tx_last_is_auto_renew", "tx_expiry_known",
                                       "tx_state_ambiguous")]
    for lo in range(0, len(seq), chunk_size):
        x = np.asarray(seq[lo:lo+chunk_size])
        if not np.isfinite(x).all() or not np.isin(x[:, :, flags], [0, 1]).all():
            raise ValueError("Invalid combined values or flags.")
        if not np.array_equal(x[:, :, :listening_channels], listening[lo:lo+chunk_size]):
            raise ValueError("Combined listening channels do not match the listening sequence.")
        if (x[:, :, [channel[name] for name in ("tx_count", "tx_plan_days_sum",
                                               "tx_cancel_count", "tx_auto_renew_count",
                                               "tx_days_since_last")]] < 0).any():
            raise ValueError("Negative counts or recency.")
        events = x[:, :, channel["tx_count"]] > 0
        known = x[:, :, channel["tx_state_known"]] > 0
        if np.any(x[:, :, listening_channels:channel["tx_state_known"]][~events] != 0) or np.any(x[:, :, channel["tx_last_is_cancel"]:][~known] != 0):
            raise ValueError("Invalid missing-event/state masks.")
        if np.any(x[:, :, channel["tx_days_to_expire"]][x[:, :, channel["tx_expiry_known"]] == 0] != 0):
            raise ValueError("Invalid missing-expiry mask.")
        stats["users_without_transactions"] += int((~known.any(axis=1)).sum())
        stats["days_with_transactions"] += int(events.sum())
        stats["ambiguous_state_days"] += int(x[:, :, channel["tx_state_ambiguous"]].sum())
    return stats
 
