"""Customer segment filter for the w2 split.

The segment is read from the subscription-state channels of the tensor on the
last day of the window (the cutoff), so it uses the same information as the
model and needs no extra export:

- ``manual``: ``tx_state_known = 1``, ``tx_last_is_cancel = 0`` and
  ``tx_last_is_auto_renew = 0``; the last transaction up to the cutoff is a
  payment without auto-renew.

February and March rows are filtered with the same rule, each at its own cutoff.
"""

import numpy as np

from ..constants import BLOCK_ROWS, SEGMENTS


def _last_day_channels(view, rows, columns, block_rows=BLOCK_ROWS * 8):
    """Read ``columns`` of the last day for ``rows`` in bounded blocks."""
    blocks = [
        np.asarray(view[rows[start:start + block_rows]][:, -1, :])[:, columns]
        for start in range(0, len(rows), block_rows)
    ]
    return np.concatenate(blocks).astype(np.float32)


def segment_mask(view, rows, channels, segment):
    """Boolean mask over ``rows``: True for customers in ``segment``."""
    if segment not in SEGMENTS:
        raise ValueError(f"Unknown segment {segment!r}; expected one of {SEGMENTS}.")
    rows = np.asarray(rows, dtype=np.int64)
    if segment == "all":
        return np.ones(len(rows), dtype=bool)

    names = ["tx_state_known", "tx_last_is_cancel", "tx_last_is_auto_renew"]
    last = _last_day_channels(view, rows, [channels.index(name) for name in names])
    known, cancel, auto_renew = (last[:, i] > 0 for i in range(3))
    return known & ~cancel & ~auto_renew


def filter_split(split, dev_rows, view, channels, segment):
    """Keep only the rows of ``segment`` in every part and in ``dev_rows``.

    The hash assignment is made on the whole cohort first, so a customer is in
    the same part (fit / stop / tune) with or without the filter.
    """
    if segment == "all":
        return split, dev_rows
    filtered = {
        part: rows[segment_mask(view, rows, channels, segment)]
        for part, rows in split.items()
    }
    dev_rows = dev_rows[segment_mask(view, dev_rows, channels, segment)]
    return filtered, dev_rows
