"""Small helpers to look at one user's sequence as a table (for the notebook)."""

import numpy as np
import pandas as pd

from w2.src.data_processing.constants import SEQ_DAYS
from w2.src.data_processing.utils import sequence_start


def day_frame(seq, channels, cutoff, days=None, decimals=3):
    """Return a ``(90, C)`` user sequence as a DataFrame with ``day`` and ``date``.

    ``days`` selects day indices (0..89); by default all 90 days are returned.
    """
    values = np.asarray(seq, dtype=np.float32)
    if values.shape != (SEQ_DAYS, len(channels)):
        raise ValueError(f"Expected shape {(SEQ_DAYS, len(channels))}, got {values.shape}.")
    frame = pd.DataFrame(values, columns=channels).round(decimals)
    dates = pd.date_range(sequence_start(cutoff), periods=SEQ_DAYS).date
    frame.insert(0, "date", dates)
    frame.insert(0, "day", np.arange(SEQ_DAYS))
    return frame if days is None else frame.iloc[list(days)]



# --------------------------------------------------------------------------
# Tensor-slice figure
# --------------------------------------------------------------------------

_GROUP_STYLE = {   # group -> (label, note, fill colour)
    "listening": ("LISTENING", "7 log1p + active flag", "#5F5E5A"),
    "event": ("EVENTS", "payment days", "#3E6176"),
    "state": ("STATE", "carried forward", "#8A5A3B"),
}


def tensor_slice_svg(seq, channel_specs, days, caption="", cell=22, gap=10):
    """SVG of one user's ``(90, C)`` slice: one column per channel, one row per day in ``days``.

    Each cell is shaded by the value relative to the channel's maximum over the
    shown days; zero cells are outlined only. A ``...`` row is drawn where days
    are skipped. Returns the SVG as a string (``IPython.display.SVG`` shows it).
    """
    from html import escape

    values = np.asarray(seq, dtype=np.float32)
    names = [item["name"] for item in channel_specs]
    groups = [item["group"] for item in channel_specs]
    days = sorted(days)
    left, top, label_h = 64, 34, 92
    width = left + len(names) * cell + 70   # room for the last rotated name
    rows = []                       # (y, day or None for an ellipsis row)
    y = top + label_h
    for k, day in enumerate(days):
        if k and day != days[k - 1] + 1:
            rows.append((y, None)); y += cell * 0.8
        rows.append((y, day)); y += cell
    height = y + gap + (18 if caption else 0)
    scale = np.abs(values[days]).max(axis=0)
    scale[scale == 0] = 1.0

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height:.0f}" '
           f'viewBox="0 0 {width} {height:.0f}" font-family="Arial, Helvetica, sans-serif">',
           f'<rect width="{width}" height="{height:.0f}" fill="white"/>']
    for group, (label, note, colour) in _GROUP_STYLE.items():
        cols = [i for i, g in enumerate(groups) if g == group]
        if not cols:
            continue
        x0, x1 = left + cols[0] * cell, left + (cols[-1] + 1) * cell - 2
        out += [f'<line x1="{x0}" y1="8" x2="{x1}" y2="8" stroke="{colour}" stroke-width="1.5"/>',
                f'<text x="{x0}" y="20" font-size="9" font-weight="700" fill="{colour}">{label} ({len(cols)})</text>',
                f'<text x="{x0}" y="30" font-size="7" fill="#6B6861">{escape(note)}</text>']
    for i, name in enumerate(names):
        x = left + i * cell + cell / 2 + 3
        out.append(f'<text transform="translate({x:.1f},{top + label_h - 4}) rotate(-60)" '
                   f'font-size="7.5" fill="#333">{escape(name)}</text>')
    for y, day in rows:
        if day is None:
            out.append(f'<text x="{left - 8}" y="{y + cell * 0.55:.1f}" font-size="9" '
                       f'text-anchor="end" fill="#888">...</text>')
            continue
        out.append(f'<text x="{left - 8}" y="{y + cell * 0.65:.1f}" font-size="8" '
                   f'text-anchor="end" fill="#333">day {day}</text>')
        for i, group in enumerate(groups):
            colour = _GROUP_STYLE[group][2]
            v = values[day, i]
            x = left + i * cell
            if v == 0:
                out.append(f'<rect x="{x + 1}" y="{y + 1:.1f}" width="{cell - 2}" height="{cell - 2}" '
                           f'fill="white" stroke="#C9C6BE" stroke-width="0.8"/>')
            else:
                alpha = 0.25 + 0.75 * min(abs(v) / scale[i], 1.0)
                out.append(f'<rect x="{x + 1}" y="{y + 1:.1f}" width="{cell - 2}" height="{cell - 2}" '
                           f'fill="{colour}" fill-opacity="{alpha:.2f}"/>')
    if caption:
        out.append(f'<text x="{left}" y="{height - 8:.0f}" font-size="8" fill="#555">{escape(caption)}</text>')
    out.append("</svg>")
    return "\n".join(out)
