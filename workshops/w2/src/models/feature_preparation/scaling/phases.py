"""Phase-specific scaler states and persistence."""

import json
from pathlib import Path

import numpy as np

from ...constants import BLOCK_ROWS, QUANTILE_SAMPLE, SEED
from .combined import fit_sequence_preprocessor
from .transform import transform_sequence


class PhasePreprocessor:
    """Fit and cache one sequence state per phase and transform rows with it."""

    def __init__(
        self,
        view,
        channel_specs,
        phase_rows,
        quantile_sample=QUANTILE_SAMPLE,
        block_rows=BLOCK_ROWS,
        seed=SEED,
    ):
        self.view = view
        self.channel_specs = channel_specs
        self.phase_rows = phase_rows
        self.quantile_sample = quantile_sample
        self.block_rows = block_rows
        self.seed = seed
        self._cache = {}

    def state(self, phase):
        """Return the fitted state for ``phase``, fitting it on first use."""
        if phase not in self._cache:
            self._cache[phase] = fit_sequence_preprocessor(
                self.view,
                self.phase_rows[phase],
                self.channel_specs,
                self.quantile_sample,
                self.block_rows,
                self.seed,
            )
        return self._cache[phase]

    def batch(self, rows, phase):
        """Transform raw rows using the phase state."""
        return transform_sequence(
            self.view[np.asarray(rows)], self.state(phase), self.channel_specs
        )


def save_preprocessing(state, path):
    """Write one fitted phase state as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return path


def load_preprocessing(path):
    """Read a phase state previously written by :func:`save_preprocessing`."""
    return json.loads(Path(path).read_text())
