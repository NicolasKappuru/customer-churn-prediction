"""PyTorch access to the prepared sequence rows.

The dataset keeps the original memory map and the fitted phase state; each
example is read and transformed on demand, so the full tensor is never copied
into RAM. Channel slices make the same object usable for the listening-only
(0:8), transaction-only (8:20) and combined (0:20) models.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """Read raw sequence rows, apply a phase scaler and return tensors."""

    def __init__(self, rows, view, labels, preprocessor, phase, channel_slice=None):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.view = view
        self.labels = np.asarray(labels, dtype=np.float32)
        self.preprocessor = preprocessor
        self.phase = phase
        self.channel_slice = channel_slice
        if len(self.rows) != len(self.labels):
            raise ValueError("rows and labels must have the same length.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        sequence = self.preprocessor.batch([row], self.phase)[0]
        if self.channel_slice is not None:
            sequence = sequence[:, self.channel_slice]
        return (
            torch.from_numpy(np.ascontiguousarray(sequence)),
            torch.tensor(self.labels[index], dtype=torch.float32),
            row,
        )


def make_loader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
):
    """Build a standard PyTorch loader for a :class:`SequenceDataset`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
