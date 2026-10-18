"""LSTM classifier for the 90-day churn sequences."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Multi-layer LSTM over the 20 sequence channels.

    ``layer_dim`` is the number of stacked LSTM layers (2-4 in the search).
    The LSTM uses its internal ``tanh`` nonlinearity. Batch normalisation is
    applied to the last hidden state before the linear classifier. The model
    returns logits; use :func:`torch.sigmoid` to obtain churn probabilities.
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(
                self.layer_dim,
                x.size(0),
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
            c0 = torch.zeros_like(h0)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        last = self.bn(out[:, -1, :])
        logits = self.fc(last)
        return logits, hn, cn
