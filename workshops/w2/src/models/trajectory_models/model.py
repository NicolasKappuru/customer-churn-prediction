"""Recurrent encoder + direct multi-horizon head."""

import torch.nn as nn

from .config import HORIZON, N_TARGETS


class TrajectoryModel(nn.Module):
    """Encode days 0..82 and emit all 7 future days at once.

    Same encoder as the classifier (stacked LSTM/GRU, last hidden state,
    BatchNorm); only the head changes, from ``Linear(H, 1)`` to
    ``Linear(H, 7 * 11)`` reshaped to ``(B, 7, 11)``. Binary targets are
    returned as logits, amounts as raw (scaled) values.
    """

    rnn_cls: type[nn.Module]

    def __init__(self, input_dim, hidden_dim, layer_dim,
                 horizon=HORIZON, n_targets=N_TARGETS):
        super().__init__()
        self.horizon = horizon
        self.n_targets = n_targets
        self.rnn = self.rnn_cls(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, horizon * n_targets)

    def forward(self, x):
        out, _ = self.rnn(x)
        z = self.fc(self.bn(out[:, -1, :]))
        return z.view(-1, self.horizon, self.n_targets)


class LSTMTrajectory(TrajectoryModel):
    rnn_cls = nn.LSTM


class GRUTrajectory(TrajectoryModel):
    rnn_cls = nn.GRU


MODELS = {"lstm": LSTMTrajectory, "gru": GRUTrajectory}


def build_model(model_name, params, input_dim):
    """Instantiate a family from tuned hyper-parameters."""
    return MODELS[model_name.lower()](
        input_dim=input_dim,
        hidden_dim=int(params["hidden_dim"]),
        layer_dim=int(params["layer_dim"]),
    )
