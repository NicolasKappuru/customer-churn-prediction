import torch
import torch.nn as nn

class RecurrentClassifier(nn.Module):
    """Recurrent encoder + BatchNorm + linear head. Returns logits."""

    rnn_cls: type[nn.Module]  # lo define cada subclase

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.rnn = self.rnn_cls(
            input_dim, hidden_dim, layer_dim, batch_first=True
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)  # estado inicial en ceros por defecto
        return self.fc(self.bn(out[:, -1, :]))


class LSTMModel(RecurrentClassifier):
    rnn_cls = nn.LSTM


class GRUModel(RecurrentClassifier):
    rnn_cls = nn.GRU