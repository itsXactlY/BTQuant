from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, feature_dim: int = 15, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Linear(hidden_dim, 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.head(x[:, -1, :])
        return {
            "entry_logits": x[:, 0:1],
            "tp_logits": x[:, 1:2],
            "sl_logits": x[:, 2:3],
            "expected_return": torch.tanh(x[:, 3:4]),
            "volatility": torch.nn.functional.softplus(x[:, 4:5]),
            "exit_timing": torch.nn.functional.softplus(x[:, 5:6]),
        }
