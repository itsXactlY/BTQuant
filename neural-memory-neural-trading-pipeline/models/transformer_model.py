from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NeuralTradingModelV3(nn.Module):
    def __init__(
        self,
        feature_dim: int = 15,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoding = SinusoidalPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        def cls_head(out_dim: int = 1):
            return nn.Sequential(
                nn.Linear(d_model, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim),
            )

        self.entry_head = cls_head()
        self.tp_head = cls_head()
        self.sl_head = cls_head()
        self.return_head = nn.Sequential(cls_head(), nn.Tanh())
        self.vol_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())
        self.timing_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return {
            "entry_logits": self.entry_head(x),
            "tp_logits": self.tp_head(x),
            "sl_logits": self.sl_head(x),
            "expected_return": self.return_head(x),
            "volatility": self.vol_head(x),
            "exit_timing": self.timing_head(x),
        }
