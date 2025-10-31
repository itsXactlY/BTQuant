
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class V1FixedTradingModel(nn.Module):
    def __init__(self, n_features: int, seq_len: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.entry_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.tp_head = nn.Sequential(nn.Linear(d_model + 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.sl_head = nn.Sequential(nn.Linear(d_model + 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.other_heads = nn.Linear(d_model, 7)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):

        x = self.embed(x)
        x = self.pos(x)
        x = self.transformer(x)
        pooled = x.mean(1)

        entry = self.entry_head(pooled).squeeze(-1)

        if context is not None:
            ctx_pooled = torch.cat([pooled, context], dim=1)
            tp = self.tp_head(ctx_pooled).squeeze(-1)
            sl = self.sl_head(ctx_pooled).squeeze(-1)
        else:
            tp = sl = torch.zeros(pooled.size(0), device=pooled.device)

        others = self.other_heads(pooled)

        return {
            'entry': entry, 'tp': tp, 'sl': sl,
            'others': others
        }
