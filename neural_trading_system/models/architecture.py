#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based Neural Trading Model — Self-Aware Upgrade
Adds 3 new classification heads:
- take_profit_head
- stop_loss_head
- hold_head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TradingTransformer(nn.Module):
    def __init__(self, feature_dim, cfg):
        super().__init__()
        d_model = cfg.get("d_model", 256)
        num_heads = cfg.get("num_heads", 8)
        num_layers = cfg.get("num_layers", 6)
        d_ff = cfg.get("d_ff", 1024)
        dropout = cfg.get("dropout", 0.15)
        self.seq_len = cfg.get("seq_len", 100)

        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.01)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # === Multi-Task Heads ===
        self.entry_head = nn.Linear(d_model, 1)
        self.exit_head = nn.Linear(d_model, 1)
        self.return_head = nn.Linear(d_model, 1)
        self.volatility_head = nn.Linear(d_model, 1)
        self.position_head = nn.Linear(d_model, 1)

        # === New Heads ===
        self.take_profit_head = nn.Linear(d_model, 1)
        self.stop_loss_head = nn.Linear(d_model, 1)
        self.hold_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x) + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)

        return {
            "entry_prob": torch.sigmoid(self.entry_head(pooled)),
            "exit_prob": torch.sigmoid(self.exit_head(pooled)),
            "expected_return": self.return_head(pooled),
            "volatility_forecast": self.volatility_head(pooled),
            "position_size": torch.sigmoid(self.position_head(pooled)),
            "take_profit_prob": torch.sigmoid(self.take_profit_head(pooled)),
            "stop_loss_prob": torch.sigmoid(self.stop_loss_head(pooled)),
            "hold_prob": torch.sigmoid(self.hold_head(pooled)),
        }


def create_model(feature_dim, cfg):
    return TradingTransformer(feature_dim, cfg)
