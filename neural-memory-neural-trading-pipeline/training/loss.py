from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = target * p + (1 - target) * (1 - p)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


class MultiTaskLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(6))

    def forward(self, pred: dict[str, torch.Tensor], tgt: dict[str, torch.Tensor]) -> torch.Tensor:
        losses = torch.stack(
            [
                focal_loss(pred["entry_logits"], tgt["entry_label"]),
                focal_loss(pred["tp_logits"], tgt["tp_label"]),
                focal_loss(pred["sl_logits"], tgt["sl_label"]),
                F.smooth_l1_loss(pred["expected_return"], tgt["expected_return"]),
                F.mse_loss(pred["volatility"], tgt["volatility"]),
                F.smooth_l1_loss(pred["exit_timing"], tgt["optimal_exit_bars"]),
            ]
        )
        precision = torch.exp(-self.log_vars)
        return torch.sum(precision * losses + self.log_vars)
