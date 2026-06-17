from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WalkForwardConfig:
    train_months: int = 12
    val_months: int = 2
    test_months: int = 2
    step_months: int = 1
    days_per_month: int = 30


def create_walk_forward_splits(n_samples: int, cfg: WalkForwardConfig) -> list[dict[str, tuple[int, int]]]:
    total_days = cfg.train_months + cfg.val_months + cfg.test_months
    window_days = total_days * cfg.days_per_month
    step_days = cfg.step_months * cfg.days_per_month
    splits: list[dict[str, tuple[int, int]]] = []

    for offset in range(0, max(0, n_samples - window_days) + 1, step_days):
        train_start = offset
        train_end = train_start + cfg.train_months * cfg.days_per_month
        val_start = train_end
        val_end = val_start + cfg.val_months * cfg.days_per_month
        test_start = val_end
        test_end = test_start + cfg.test_months * cfg.days_per_month
        splits.append(
            {
                "train": (train_start, train_end),
                "val": (val_start, val_end),
                "test": (test_start, test_end),
            }
        )
    return splits


def slice_split(features: np.ndarray, prices: np.ndarray, split: tuple[int, int]) -> dict[str, np.ndarray]:
    s, e = split
    return {"features": features[s:e], "prices": prices[s:e]}
