from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TradingDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: dict[str, np.ndarray],
        seq_len: int = 100,
    ) -> None:
        self.features = features.astype(np.float32)
        self.labels = labels
        self.seq_len = seq_len
        self.length = len(features) - seq_len

    def __len__(self) -> int:
        return max(0, self.length)

    def __getitem__(self, idx: int):
        i = idx + self.seq_len
        x = torch.from_numpy(self.features[i - self.seq_len : i])
        y = {
            "entry_label": torch.tensor([float(self.labels["expected_return"][i] > 0)], dtype=torch.float32),
            "tp_label": torch.tensor([self.labels["tp_label"][i]], dtype=torch.float32),
            "sl_label": torch.tensor([self.labels["sl_label"][i]], dtype=torch.float32),
            "expected_return": torch.tensor([self.labels["expected_return"][i]], dtype=torch.float32),
            "volatility": torch.tensor([self.labels["volatility"][i]], dtype=torch.float32),
            "optimal_exit_bars": torch.tensor([self.labels["optimal_exit_bars"][i]], dtype=torch.float32),
        }
        return x, y
