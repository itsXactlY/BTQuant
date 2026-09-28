from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EnvConfig:
    stop_loss: float = -0.01


class TradingEnv:
    def __init__(self, prices: np.ndarray, neural_signals: np.ndarray, config: EnvConfig | None = None) -> None:
        self.prices = prices
        self.neural_signals = neural_signals
        self.cfg = config or EnvConfig()
        self.reset()

    def reset(self):
        self.idx = 1
        self.entry = self.prices[0]
        self.done = False
        self.time_in_position = 0
        return self._state()

    def _state(self):
        return {
            "price": float(self.prices[self.idx - 1]),
            "signal": float(self.neural_signals[self.idx - 1]),
            "time_in_position": self.time_in_position,
            "unrealized_pnl": (self.prices[self.idx - 1] - self.entry) / self.entry,
        }

    def step(self, action: int):
        if self.done:
            return self._state(), 0.0, True, {}

        reward = 0.0
        self.time_in_position += 1
        pnl = (self.prices[self.idx] - self.entry) / self.entry
        if action == 1 or pnl <= self.cfg.stop_loss or self.idx == len(self.prices) - 1:
            reward = pnl - 0.1 * self.time_in_position / 50.0
            self.done = True

        self.idx = min(self.idx + 1, len(self.prices) - 1)
        return self._state(), float(reward), self.done, {}
