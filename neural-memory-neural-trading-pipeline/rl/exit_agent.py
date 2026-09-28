from __future__ import annotations

import numpy as np


class HeuristicExitAgent:
    """Drop-in RL-agent interface while PPO training is optional."""

    def __init__(self, close_threshold: float = 0.01, max_bars: int = 120) -> None:
        self.close_threshold = close_threshold
        self.max_bars = max_bars

    def act(self, state: dict) -> int:
        if state.get("unrealized_pnl", 0.0) > self.close_threshold:
            return 1  # close
        if state.get("time_in_position", 0) > self.max_bars:
            return 1
        if state.get("sl_prob", 0.0) > 0.6:
            return 2  # tighten
        return 0  # hold


class RandomPolicy:
    def act(self, state: dict) -> int:
        _ = state
        return int(np.random.choice([0, 1, 2]))
