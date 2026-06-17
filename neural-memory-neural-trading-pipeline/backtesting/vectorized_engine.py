from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestConfig:
    seq_len: int = 100
    entry_threshold: float = 0.6
    tp_threshold: float = 0.5
    sl_threshold: float = 0.5
    timeout_bars: int = 500
    commission: float = 0.001
    slippage: float = 0.0005


class VectorizedBacktest:
    def __init__(self, model, rl_agent=None, config: BacktestConfig | None = None) -> None:
        self.model = model
        self.rl_agent = rl_agent
        self.config = config or BacktestConfig()

    def run(self, features: np.ndarray, prices: np.ndarray):
        n = len(prices)
        pnl = np.zeros(n, dtype=np.float64)
        position = False
        entry_price = 0.0
        entry_time = -1
        size = 0.0
        trades: list[dict] = []

        for i in range(self.config.seq_len, n):
            seq = features[i - self.config.seq_len : i]
            pred = self.model.predict(seq)

            if not position and pred["entry_prob"] > self.config.entry_threshold:
                vol = max(pred.get("volatility", 0.01), 1e-6)
                expected = pred.get("expected_return", 0.0)
                size = float(np.clip(expected / (vol**2), 0.1, 0.5))
                position = True
                entry_price = float(prices[i])
                entry_time = i
                trades.append({"entry_time": i, "entry_price": entry_price, "size": size})
                continue

            if position:
                unrealized = (prices[i] - entry_price) / max(entry_price, 1e-12)
                held = i - entry_time

                if self.rl_agent is not None:
                    action = self.rl_agent.act({"unrealized_pnl": unrealized, "time_in_position": held, **pred})
                    exit_signal = action == 1
                else:
                    exit_signal = bool(
                        pred.get("tp_prob", 0.0) > self.config.tp_threshold
                        or pred.get("sl_prob", 0.0) > self.config.sl_threshold
                    )

                if held > self.config.timeout_bars:
                    exit_signal = True

                if exit_signal:
                    exit_price = float(prices[i])
                    gross = (exit_price - entry_price) / entry_price
                    net = gross - self.config.commission - self.config.slippage
                    pnl[i] = net * size
                    trades[-1].update(
                        {
                            "exit_time": i,
                            "exit_price": exit_price,
                            "gross_pnl": gross,
                            "net_pnl": net,
                            "bars_held": held,
                        }
                    )
                    position = False

        return pnl, trades
