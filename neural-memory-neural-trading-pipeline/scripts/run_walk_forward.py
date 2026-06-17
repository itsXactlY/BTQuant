from __future__ import annotations

import numpy as np

from neural_trading_system.backtesting.metrics import compute_metrics
from neural_trading_system.backtesting.vectorized_engine import VectorizedBacktest


class DemoModel:
    def predict(self, seq):
        ret = float((seq[-1, 0] - seq[0, 0]) / (abs(seq[0, 0]) + 1e-6))
        return {
            "entry_prob": 0.7 if ret > 0 else 0.4,
            "tp_prob": 0.6 if ret > 0.01 else 0.2,
            "sl_prob": 0.6 if ret < -0.01 else 0.2,
            "volatility": 0.02,
            "expected_return": ret,
        }


def main() -> None:
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 0.2, size=5000))
    features = np.column_stack([prices, rng.normal(size=(5000, 14))])
    pnl, trades = VectorizedBacktest(DemoModel()).run(features, prices)
    print(compute_metrics(pnl, trades))


if __name__ == "__main__":
    main()
