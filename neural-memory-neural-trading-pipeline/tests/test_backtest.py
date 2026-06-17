import numpy as np

from neural_trading_system.backtesting.vectorized_engine import VectorizedBacktest


class StubModel:
    def predict(self, seq):
        _ = seq
        return {
            "entry_prob": 0.8,
            "tp_prob": 0.55,
            "sl_prob": 0.1,
            "volatility": 0.02,
            "expected_return": 0.01,
        }


def test_backtest_generates_trades():
    prices = np.linspace(100, 110, 400)
    feats = np.random.default_rng(0).normal(size=(400, 15))
    pnl, trades = VectorizedBacktest(StubModel()).run(feats, prices)
    assert len(trades) > 0
    assert pnl.shape == prices.shape
