"""Probe: is the REAL BTQuant BaseStrategy importable and runnable?

Read-only: touches no project file. Puts PubBTQuant/dependencies FIRST on
sys.path so the BTQ fork (1.11.0) shadows the stock backtrader in ~/.local,
then tries a 200-bar backtest with a minimal subclass.
"""
import sys
import traceback

FORK = "/home/alca/projects/PubBTQuant/dependencies"
sys.path.insert(0, FORK)

import backtrader as bt  # noqa: E402

print("fork loaded:", bt.__version__, bt.__file__)
print("has backtrader.backtest helper:", hasattr(bt, "backtest"))
print("has feeds.PolarsData         :", hasattr(bt.feeds, "PolarsData"))
print("BrokerBase.get_cash          :", hasattr(bt.broker.BrokerBase, "get_cash"))

# ---- 1) can the real BaseStrategy even be imported? --------------------
try:
    from backtrader.strategies.base import BaseStrategy, OrderTracker
    print("IMPORT OK: BaseStrategy =", BaseStrategy, "| OrderTracker =", OrderTracker)
except Exception:
    print("IMPORT FAIL:")
    traceback.print_exc()
    raise SystemExit(1)

print("base param names:", sorted(BaseStrategy.params._keys())
      if hasattr(BaseStrategy.params, "_keys") else sorted(dir(BaseStrategy.params)))


# ---- 2) can a minimal subclass be constructed + run? -------------------
class Probe(BaseStrategy):
    params = (
        ("percent_sizer", 0.95),
        ("take_profit", 1.0),
        ("fast", 5),
    )

    def __init__(self):
        super().__init__()
        self.sma = bt.ind.SMA(self.data, period=self.p.fast)

    def buy_or_short_condition(self):
        if self.dataclose[0] > self.sma[0]:
            self.create_order(action="BUY")
            return True
        return False

    def sell_or_cover_condition(self):
        if self.dataclose[0] < self.sma[0] and self.buy_executed:
            self.close_order(self.active_orders[0])
            return True
        return False


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

n = 200
idx = pd.date_range("2026-01-01", periods=n, freq="h")
rng = np.random.default_rng(7)
close = 100 + np.cumsum(rng.normal(0, 0.4, n))
cols = {
    "open": close,
    "high": close * 1.002,
    "low": close * 0.998,
    "close": close,
    "volume": np.ones(n),
}
df = pd.DataFrame(cols, index=idx).rename_axis("datetime").reset_index()


def make_feed():
    """The fork's data path is polars-native."""
    import polars as pl
    from backtrader.feeds.polarfeed import PolarsData

    pdf = pl.DataFrame(
        {
            # PolarsData is POSITIONAL: datetime, open, high, low, close, volume
            "datetime": idx,
            "open": cols["open"], "high": cols["high"],
            "low": cols["low"], "close": cols["close"],
            "volume": cols["volume"],
        }
    )
    return PolarsData(dataname=pdf), "POLARS (fork-native bt.feeds.PolarsData)"


feed, feed_kind = make_feed()
print("feed used:", feed_kind)

cerebro = bt.Cerebro(stdstats=False)
cerebro.broker.setcash(1000.0)
cerebro.adddata(feed)
cerebro.addstrategy(Probe)
try:
    res = cerebro.run()
    st = res[0]
    print("RUN OK  final_value =", round(cerebro.broker.getvalue(), 2),
          "| trades =", st.total_trades, "| wins =", st.total_wins,
          "| legs =", len(st.active_orders))
except Exception:
    print("RUN FAIL:")
    traceback.print_exc()
