"""Bisect: is the margin rejection caused by BaseStrategy or by the feed?

Every order from every generated strategy ends Submitted(1) -> Accepted(2) ->
Margin(4), even at percent_sizer=0.05 where the notional is 5% of cash. This
runs the same buy three ways -- plain bt.Strategy, BaseStrategy, and both with
a synthetic frame instead of the MSSQL parquet -- to find which layer refuses.

Usage: bisect_margin.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

import numpy as np
import polars as pl

import backtrader as bt
from backtrader.strategies.base import BaseStrategy

CACHE = Path("/home/alca/projects/PubBTQuant/.btq_cache/mssql")
STATUS = {0: "Created", 1: "Submitted", 2: "Accepted", 3: "Completed",
          4: "Margin", 5: "Expired", 6: "Rejected", 7: "Canceled",
          8: "Partial"}


def synth(n=200):
    px = 100.0
    rows = []
    for i in range(n):
        px *= 1.0 + 0.002 * np.sin(i / 9.0)
        rows.append({"datetime": 1609459200 + i * 3600, "open": px,
                     "high": px * 1.002, "low": px * 0.998, "close": px,
                     "volume": 100.0})
    return pl.DataFrame(rows)


def real(n=200):
    return pl.read_parquet(CACHE / "BTCUSDT_1h.parquet").head(n)


def ex(order):
    """`order.executed` is an OrderData object in this fork, not a number."""
    e = getattr(order, "executed", None)
    if e is None:
        return "0"
    size = getattr(e, "size", None)
    price = getattr(e, "price", None)
    if size is None and price is None:
        return "0"
    return f"size={size if size is not None else 0.0:.6f}"


class Plain(bt.Strategy):
    """A bare backtrader strategy: no BaseStrategy in the way at all."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent = False
        self.ev = []

    def next(self):
        if self.sent or len(self) < 30:
            return
        self.sent = True
        size = (self.broker.getcash() * 0.5) / self.data.close[0]
        self.ev.append(f"bare: cash={self.broker.getcash():.2f} "
                       f"value={self.broker.getvalue():.2f} size={size:.6f} "
                       f"notional={size * self.data.close[0]:.2f}")
        o = self.buy(size=size, exectype=bt.Order.Market)
        self.ev.append(f"bare: sent {o}")

    def notify_order(self, order):
        self.ev.append(f"bare: status={STATUS.get(order.status, order.status)} "
                       f"executed[{ex(order)}] "
                       f"cash={self.broker.getcash():.2f} "
                       f"value={self.broker.getvalue():.2f}")

    def stop(self):
        print("\n".join("    " + e for e in self.ev))


class Based(BaseStrategy):
    def next(self):
        if getattr(self, "_sent", False) or len(self) < 30:
            return
        self._sent = True
        ev = getattr(self, "_ev", None)
        if ev is None:
            ev = self._ev = []
        size = (self.broker.getcash() * 0.5) / self.data.close[0]
        ev.append(f"base: cash={self.broker.getcash():.2f} "
                  f"get_cash={self.broker.get_cash():.2f} "
                  f"value={self.broker.getvalue():.2f} size={size:.6f} "
                  f"notional={size * self.data.close[0]:.2f}")
        self.create_order(action="BUY", size=size)
        ev.append(f"base: cash after submit={self.broker.getcash():.2f} "
                  f"value={self.broker.getvalue():.2f}")

    def notify_order(self, order):
        ev = getattr(self, "_ev", None)
        if ev is None:
            return
        ev.append(f"base: status={STATUS.get(order.status, order.status)} "
                  f"executed[{ex(order)}] "
                  f"cash={self.broker.getcash():.2f} "
                  f"value={self.broker.getvalue():.2f}")

    def stop(self):
        print("\n".join("    " + e for e in getattr(self, "_ev", [])))


def case(label, klass, df, cash, **kw):
    print(f"\n--- {label}")
    cerebro = bt.Cerebro(stdstats=False, runonce=False)
    cerebro.adddata(bt.feeds.PolarsData(dataname=df))
    cerebro.broker.setcash(cash)
    cerebro.addstrategy(klass, **kw)
    try:
        cerebro.run()
    except Exception as e:  # noqa: BLE001
        print(f"    RAISED {type(e).__name__}: {str(e)[:120]}")


def main():
    df_s, df_r = synth(), real()
    print(f"synthetic close[30] = {df_s['close'][30]:.2f}")
    print(f"real      close[30] = {df_r['close'][30]:.2f}")
    case("plain bt.Strategy, synthetic frame, 100k", Plain, df_s, 100_000.0)
    case("plain bt.Strategy, REAL parquet,      100k", Plain, df_r, 100_000.0)
    case("BaseStrategy,       synthetic frame, 100k", Based, df_s, 100_000.0,
         percent_sizer=0.5, backtest=True)
    case("BaseStrategy,       REAL parquet,      100k", Based, df_r, 100_000.0,
         percent_sizer=0.5, backtest=True)
    case("BaseStrategy,       REAL parquet,        1", Based, df_r, 1.0,
         percent_sizer=0.5, backtest=True)


if __name__ == "__main__":
    main()
