"""Why does every order get margin-rejected at percent_sizer=0.95?

The generation smoke test runs with percent_sizer=0.0, so it only proves an
order was *called*. In a real run with percent_sizer=0.95 every single order
comes back status 4 (Margin) and nothing trades. This isolates which of the two
is at fault: the position sizing, the broker setup, or the base's create_order.

Usage: why_margin.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

import polars as pl

import backtrader as bt
from backtrader.strategies.base import BaseStrategy

CACHE = Path("/home/alca/projects/PubBTQuant/.btq_cache/mssql")


class Reporter(BaseStrategy):
    """Buy once, then print everything the broker said about it."""
    params = (("percent_sizer", 0.95),)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent = False
        self.log = []

    def next(self):
        if self.sent or len(self) < 20:
            return
        self.sent = True
        cash = self.broker.getcash()
        price = float(self.data.close[0])
        stake = getattr(self, "stake", "<unset>")
        self.log.append(f"cash={cash:.2f} close={price:.4f} self.stake={stake}")
        o = self.create_order(action="BUY")
        self.log.append(f"create_order -> {o!r}")
        if o is not None:
            for attr in ("size", "executed", "comm", "status", "price",
                         "created", "accepted", "margin", "transmit"):
                self.log.append(f"   order.{attr} = {getattr(o, attr, '<none>')}")
        self.log.append(f"   broker.getcash() after = {self.broker.getcash():.2f}")
        self.log.append(f"   broker.getvalue()      = {self.broker.getvalue():.2f}")

    def notify_order(self, order):
        self.log.append(f"NOTIFY status={order.status} executed={order.executed} "
                        f"size={order.size} isbuy={order.isbuy()} price={order.price}")

    def stop(self):
        print("\n".join("  " + ln for ln in self.log))


def run(cash, sizer, levered=False, commission=0.0):
    df = pl.read_parquet(CACHE / "BTCUSDT_1h.parquet").head(400)
    cerebro = bt.Cerebro(stdstats=False, runonce=False)
    cerebro.adddata(bt.feeds.PolarsData(dataname=df))
    cerebro.broker.setcash(cash)
    if levered:
        cerebro.broker.set_leverage(1.0)
    cerebro.addstrategy(Reporter, percent_sizer=sizer, backtest=True)
    if commission:
        cerebro.broker.addcommission(bt.CommInfoBase(commission=commission))
    print(f"\n--- cash={cash} percent_sizer={sizer} "
          f"leverage={cerebro.broker.get_leverage()} commission={commission}")
    cerebro.run()


def main():
    import inspect
    src = inspect.getsource(BaseStrategy.create_order)
    print("=== BaseStrategy.create_order source ===")
    for i, ln in enumerate(src.split("\n")[:45], 1):
        print(f"  {i:3d} {ln}")
    run(100_000.0, 0.95)
    run(100_000.0, 0.5)
    run(100_000.0, 0.05)
    run(100_000.0, 0.95, commission=0.001)
    run(1_000_000.0, 0.95)


if __name__ == "__main__":
    main()
