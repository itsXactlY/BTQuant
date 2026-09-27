"""Does an 'empty' cell really see the bars, or does it run over nothing?

An empty hull (0 trades) and a run over 0 bars look identical in the results
file. This distinguishes them.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

import backtrader as bt
import polars as pl

from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy
from autonomous_agency.sweep import class_name_of
from autonomous_agency import mssql_store as ms

path = Path(sys.argv[1])
symbol, tf = sys.argv[2], sys.argv[3]

frame = ms.load(symbol, tf)
if len(sys.argv) > 4:
    frame = frame.tail(int(sys.argv[4]))
print(f"frame rows: {frame.height:,}  {frame['datetime'][0]} .. {frame['datetime'][-1]}")

cls = class_name_of(path)
print("class:", cls)
gs = GeneratedStrategy(hypothesis_id="probe", strategy_name=path.stem,
                       code_path=str(path.resolve()), class_name=cls,
                       parameters={}, indicators=[], generated_at="probe",
                       validation_status="ok", codegen_path="llm")

engine = AutomatedBacktester()
cerebro = bt.Cerebro()
cerebro.broker.setcash(100_000)
cerebro.broker.setcommission(commission=0.001)
feed = bt.feeds.PolarsData(dataname=frame)
cerebro.adddata(feed)
engine._add_analyzers(cerebro)

calls = {"next": 0, "buy": 0, "sell": 0}
kls = engine._load_strategy_class(gs)


class Counting(kls):
    def next(self):
        calls["next"] += 1
        if calls["next"] == 1:
            import inspect
            src = inspect.getsource(type(self).next)
            print("--- next() source ---")
            print(src[:1500])
        return super().next()

    def buy(self, *a, **kw):
        calls["buy"] += 1
        return super().buy(*a, **kw)

    def sell(self, *a, **kw):
        calls["sell"] += 1
        return super().sell(*a, **kw)


cerebro.addstrategy(Counting)
res = cerebro.run()
r = res[0]
print(f"\nnext() calls: {calls['next']:,}   buy(): {calls['buy']}  sell(): {calls['sell']}")
print("feed length after run:", len(cerebro.datas[0]))
print("num trades:", len(r.orders) if hasattr(r, "orders") else "n/a",
      " closed:", len(getattr(r, "trades", [])))
print("final value:", r.broker.getvalue())
an = r.analyzers.trades.get_analysis()
print("trade analyzer:", {k: v.get("total", {}).get("total") for k, v in an.items()
                         if isinstance(v, dict) and "total" in v})
