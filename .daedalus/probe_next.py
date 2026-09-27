import importlib.util, sys, polars as pl, backtrader as bt
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
p = "/home/alca/projects/PubBTQuant/autonomous_agency/strategies/ATR_Breakout_20260927_100914.py"
spec = importlib.util.spec_from_file_location("gen_x", p)
mod = importlib.util.module_from_spec(spec); sys.modules["gen_x"] = mod
spec.loader.exec_module(mod)
K = [getattr(mod, n) for n in dir(mod)
     if isinstance(getattr(mod, n), type) and issubclass(getattr(mod, n), bt.Strategy)
     and getattr(mod, n) is not bt.Strategy and "Base" not in n][0]
print("class:", K.__name__, "has next:", "next" in K.__dict__)

calls = {"next": 0, "entry": 0, "exit": 0}
class Probe(K):
    def next(self):
        calls["next"] += 1
        return super().next()
    def buy_or_short_condition(self):
        r = super().buy_or_short_condition()
        if r: calls["entry"] += 1
        return r
    def sell_or_cover_condition(self):
        r = super().sell_or_cover_condition()
        if r: calls["exit"] += 1
        return r

df = pl.read_parquet(".btq_cache/mssql/BTCUSDT_1h.parquet").head(12000)
c = bt.Cerebro(stdstats=False, runonce=False)
c.adddata(bt.feeds.PolarsData(dataname=df)); c.broker.setcash(100_000.0)
c.addstrategy(Probe, percent_sizer=0.95)
s = c.run()[0]
print("bars fed:", len(df), "| next() calls:", calls["next"],
      "| entries:", calls["entry"], "| exits:", calls["exit"])
print("buy_executed:", s.buy_executed, "| len(s):", len(s),
      "| pos:", s.getposition())
