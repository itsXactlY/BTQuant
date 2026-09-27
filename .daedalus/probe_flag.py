import importlib.util, sys, polars as pl, backtrader as bt
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
p = "/home/alca/projects/PubBTQuant/autonomous_agency/strategies/Bollinger_Reversion_20260927_101714.py"
spec = importlib.util.spec_from_file_location("gen_b", p)
mod = importlib.util.module_from_spec(spec); sys.modules["gen_b"] = mod
spec.loader.exec_module(mod)
K = [getattr(mod, n) for n in dir(mod)
     if isinstance(getattr(mod, n), type) and issubclass(getattr(mod, n), bt.Strategy)
     and getattr(mod, n) is not bt.Strategy and "Base" not in n][0]

c = {"entry_sig": 0, "exit_sig": 0, "entered": 0, "exited": 0}
class Probe(K):
    def buy_or_short_condition(self):
        # how often the RAW signal is true, ignoring the in_position gate
        close, lb = self.data.close, self.bollinger.bot
        if (close[0] > lb[0] and close[-1] <= lb[-1]):
            c["entry_sig"] += 1
        return super().buy_or_short_condition()
    def sell_or_cover_condition(self):
        close, mb = self.data.close, self.bollinger.mid
        if self.in_position and close[0] > mb[0] and close[-1] <= mb[-1]:
            c["exit_sig"] += 1
        r = super().sell_or_cover_condition()
        if r: c["exited"] += 1
        return r
    def create_order(self, *a, **kw):
        c["entered"] += 1
        return super().create_order(*a, **kw)

df = pl.read_parquet(".btq_cache/mssql/BTCUSDT_1h.parquet").head(12000)
ce = bt.Cerebro(stdstats=False, runonce=False)
ce.adddata(bt.feeds.PolarsData(dataname=df)); ce.broker.setcash(100_000.0)
ce.addstrategy(Probe, percent_sizer=0.95)
s = ce.run()[0]
print("bars:", len(df), "| raw entry signals:", c["entry_sig"],
      "| raw exit signals:", c["exit_sig"])
print("orders placed:", c["entered"], "| exits taken:", c["exited"])
print("self.in_position at end:", s.in_position,
      "| base buy_executed:", s.buy_executed,
      "| active_orders:", len(s.active_orders))
