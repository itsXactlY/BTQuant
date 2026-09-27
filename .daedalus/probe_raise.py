
import sys, os, re, traceback
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
import backtrader as bt, pandas as pd
from autonomous_agency.strategy_factory import GeneratedStrategy

def clsname(p): return re.findall(r"^class (\w+)\(", open(p).read(), re.M)[-1]

def run(path, tag):
    import importlib.util
    mod = "autonomous_agency.strategies._probe_" + clsname(path)[:20]
    spec = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    cls = getattr(m, clsname(path))
    # unwrap the bare except: patch the method to NOT swallow
    orig_next = cls.buy_or_short_condition
    errors = []
    def loud(self):
        try:
            return orig_next(self)
        except Exception:
            errors.append(traceback.format_exc())
            raise
    cls.buy_or_short_condition = loud

    df = pd.read_parquet("/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet")
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, unit="us")
    feed = bt.feeds.PandasData(dataname=df)
    cerebro = bt.Cerebro(); cerebro.adddata(feed)
    cerebro.broker.setcash(10000.0); cerebro.broker.setcommission(commission=0.001)
    cerebro.addstrategy(cls)
    cerebro.run()
    print("==", tag)
    if errors:
        print("   entry() raised on bar 1, %d times. FIRST TRACEBACK:" % len(errors))
        print("   " + errors[0].strip().replace("\n", "\n   "))
    else:
        print("   no exception from entry()")

run("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/Persistent_Homology_Microstructure_Topology_with_Wasserstein_Mean_Reversion_and_Ergodic_Entropic_Allocation_20260623_090036.py", "BEFORE(AND)")
run("/home/alca/projects/PubBTQuant/.daedalus/probe_vote_strategy.py", "AFTER(vote)")
