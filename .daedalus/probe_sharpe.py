"""Why is sharpe_ratio 0.0 in all 1628 result files?

Returns works (rtot non-zero), so data flows. Is the analyzer itself
returning 0, or is our extraction dropping it? Print the raw analysis.
"""
import sys, glob, json
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
import backtrader as bt
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy
import re

# a strategy that historically traded
jf = None
for j in glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/results/*_backtest_results.json"):
    try:
        if json.load(open(j)).get("num_trades", 0) > 500:
            jf = j
            break
    except Exception:
        pass
stem = jf.split("/")[-1][: -len("_backtest_results.json")]
path = [p for p in glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py")
        if p.split("/")[-1][:-3] == stem][0]
print("probe strategy:", stem[:60])

DATA = {"source": "parquet",
        "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
        "initial_cash": 10000.0}
b = AutomatedBacktester()
gs = GeneratedStrategy(hypothesis_id="sp", strategy_name="sharpeprobe", code_path=path,
                       class_name=re.findall(r"^class (\w+)\(",
                       open(path, encoding="utf-8", errors="replace").read(), re.M)[-1],
                       parameters={}, indicators=[], generated_at="")

# rebuild the cerebro ourselves so we can see raw analyzer dicts
cerebro = b._setup_cerebro(gs, dict(DATA))
b._add_analyzers(cerebro)
results = cerebro.run()
r = results[0]
for name in ("sharpe", "returns", "drawdown", "sqn", "vwr"):
    a = getattr(r.analyzers, name, None)
    if a is None:
        print(f"{name:9} ABSENT")
        continue
    d = a.get_analysis()
    print(f"{name:9} {json.dumps(d, default=str)[:200]}")
