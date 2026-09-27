"""What bar-duration info is available on the feed/result after a run?

Needed to annualize the sharpe fallback without guessing.
"""
import sys, glob, json, re
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
import backtrader as bt
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy

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

DATA = {"source": "parquet",
        "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
        "initial_cash": 10000.0}
b = AutomatedBacktester()
src = open(path, encoding="utf-8", errors="replace").read()
gs = GeneratedStrategy(hypothesis_id="sp", strategy_name="tfprobe", code_path=path,
                       class_name=re.findall(r"^class (\w+)\(", src, re.M)[-1],
                       parameters={}, indicators=[], generated_at="")
cerebro = b._setup_cerebro(gs, dict(DATA))
b._add_analyzers(cerebro)
results = cerebro.run()
r = results[0]
d = r.datas[0]
print("feed attrs:", [a for a in dir(d) if "frame" in a or "compress" in a or "from" in a or "period" in a])
for a in ("_timeframe", "_compression", "fromdate", "todate", "_fromdate", "_todate", "_barstatus"):
    print(f"  {a} = {getattr(d, a, 'ABSENT')}")
print("n bars =", len(d))
print("period =", getattr(d, "period", "ABSENT"))
print("data.datetime first/last:", d.datetime.array[0], d.datetime.array[-1])
st = r.strategy
print("strategy.data is d:", st.data is d, "| bars =", len(st.data))
import backtrader.utils.date2num as d2n
print("TimeFrame attrs:", [a for a in dir(bt.TimeFrame) if not a.startswith("_")][:14])
print("TimeFrame.Days =", bt.TimeFrame.Days, "Weeks =", bt.TimeFrame.Weeks)
print("sharpe analyzer p:", r.analyzers.sharpe.get_params())
