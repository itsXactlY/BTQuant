"""End-to-end: does run_backtest now report a real sharpe_ratio?

Picks a historically busy strategy and one known-degenerate one, runs both
through the public API, prints the metrics the evaluator would gate on.
"""
import sys, glob, json, re
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy
from autonomous_agency.config import config

DATA = {"source": "parquet",
        "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
        "initial_cash": 10000.0}

busy = None
for j in sorted(glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/results/*_backtest_results.json")):
    try:
        d = json.load(open(j))
        if d.get("num_trades", 0) > 500:
            busy = j
            break
    except Exception:
        pass
stem = busy.split("/")[-1][: -len("_backtest_results.json")]
path = [p for p in glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py")
        if p.split("/")[-1][:-3] == stem][0]
print("strategy:", stem[:60], "| gate min_sharpe =", config.min_sharpe_ratio)

src = open(path, encoding="utf-8", errors="replace").read()
gs = GeneratedStrategy(hypothesis_id="probe", strategy_name="sharpeprobe", code_path=path,
                       class_name=re.findall(r"^class (\w+)\(", src, re.M)[-1],
                       parameters={}, indicators=[], generated_at="")
res = AutomatedBacktester().run_backtest(gs, dict(DATA))
if res is None:
    print("RESULT: None  <-- still broken")
    sys.exit(1)
print(f"  sharpe_ratio = {res.sharpe_ratio!r}   <-- was 0.0 in every stored file")
print(f"  total_return = {res.total_return}  max_drawdown = {res.max_drawdown}")
print(f"  num_trades   = {res.num_trades}  win_rate = {res.win_rate}")
print(f"  passes sharpe gate: {res.sharpe_ratio >= config.min_sharpe_ratio}")
