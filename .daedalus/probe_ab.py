
import sys, time, os, re
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy
DATA = {"source":"parquet","path":"/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet","initial_cash":10000.0}
def clsname(path):
    src = open(path).read()
    return re.findall(r"^class (\w+)\(", src, re.M)[-1]
def probe(tag, path):
    b = AutomatedBacktester()
    gs = GeneratedStrategy(hypothesis_id="probe", strategy_name=tag, code_path=path, class_name=clsname(path),
                           parameters={}, indicators=[], generated_at="")
    t0=time.time(); res = b.run_backtest(gs, dict(DATA))
    if res is None: print(f"{tag}: None"); return
    d = res.__dict__
    print("{:<12} trades={:<6} return={:+.4%}  sharpe={:.3f}  maxDD={:.3f}  win={:.2f}  ({:.1f}s)".format(
        tag, d["num_trades"], d["total_return"], d["sharpe_ratio"], d["max_drawdown"], d["win_rate"], time.time()-t0))
S = "/home/alca/projects/PubBTQuant/autonomous_agency/strategies/"
probe("BEFORE(AND)", S+"Persistent_Homology_Microstructure_Topology_with_Wasserstein_Mean_Reversion_and_Ergodic_Entropic_Allocation_20260623_090036.py")
probe("AFTER(vote)", "/home/alca/projects/PubBTQuant/.daedalus/probe_vote_strategy.py")
