
import sys, time, traceback, importlib.util, os
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
import pandas as pd, numpy as np
import backtrader as bt
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy

DATA = {
    "source": "parquet",
    "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
    "initial_cash": 10_000.0,
}
S = "/home/alca/projects/PubBTQuant/autonomous_agency/strategies/"

def load_class(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, name)

def probe(tag, path):
    t0 = time.time()
    cls = load_class(path)
    gs = GeneratedStrategy(hypothesis_id="probe", strategy_name=tag, code_path=path,
                           class_name=cls.__name__, parameters={}, indicators=[], generated_at="")
    b = AutomatedBacktester()
    res = b.run_backtest(gs, dict(DATA))
    if res is None:
        print(f"== {tag}: run_backtest returned None (see log above)")
        return
    d = res.__dict__ if hasattr(res, "__dict__") else {}
    keys = ("strategy_name","total_return","sharpe_ratio","max_drawdown","num_trades","total_trades","win_rate","profit_factor")
    print("==", tag)
    for k in keys:
        if k in d:
            v = d[k]
            print(f"   {k} = {v if not isinstance(v,list) else len(v)}")
    print("   all fields:", sorted(d.keys()))
    print("   %.1fs" % (time.time()-t0))

probe("concept", S+"Persistent_Homology_Microstructure_Topology_with_Wasserstein_Mean_Reversion_and_Ergodic_Entropic_Allocation_20260623_090036.py")
probe("v9",      S+"Wavelet_Scattering_Signature_Momentum_with_Conformal_Risk_Scaling_WSSM_CRS_20260618_143344.py")
