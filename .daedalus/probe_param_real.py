"""Settle it on the real thing: does `self.<param>` work in BaseStrategy?

Injects a print into a real corpus strategy and runs the real backtester on
the real parquet. If `self.X` on a declared param raises, then the guard
flagging `refit_interval_bars` is CORRECT, not a false positive.
"""
import sys, re, glob, time
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy

LOG = "/home/alca/projects/PubBTQuant/.daedalus/param_probe_result.txt"

# a strategy that is CLEAN (so it runs to completion) and declares stop_loss
cands = [p for p in sorted(glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py"))
         if "self.stop_loss" not in open(p, encoding="utf-8", errors="replace").read()]
src_path = None
for p in cands:
    s = open(p, encoding="utf-8", errors="replace").read()
    if "class " in s and "BaseStrategy" in s:
        src_path = p
        break
print("using:", src_path.split("/")[-1][:70])

src = open(src_path, encoding="utf-8", errors="replace").read()
probe = '''
        with open("%s", "a") as _pf:
            try:
                _pf.write("self.p.stop_loss = %%r\\n" %% (self.p.stop_loss,))
            except Exception as e:
                _pf.write("self.p.stop_loss ERR %%s\\n" %% e)
            try:
                _pf.write("self.stop_loss  = %%r\\n" %% (self.stop_loss,))
            except Exception as e:
                _pf.write("self.stop_loss  ERR %%s\\n" %% e)
''' % LOG
# inject right after the first super().__init__() in the generated class
m = re.search(r"^(\s*)super\(\)\.__init__\(.*?\)\s*$", src, re.M)
src2 = src[:m.end()] + "\n" + probe + src[m.end():]
tmp = "/home/alca/projects/PubBTQuant/.daedalus/probe_param_strategy.py"
open(tmp, "w").write(src2)

import ast; ast.parse(src2)
open(LOG, "w").close()

cls = re.findall(r"^class (\w+)\(", src2, re.M)[-1]
b = AutomatedBacktester()
gs = GeneratedStrategy(hypothesis_id="pp", strategy_name="pp", code_path=tmp,
                       class_name=cls, parameters={}, indicators=[], generated_at="")
DATA = {"source": "parquet",
        "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
        "initial_cash": 10000.0}
t0 = time.time()
res = b.run_backtest(gs, dict(DATA))
print(f"ran in {time.time()-t0:.1f}s ->", "None" if res is None else
      f"trades={res.__dict__['num_trades']} ret={res.__dict__['total_return']:+.2%}")
print("--- probe output ---")
print(open(LOG).read())
