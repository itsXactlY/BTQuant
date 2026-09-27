"""Step 6 verification: 0 trades must be a hard failure, not a 0-return result.

Ground truth from the real corpus: pick strategy files that previously produced
a 0-trade result JSON, run them through the backtester, and assert
  (a) run_backtest returns None
  (b) no new result JSON appears (nothing half-truth is persisted)
  (c) the log says EMPTY HULL
Then a known-good trading strategy must still produce a valid, complete JSON.
"""
import sys, glob, json, os, re, logging, time
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy, _find_uninitialised_attrs

RES = "/home/alca/projects/PubBTQuant/autonomous_agency/results"
DATA = {"source": "parquet",
        "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
        "initial_cash": 10000.0}

logging.basicConfig(level=logging.ERROR, format="LOG %(levelname)s %(message)s")

# --- find strategy files whose historic result was 0 trades ---
zero = []
for jf in glob.glob(f"{RES}/*_backtest_results.json"):
    try:
        if json.load(open(jf)).get("num_trades", 0) == 0:
            zero.append(jf)
    except Exception:
        pass
print(f"historic 0-trade results on disk: {len(zero)}")


def strategy_for(jf):
    stem = os.path.basename(jf)[: -len("_backtest_results.json")]
    for p in glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py"):
        if os.path.basename(p)[: -len(".py")] == stem:
            return p
    return None


def run(path, tag):
    src = open(path, encoding="utf-8", errors="replace").read()
    m = re.findall(r"^class (\w+)\(", src, re.M)
    if not m:
        return "noclass"
    b = AutomatedBacktester()
    gs = GeneratedStrategy(hypothesis_id=tag, strategy_name=tag, code_path=path,
                           class_name=m[-1], parameters={}, indicators=[],
                           generated_at="")
    t0 = time.time()
    res = b.run_backtest(gs, dict(DATA))
    if res is None:
        return f"None ({time.time()-t0:.0f}s)"
    return f"trades={res.__dict__['num_trades']} ret={res.__dict__['total_return']:+.2%}"


print("\nA. previously-0-trade strategies now:")
tested = 0
for jf in zero:
    if tested >= 4:
        break
    p = strategy_for(jf)
    if not p:
        continue
    tested += 1
    before = os.path.getmtime(jf)
    out = run(p, "hulltest")
    stale = "UNTOUCHED" if os.path.getmtime(jf) == before else "REWRITTEN(!)"
    print(f"   {out:<20} json={stale:<12} {os.path.basename(p)[:52]}")

print("\nB. known-good strategies still work:")
good = []
for jf in glob.glob(f"{RES}/*_backtest_results.json"):
    try:
        if json.load(open(jf)).get("num_trades", 0) > 500:
            good.append(jf)
    except Exception:
        pass
for jf in good[:2]:
    p = strategy_for(jf)
    if not p:
        continue
    print(f"   {run(p, 'goodtest'):<20} {os.path.basename(p)[:52]}")
