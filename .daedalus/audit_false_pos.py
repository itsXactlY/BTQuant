"""False-positive hunt for the hollow guard.

If the guard has a false positive it will reject a strategy that actually
works. The dangerous shapes are assignment forms the regex cannot see:
  setattr(self, "atr", ...)   annotations   self.atr: btind.ATR
  a loop that assigns          a comprehension
Also: does a CLEAN file actually trade? Run 3 of each through the real
backtester on real data. Ground truth beats regex reasoning.
"""
import sys, glob, random, re, time
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.strategy_factory import _find_uninitialised_attrs
from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy

files = sorted(glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py"))
flagged, clean = [], []
for p in files:
    src = open(p, encoding="utf-8", errors="replace").read()
    (flagged if _find_uninitialised_attrs(src) else clean).append(p)

# --- 1. structural false-positive scan on flagged files ---
EXOTIC = re.compile(r"setattr\(\s*self\s*,\s*['\"](\w+)['\"]|self\.(\w+)\s*:|for\s+\w+.*:\s*$", re.M)
exotic_hits = 0
for p in flagged:
    src = open(p, encoding="utf-8", errors="replace").read()
    for m in _find_uninitialised_attrs(src):
        if re.search(rf"setattr\(\s*self\s*,\s*['\"]{m}['\"]", src) or re.search(rf"self\.{m}\s*:", src):
            exotic_hits += 1
            print(f"  FALSE-POSITIVE RISK {p.split('/')[-1][:60]}: {m} via setattr/annotation")
            break
print(f"\n1. exotic-assignment risks among flagged: {exotic_hits}")

# --- 2. ground truth: does a FLAGGED file actually trade? ---
DATA = {"source": "parquet",
        "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
        "initial_cash": 10000.0}

def run(path):
    src = open(path, encoding="utf-8", errors="replace").read()
    m = re.findall(r"^class (\w+)\(", src, re.M)
    if not m:
        return ("noclass", None)
    b = AutomatedBacktester()
    gs = GeneratedStrategy(hypothesis_id="fp", strategy_name="fp", code_path=path,
                           class_name=m[-1], parameters={}, indicators=[],
                           generated_at="")
    t0 = time.time()
    try:
        res = b.run_backtest(gs, dict(DATA))
    except Exception as e:
        return (f"RAISED:{type(e).__name__}", None)
    if res is None:
        return ("None", None)
    d = res.__dict__
    return (f"trades={d['num_trades']:<4} ret={d['total_return']:+.3%}", d.get("num_trades"))

random.seed(7)
print("\n2. FLAGGED files through the real backtester (should now RAISE, not return 0 silently):")
for p in random.sample(flagged, min(4, len(flagged))):
    out, n = run(p)
    print(f"   {out:<28} {p.split('/')[-1][:56]}")

print("\n3. CLEAN files through the real backtester (must NOT be rejected):")
for p in random.sample(clean, min(4, len(clean))):
    out, n = run(p)
    print(f"   {out:<28} {p.split('/')[-1][:56]}")
