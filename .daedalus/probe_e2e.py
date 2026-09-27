"""Step 7: the decisive end-to-end test.

Real factory (full dispatch) -> real backtester -> real parquet.
Input: a Hurst/entropy hypothesis, i.e. the exact family that produced 1027
hulls via the never-instantiated self.atr.

Pass condition: strategy generated AND trades > 0 AND a complete result JSON.
"""
import sys, json, glob, os, logging
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.strategy_factory import StrategyFactory
from autonomous_agency.ai_interface import StrategyHypothesis
from autonomous_agency.backtester import AutomatedBacktester

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s",
                    stream=sys.stdout)

DATA = {"source": "parquet",
        "path": "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet",
        "initial_cash": 10000.0}

# The dead strategy from earlier, verbatim in shape.
h = StrategyHypothesis(
    id="e2e-hurst-1",
    name="Hurst Regime Adaptive Entropy Momentum Strategy with Fractal Memory",
    description="Hurst exponent regime detection with entropy momentum and fractal memory gating.",
    indicators=["hurst", "entropy", "fractal", "rsi"],
    entry_conditions=["close > self.ema_sma[0]", "self.rsi[0] < 70"],
    exit_conditions=["close < self.ema_sma[0]", "self.rsi[0] > 30"],
    parameters={"stop_loss": 0.02, "take_profit": 0.04},
    rationale="test",
    mathematical_beauty_score=0.8,
    expected_regime="trending",
    risk_profile="moderate",
)

f = StrategyFactory()
before = set(glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py"))
s = f.generate_strategy(h)
if s is None:
    print("\nE2E FAIL: factory returned None")
    sys.exit(1)
new = set(glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py")) - before
print(f"\nE2E generated: {s.strategy_name}")
print(f"   new files on disk: {[os.path.basename(x) for x in new]}")

src = open(s.code_path, encoding="utf-8", errors="replace").read()
print(f"   reads self.atr?   {'self.atr' in src}")
print(f"   no-op/return stub? {'return  # no-op' in src}")

b = AutomatedBacktester()
res = b.run_backtest(s, dict(DATA))
if res is None:
    print("\nE2E FAIL: backtester returned None (empty hull or crash)")
    sys.exit(1)

d = res.__dict__
jf = f"{b.results_dir}/{s.strategy_name}_backtest_results.json"
raw = open(jf).read()
parsed = json.loads(raw)  # raises if truncated
print(f"\nE2E BACKTEST: trades={d['num_trades']}  return={d['total_return']:+.2%}  "
      f"sharpe={d['sharpe_ratio']}  dd={d['max_drawdown']:.2%}")
print(f"E2E JSON:     {len(raw)} bytes, {len(parsed)} keys, valid=True")

ok = d["num_trades"] > 0 and d["total_return"] != 0.0 and len(parsed) == 22
print(f"\nE2E {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
