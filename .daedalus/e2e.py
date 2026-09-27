"""End-to-end: real StrategyFactory -> real AutomatedBacktester -> real parquet."""
import json
import logging
import os
import sys
from pathlib import Path

PARQUET_PATH = (
    "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet"
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from autonomous_agency.ai_interface import StrategyHypothesis  # noqa: E402
from autonomous_agency.strategy_factory import StrategyFactory  # noqa: E402
from autonomous_agency.backtester import AutomatedBacktester  # noqa: E402

PARQUET = PARQUET_PATH
print("parquet:", PARQUET)

HYPOTHESES = [
    ("EMA Crossover Momentum with RSI Confirmation",
     "Fast/slow EMA cross confirmed by RSI regime", ["ema", "rsi"]),
    ("Bollinger Band Mean Reversion on Volatility Compression",
     "Fade the edge of a squeeze when volatility expands", ["rsi", "atr"]),
    ("RSI Momentum Trend Continuation",
     "Buy strength confirmed by trend filter", ["rsi", "sma"]),
    ("ATR Channel Breakout with Volume Surge",
     "Volatility expansion with participation", ["atr", "sma"]),
    ("MACD Momentum Reversal Strategy",
     "Histogram flip inside an established trend", ["rsi", "sma"]),
    ("Dual Moving Average Trend Follower with Volatility Filter",
     "Classic trend following gated on calm volatility", ["sma", "atr"]),
]

fails = 0
paths = {}
fingerprints = {}
for i, (name, desc, inds) in enumerate(HYPOTHESES):
    h = StrategyHypothesis(
        id=f"e2e-{i:03d}",
        name=name,
        description=desc,
        indicators=inds,
        entry_conditions=["condition1"],
        exit_conditions=["condition2"],
        parameters={"fast": 10, "slow": 30},
        rationale="end-to-end verification of the factory fix",
        mathematical_beauty_score=0.5,
        expected_regime="trending",
        risk_profile="moderate",
    )
    f = StrategyFactory()
    strat = f.generate_strategy(h)
    if strat is None:
        print(f"[{i}] GENERATE FAILED: {name}")
        fails += 1
        continue
    # provenance must be stamped on EVERY strategy, and never be a template
    path = getattr(strat, "codegen_path", "MISSING")
    paths[path] = paths.get(path, 0) + 1
    assert path != "MISSING", f"{name}: no codegen provenance stamped"
    assert path != "template", f"{name}: template hull reached the backtester"
    assert strat.is_real, f"{name}: is_real False but path={path}"
    bt = AutomatedBacktester()
    r = bt.run_backtest(
        strat, {"source": "parquet", "path": PARQUET, "symbol": "BTC"}
    )
    if r is None:
        print(f"[{i}] BACKTEST REJECTED (no result written): {name} "
              f"[path={path}]")
        continue
    print(
        f"[{i}] {r.strategy_name[:40]:40s} path={path:8s} trades={r.num_trades:5d} "
        f"ret={r.total_return:+.4f} sharpe={r.sharpe_ratio:+.3f} "
        f"dd={r.max_drawdown:.3f} win={r.win_rate:.3f}"
    )
    fp = (r.num_trades, round(r.total_return, 6), round(r.sharpe_ratio, 6))
    fingerprints.setdefault(fp, []).append(name)
    result_file = (
        bt.results_dir / f"{r.strategy_name}_backtest_results.json"
    )
    assert result_file.exists(), f"no result JSON at {result_file}"
    with open(result_file) as fh:
        assert json.load(fh)["sharpe_ratio"] == r.sharpe_ratio, "JSON != object"
    assert r.num_trades > 0, "0 trades in a stored result"
    assert r.sharpe_ratio != 0.0, "sharpe laundered to zero"

print("codegen path distribution:", paths)
dupes = {k: v for k, v in fingerprints.items() if len(v) > 1}
print("generate failures:", fails)
print("identical-result groups:", dupes if dupes else "none")
