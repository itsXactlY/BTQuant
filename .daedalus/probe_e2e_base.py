"""E2E: every agency codegen path must emit a strategy that RUNS on the
BTQuant backtrader fork and actually trades.

Run from /home/alca/projects/PubBTQuant:
    /usr/bin/python3 .daedalus/probe_e2e_base.py
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dependencies"))
sys.path.insert(0, str(ROOT))

import backtrader as bt
import polars as pl

from autonomous_agency.strategy_factory import StrategyFactory
from autonomous_agency import strategy_template

N = 700
rows = []
px = 100.0
for i in range(N):
    # Phase 1: sharp decline (drives RSI to 0, slow EMA down)
    # Phase 2: tight chop (slow EMA flat -> the reversion filter must pass)
    # Phase 3: recovery zig-zag (momentum concept must fire)
    if i < 120:
        px -= 0.55
    elif i < 280:
        # deep enough swings to push RSI under 35, small net drift so the
        # slow EMA is flat — that is the window a reversion entry needs
        px += 1.4 if (i % 10) < 5 else -1.3
    else:
        px += 2.0 if ((i - 280) // 11) % 2 == 0 else -1.7
    rows.append({
        "datetime": datetime(2024, 1, 1) + timedelta(minutes=i),
        "open": px, "high": px + 0.7, "low": px - 0.7,
        "close": px + 0.2, "volume": 10.0 + (i % 7),
    })
FEED = pl.DataFrame(rows)

SPEC = {
    "strategy_name": "E2E_Momentum_RSI",
    "hypothesis_id": "probe-1",
    "description": "EMA cross with RSI filter",
    "indicators": ["ema", "rsi", "atr"],
    "entry_conditions": ["self.ema_fast[0] > self.ema_slow[0]", "self.rsi[0] < 70"],
    "exit_conditions": ["self.ema_fast[0] < self.ema_slow[0]"],
    "parameters": {"rsi_period": 14, "ema_period": 21},
    "risk_management": {"stop_loss": 2.0},
}


def run(label, code):
    ns = {}
    try:
        exec(compile(code, f"<{label}>", "exec"), ns)
    except Exception as e:
        print(f"{label:22s} IMPORT-FAIL  {type(e).__name__}: {e}")
        return False
    import backtrader.strategies.base as bmod
    cls = None
    for name, obj in ns.items():
        # skip the imported base itself — it is in the namespace too
        if (isinstance(obj, type) and issubclass(obj, bmod.BaseStrategy)
                and obj is not bmod.BaseStrategy):
            cls = obj
            break
    if cls is None:
        print(f"{label:22s} NO-BASESTRATEGY-SUBCLASS")
        return False

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PolarsData(dataname=FEED), name="probe")
    cerebro.broker.setcash(1000.0)
    cerebro.addstrategy(cls, print_params=False, quantstats=False)
    import contextlib, io
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            results = cerebro.run(runonce=False)
    except Exception as e:
        print(f"{label:22s} RUN-FAIL     {type(e).__name__}: {e}")
        print("   last output:", buf.getvalue()[-400:].replace("\n", " | "))
        return False
    strat = results[0]
    print(f"{label:22s} OK  class={cls.__name__:20s} "
          f"filled_orders={getattr(strat, 'total_trades', 0)} "
          f"wins={getattr(strat, 'total_wins', 0)} "
          f"open_legs={len(getattr(strat, 'active_orders', []) or [])} "
          f"final_value={cerebro.broker.getvalue():.2f}")
    return getattr(strat, "total_trades", 0) > 0


f = StrategyFactory()
paths = {
    "template-fallback": f._generate_template_code_from_spec(SPEC),
    "concept-driven": f._generate_concept_driven_code(SPEC),
    "strategy_template.py": strategy_template.generate_full_strategy(
        "E2E_Wavelet_Momentum", description="probe"),
}
ok = {k: run(k, v) for k, v in paths.items()}
print("\nRESULT:", "ALL OK" if all(ok.values()) else f"FAILURES {[k for k,v in ok.items() if not v]}")
sys.exit(0 if all(ok.values()) else 1)
