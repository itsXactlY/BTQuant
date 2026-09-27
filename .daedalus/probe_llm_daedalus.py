"""E2E: die Agency erzeugt per LLM eine Strategie auf der echten BaseStrategy.

Treibt den LLM-Pfad von strategy_factory mit BTQ_LLM_BACKEND=daedalus_cli:
Hypothese -> Spec -> Code (daedalus CLI) -> Lint -> Datei -> Loader -> Backtest
auf dem Fork mit PolarsData. Prueft, dass wirklich getradet wird.

    BTQ_LLM_BACKEND=daedalus_cli ~/.btq/bin/python .daedalus/probe_llm_daedalus.py
"""
import os
import sys
import contextlib
import io
import logging
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("BTQ_LLM_BACKEND", "daedalus_cli")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import backtrader as bt          # noqa: E402
import polars as pl              # noqa: E402

from autonomous_agency.ai_interface import StrategyHypothesis   # noqa: E402
from autonomous_agency.strategy_factory import StrategyFactory  # noqa: E402
from autonomous_agency.strategy_loader import StrategyClassLoader                # noqa: E402

assert bt.__file__.startswith(str(Path.home() / ".btq")), f"nicht das Venv: {bt.__file__}"
assert hasattr(bt.feeds, "PolarsData"), "kein PolarsData -> nicht der Fork"

HYP = StrategyHypothesis(
    id="probe-daedalus-1",
    name="RSI Dip Reversion Probe",
    description=("Buy oversold RSI dips below 35, exit when RSI recovers above 60. "
                 "percent_sizer 0.95."),
    indicators=["rsi"],
    entry_conditions=["CrossOver(self.rsi, 35.0) > 0"],
    exit_conditions=["self.rsi[0] > 60"],
    parameters={"rsi_period": 14, "rsi_oversold": 35.0, "rsi_overbought": 60.0},
    rationale="Mean reversion after short-term oversold extremes.",
    mathematical_beauty_score=0.6,
    expected_regime="ranging",
    risk_profile="moderate",
)


def synthetic(n=700):
    rows, px = [], 100.0
    for i in range(n):
        if i < 120:
            px -= 0.55
        elif i < 280:
            px += 1.4 if (i % 10) < 5 else -1.3
        else:
            px += 2.0 if ((i - 280) // 11) % 2 == 0 else -1.7
        rows.append({"datetime": datetime(2024, 1, 1) + timedelta(minutes=i),
                     "open": px, "high": px + 0.7, "low": px - 0.7,
                     "close": px + 0.2, "volume": 10.0 + (i % 7)})
    return pl.DataFrame(rows)


def main() -> int:
    factory = StrategyFactory()
    print("PROBE  codegen (LLM via daedalus CLI) …", flush=True)
    gen = factory.generate_strategy(HYP)
    if gen is None:
        print("ERGEBNIS FAIL: generate_strategy() lieferte None")
        return 1

    path = Path(gen.code_path)
    src = path.read_text()
    print(f"PROBE  datei     : {path.name} ({len(src)} bytes)")
    print(f"PROBE  klasse    : {gen.class_name}")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("from backtrader") or s.startswith("class "):
            print(f"PROBE  {s}")

    is_base = "backtrader.strategies.base" in src
    print(f"PROBE  erbt echte BaseStrategy: {is_base}")
    if not is_base:
        print("ERGEBNIS FAIL: nicht von backtrader.strategies.base importiert")
        return 1

    loader = StrategyClassLoader()
    cls = loader.load_class(gen)
    bases = [b.__module__ + "." + b.__name__ for b in cls.__mro__[1:4]]
    print(f"PROBE  MRO       : {' -> '.join(bases)}")

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PolarsData(dataname=synthetic()))
    cerebro.broker.setcash(1000.0)
    cerebro.addstrategy(cls, percent_sizer=0.95)
    with contextlib.redirect_stdout(io.StringIO()):
        res = cerebro.run()

    st = res[0]
    filled = len(getattr(st, "active_orders", []) or [])
    broker = cerebro.broker
    print(f"PROBE  orders     : {filled}")
    print(f"PROBE  final value: {broker.getvalue():.2f} (start 1000.00)")
    print("ERGEBNIS OK" if broker.getvalue() > 0 else "ERGEBNIS FAIL: value <= 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
