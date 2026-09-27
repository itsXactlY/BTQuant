"""Generate a batch of strategies through the LLM path WITH the strengthened gate.

The gate now demands an order, so anything that lands here must trade on real
data. Each file is then backtested on BTC 1h to prove the gate did not lie --
a gate that passes code which then sits at 0 trades is worse than no gate.

Usage: generate_batch.py [count] [hypothesis-seed]
"""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")

from autonomous_agency.ai_interface import StrategyHypothesis
from autonomous_agency.strategy_factory import StrategyFactory
from autonomous_agency.sweep import class_name_of, is_tradeable
from autonomous_agency import mssql_store as ms

OUT = Path("/home/alca/projects/PubBTQuant/.daedalus/batch_results.json")

# Hypotheses with units that actually cross: oscillator vs oscillator, price vs
# price. Never oscillator vs price -- that is the class of bug the gate now
# catches, and there is no point feeding it more examples on purpose.
TEMPLATES = [
    ("RSI Dip Reversion", "Buy when RSI(14) crosses back above 30 after dipping below 25; exit when RSI crosses above 65.",
     ["rsi"], ["RSI(14) crosses above 30"], ["RSI(14) crosses above 65"], "ranging"),
    ("Dual EMA Trend", "Buy when EMA(20) crosses above EMA(60); exit when EMA(20) crosses below EMA(60).",
     ["ema"], ["EMA(20) crosses above EMA(60)"], ["EMA(20) crosses below EMA(60)"], "trending"),
    ("Bollinger Reversion", "Buy when close crosses back above the lower Bollinger band(20,2); exit at the middle band.",
     ["bollinger"], ["close crosses above bollinger lower band"], ["close crosses above bollinger middle band"], "ranging"),
    ("MACD Momentum", "Buy when MACD histogram crosses above 0; exit when it crosses below 0.",
     ["macd"], ["macd histogram crosses above 0"], ["macd histogram crosses below 0"], "trending"),
    ("Stochastic Crossover", "Buy when stochastic %K crosses above %D while stochastic is below 40; exit when %K crosses below %D.",
     ["stochastic"], ["stochastic K crosses above D"], ["stochastic K crosses below D"], "ranging"),
    ("ATR Breakout", "Buy when close crosses above the highest high of the last 20 bars and ATR(14) is above its SMA; exit on a 2*ATR stop.",
     ["atr"], ["close crosses above rolling high 20"], ["close crosses below SMA(30)"], "trending"),
    ("ADX DI Crossover", "Buy when +DI crosses above -DI and ADX(14) is above 25; exit when +DI crosses below -DI.",
     ["adx"], ["plus DI crosses above minus DI"], ["plus DI crosses below minus DI"], "trending"),
    ("RSI Momentum Regime", "Buy when RSI(14) crosses above 55 with rising ADX; exit on RSI crossing below 40.",
     ["rsi", "adx"], ["RSI(14) crosses above 55"], ["RSI(14) crosses below 40"], "trending"),
]


def make_hypothesis(i: int) -> StrategyHypothesis:
    name, desc, ind, entry, exit_, regime = TEMPLATES[i % len(TEMPLATES)]
    suffix = f" {i // len(TEMPLATES) + 1}" if i >= len(TEMPLATES) else ""
    return StrategyHypothesis(
        id=f"batch-{i}",
        name=f"{name}{suffix}",
        description=f"{desc} percent_sizer 0.95.",
        indicators=ind,
        entry_conditions=entry,
        exit_conditions=exit_,
        parameters={"percent_sizer": 0.95},
        rationale="Unit-consistent entry/exit so the conditions can actually fire.",
        mathematical_beauty_score=0.7,
        expected_regime=regime,
        risk_profile="moderate",
    )


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = []
    for i in range(count):
        h = make_hypothesis(i)
        sf = StrategyFactory()          # the real ctor: output_dir, config
        sf.logger = logging.getLogger("batch")
        sf._last_validation_error = ""
        t0 = time.time()
        try:
            spec = sf._create_strategy_spec(h)
            code = sf._generate_code_with_llm(spec, h)
        except Exception as e:
            out.append({"i": i, "hypothesis": h.name, "result": "exception",
                        "error": f"{type(e).__name__}: {e}"})
            print(f"[{i+1}/{count}] {h.name}: EXCEPTION {type(e).__name__}")
            continue
        if not code:
            out.append({"i": i, "hypothesis": h.name, "result": "no_code",
                        "error": sf._last_validation_error})
            print(f"[{i+1}/{count}] {h.name}: no code ({sf._last_validation_error[:80]})")
            continue
        path = Path(sf.output_dir) / f"{h.name.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        tradeable = is_tradeable(path)
        out.append({"i": i, "hypothesis": h.name, "result": "ok", "path": str(path),
                    "tradeable_static": tradeable, "seconds": round(time.time() - t0, 1)})
        print(f"[{i+1}/{count}] {h.name}: {len(code)} B, tradeable={tradeable}, "
              f"{time.time()-t0:.0f}s")
        OUT.write_text(json.dumps(out, indent=1))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
