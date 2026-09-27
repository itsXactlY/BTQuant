"""Ground-truth probe: what do this fork's indicators actually expose?

Prints the real line names and the value each line actually takes, so the
strategy-generation prompt can state facts instead of guesses. Runs on real
BTCUSDT bars from the materialized MSSQL cache through the project's PolarsData.

This fork's fork-specific facts, all of which the LLM kept getting wrong:
  * ``indicator.lines.getnames()`` and ``indicator.getline()`` DO NOT EXIST.
    Line names are attributes of the indicator instance (``ind.percK``).
  * ``MACD`` has NO histogram line -- that is the separate ``MACDHisto``.
  * ``OBV``/``Aroon``/``StochasticRSI``/``Donchian``/``Cross`` do not exist.

Usage: probe_indicators.py
"""
import ast
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

import polars as pl

import backtrader as bt

CACHE = Path("/home/alca/projects/PubBTQuant/.btq_cache/mssql")


def load(n: int = 60) -> pl.DataFrame:
    """Real bars from the materialized MSSQL cache -- the production path.

    A synthetic frame is misleading here: this fork's PolarsData parses the
    datetime column as a *string* or a *second/millisecond epoch* (>1e10 is
    read as ms), so a raw-microsecond int yields year 52971. The cache stores
    a real Datetime column, which is what every sweep cell loads.
    """
    for tf in ("1h", "1m", "1d"):
        p = CACHE / f"BTCUSDT_{tf}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            print(f"# bars from {p.name} ({df.height} rows available, using {n})")
            return df.head(n)
    raise SystemExit(f"no cache parquet under {CACHE} -- materialize first")


def line_names(klass) -> list:
    """Line names declared in the class body: ``lines = ('rsi',)``.

    Walked up the MRO because most indicators inherit their lines -- RSI from
    _RSI, ATR from AverageTrueRange, Stochastic from StochasticBase. Reading
    only the subclass finds nothing.
    """
    names: list = []
    for base in inspect.getmro(klass):
        try:
            tree = ast.parse(inspect.getsource(base))
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "lines":
                    try:
                        for nm in ast.literal_eval(node.value):
                            if nm not in names:
                                names.append(nm)
                    except (ValueError, SyntaxError, TypeError):
                        pass
    return names


def probe(klass, **kw):
    names = line_names(klass)
    print(f"  {klass.__name__}({', '.join(f'{k}={v}' for k, v in kw.items())})")
    print(f"      lines = {names}")

    seen = {}

    class Probe(bt.Strategy):
        def __init__(self):
            try:
                self.ind = klass(**kw)
            except Exception as e:                    # noqa: BLE001
                self.ind = None
                self.err = f"{type(e).__name__}: {e}"
                return
            self.err = None

        def next(self):
            if len(self) != 40:
                return
            if self.ind is None:
                # the constructor already failed -- report that, not a
                # misleading missing line
                seen["CONSTRUCTOR FAILED"] = self.err
                raise SystemExit(0)
            for nm in names:
                buf = getattr(self.ind, nm, None)
                if buf is None:
                    seen[nm] = "NO SUCH ATTRIBUTE"
                    continue
                try:
                    seen[nm] = (round(float(buf[0]), 4), round(float(buf[-1]), 4))
                except Exception as e:                # noqa: BLE001
                    seen[nm] = f"{type(e).__name__}: {e}"
            raise SystemExit(0)

    cerebro = bt.Cerebro(stdstats=False, runonce=False)
    cerebro.adddata(bt.feeds.PolarsData(dataname=load()))
    cerebro.addstrategy(Probe)
    err = None
    try:
        cerebro.run()
    except SystemExit:
        pass
    except Exception as e:                            # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    if err:
        print(f"      RAISED {err}")
        return
    if not seen:
        print(f"      NO VALUES (bar 40 never reached; minperiod={klass.__name__} "
              f"needs more bars, or the constructor raised later)")
        return
    for nm, v in seen.items():
        print(f"      .{nm:20s} = {v}" + ("   (now, [-1])" if "FAILED" not in nm else ""))


CASES = [
    (bt.ind.RSI, dict(period=14, upperband=70.0, lowerband=30.0)),
    (bt.ind.EMA, dict(period=20)),
    (bt.ind.SMA, dict(period=20)),
    (bt.ind.WMA, dict(period=20)),
    (bt.ind.ROC, dict(period=9)),
    (bt.ind.Momentum, dict(period=10)),
    (bt.ind.CCI, dict(period=20)),
    (bt.ind.ATR, dict(period=14)),
    (bt.ind.ADX, dict(period=14)),
    (bt.ind.WilliamsR, dict(period=14)),
    (bt.ind.BollingerBands, dict(period=20, devfactor=2.0)),
    (bt.ind.MACD, dict(period_me1=12, period_me2=26, period_signal=9)),
    (bt.ind.MACDHisto, dict(period_me1=12, period_me2=26, period_signal=9)),
    (bt.ind.Stochastic, dict(period=14, period_dfast=3, period_dslow=3)),
    (bt.ind.Stochastic, dict(period=14, period_dfast=3, period_dslow=3,
                             period_d=3)),
    (bt.ind.Stochastic, dict(period=14, smooth=2)),
    (bt.ind.RSI, dict(period=14, smooth=3)),
    (bt.ind.PSAR, dict(step=0.02, maxstep=0.2)),
    (bt.ind.Highest, dict(period=20)),
    (bt.ind.Lowest, dict(period=20)),
    (bt.ind.SMMA, dict(period=20)),
    (bt.ind.ExponentialMovingAverage, dict(period=20)),
]

NONEXISTENT = [
    ("bt.ind.OBV()", lambda: bt.ind.OBV()),
    ("bt.ind.Aroon()", lambda: bt.ind.Aroon()),
    ("bt.ind.StochasticRSI()", lambda: bt.ind.StochasticRSI()),
    ("bt.ind.Donchian()", lambda: bt.ind.Donchian()),
    ("bt.ind.Cross(a, b)", lambda: bt.ind.Cross(bt.ind.RSI(), bt.ind.SMA())),
    ("bt.ind.ZeroCross(a, b)", lambda: bt.ind.ZeroCross(bt.ind.RSI(), bt.ind.SMA())),
    ("bt.ind.Ichimoku()", lambda: bt.ind.Ichimoku()),
    ("bt.ind.PivotPoint()", lambda: bt.ind.PivotPoint()),
    ("bt.ind.VolumeProfile()", lambda: bt.ind.VolumeProfile()),
    ("bt.ind.KeltnerChannel()", lambda: bt.ind.KeltnerChannel()),
    ("bt.ind.DEMA(period=10)", lambda: bt.ind.DEMA(period=10)),
    ("bt.ind.TEMA(period=10)", lambda: bt.ind.TEMA(period=10)),
    ("bt.ind.SqueezeIndicator()", lambda: bt.ind.SqueezeIndicator()),
    ("bt.ind.TRIX(period=15)", lambda: bt.ind.TRIX(period=15)),
    ("bt.ind.ROC(period=9, _ma=bt.ind.SMA)", lambda: bt.ind.ROC(period=9)),
    ("bt.ind.ADX(period=14).plusDI", None),
]


def main():
    print("=== LineBuffer.get signature ===")
    print("  ", inspect.signature(bt.LineBuffer.get))
    print("  attrs starting with 'get':",
          [a for a in dir(bt.LineBuffer) if a.startswith("get")])
    print("=== indicator .lines API present? ===")
    L = bt.ind.RSI.lines
    print("   getnames:", hasattr(L, "getnames"), " getline:", hasattr(L, "getline"))
    print("=== live probe on real bars, bar 40 ===")
    for klass, kw in CASES:
        try:
            probe(klass, **kw)
        except SystemExit:
            continue
    print("=== names the LLM reaches for (need a strategy to instantiate, so")
    print("    only the attribute-existence part is checkable here) ===")
    for name in ("OBV", "Aroon", "StochasticRSI", "Donchian", "Cross",
                 "ZeroCross", "Ichimoku", "PivotPoint", "VolumeProfile",
                 "KeltnerChannel", "DEMA", "TEMA", "SqueezeIndicator", "TRIX",
                 "CrossOver", "Alligator", "ChandelierExit", "Divergence"):
        print(f"  bt.ind.{name:16s} {'EXISTS' if hasattr(bt.ind, name) else 'MISSING'}")
    print("=== DI line names on ADX (static, from source) ===")
    print("   ADX  :", line_names(bt.ind.ADX))
    print("   ADXDI:", line_names(bt.ind.ADXDI) if hasattr(bt.ind, "ADXDI") else "n/a")
    print("=== does the source declare movav/_ma anywhere in RSI? ===")
    src = inspect.getsource(bt.ind.RSI)
    print("   ", [ln.strip() for ln in src.splitlines()
                  if "_ma" in ln or "movav" in ln][:5])


if __name__ == "__main__":
    main()
