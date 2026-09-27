"""Read-only probe: which interpreter + which backtrader does the agency see?"""
import importlib.util
import sys

print("exe   :", sys.executable)
print("ver   :", sys.version.split()[0])
print("prefix:", sys.prefix)
try:
    import backtrader as bt
    print("bt ver:", bt.__version__)
    print("bt lib:", bt.__file__)
    print("has utils.backtest :", hasattr(bt, "backtest"))
    print("has feeds.PolarsData:", hasattr(bt.feeds, "PolarsData"))
    print("has strategies pkg :", importlib.util.find_spec("backtrader.strategies") is not None)
    print("has transparencypatch:", importlib.util.find_spec("backtrader.transparencypatch") is not None)
    try:
        from backtrader.strategies.base import BaseStrategy
        print("BaseStrategy import: OK", BaseStrategy)
    except Exception as e:
        print("BaseStrategy import: FAIL", type(e).__name__, e)
    try:
        import polars
        print("polars:", polars.__version__)
    except Exception as e:
        print("polars: MISSING", e)
except Exception as e:
    print("backtrader import: FAIL", type(e).__name__, e)
for mod in ("pandas", "numpy", "quantstats_lumi", "pyarrow"):
    try:
        m = __import__(mod)
        print(f"{mod:16s}", getattr(m, "__version__", "?"), m.__file__)
    except Exception as e:
        print(f"{mod:16s} MISSING {e}")
