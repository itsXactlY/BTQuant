#!/usr/bin/env python3
"""
BTQuant Native Aggressive Backtest
Importiert echte BTQuant-Strategieklassen ohne stdlib-Konflikte.
"""
import sys
import os

# PYTHONPATH so setzen dass stdlib nicht von BTQ überschattet wird
# Problem: BTQ's dependencies/backtrader/signal.py shadowed stdlib signal
# Lösung: dependencies/backtrader als Paket importieren, nicht als Verzeichnis

# Only add the project root, NOT dependencies/backtrader directly
BTQ = "/home/alca/projects/PubBTQuant"
sys.path.insert(0, BTQ)
sys.path.insert(0, f"{BTQ}/dependencies")

# Workaround: remove signal from backtrader path if it's shadowing
import importlib
spec = importlib.util.find_spec("backtrader")
if spec and spec.origin and "PubBTQuant" in str(spec.origin):
    # Already got BTQuant's version - good
    pass

import ccxt
import pandas as pd
from datetime import datetime

# --- Lazy-Import der BTQuant-Strategie ---
# Wir müssen BTQuant's backtrader importieren OHNE dass signal.py shadowed
# Lösung: import backtrader zuerst, dann stdlib signal überschreiben

import importlib, sys
backtrader_path = f"{BTQ}/dependencies/backtrader"

# Trick: backtrader zuerst importieren, signal-problem umgehen
import signal as _signal  # stdlib signal zuerst sichern

sys.path.insert(0, backtrader_path)
try:
    import backtrader as bt
except Exception:
    # Wenn circular import: signal reparieren und neu versuchen
    if 'backtrader' in sys.modules:
        del sys.modules['backtrader']
    sys.path = [p for p in sys.path if p != backtrader_path]
    # Jetzt mit korrektem path setup
    sys.path.insert(0, f"{BTQ}/dependencies/backtrader")
    import backtrader as bt

# stdlib signal wiederherstellen (falls BTQ's signal.py es überschrieben hat)
sys.modules['signal'] = _signal

from strategies.SuperTrend_Scalp import SuperSTrend_Scalp


def main():
    print("=" * 68)
    print("  🔴 AGGRO SCALP — BTQUANT NATIVE")
    print("  Strategie: SuperSTrend_Scalp (ADX + DI + SuperTrend)")
    print("  1m BTC/USDT · Binance CCXT")
    print("=" * 68)

    exchange = ccxt.binance()
    since = exchange.parse8601("2026-05-01T00:00:00Z")
    all_ohlcv = []
    for i in range(35):  # max ~35000 candles bei 1m
        chunk = exchange.fetch_ohlcv("BTC/USDT", "1m", since=since, limit=1000)
        if not chunk:
            break
        all_ohlcv.extend(chunk)
        since = chunk[-1][0] + 60000
        import time as _t
        _t.sleep(0.3)
    ohlcv = all_ohlcv
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    print(f"\n📊 {len(df):,} Candles")
    print(f"   {df.index[0]} → {df.index[-1]}")
    print(f"   BTC: ${df['low'].min():,.0f} – ${df['high'].max():,.0f}")

    # Benutze PandasData aus BTQ's pandafeed
    from backtrader.feeds.pandafeed import PandasData
    data = PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SuperSTrend_Scalp,
        adx_period=13, adxth=25, di_period=14,
        st_fast=2, st_fast_multiplier=3,
        st_slow=6, st_slow_multiplier=7,
        take_profit=2, trailing_stop_pct=0.4,
        dca_deviation=1.5, percent_sizer=0.25,
        debug=False, backtest=True,
    )
    cerebro.adddata(data)
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(commission=0.001)

    start = cerebro.broker.getvalue()
    print(f"\n🏁 Start: ${start:,.2f}")
    results = cerebro.run()
    end = cerebro.broker.getvalue()
    strat = results[0]

    ret = (end / start - 1) * 100
    trades = getattr(strat, 'trades', 0) or len(getattr(strat, 'active_orders', []))
    print(f"\n{'='*68}")
    print(f"  📊 ERGEBNIS")
    print(f"{'='*68}")
    print(f"   Start:      ${start:>10,.2f}")
    print(f"   End:        ${end:>10,.2f}")
    print(f"   Rendite:    {ret:>+10.2f}%")
    print(f"   Trades:     {trades}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()