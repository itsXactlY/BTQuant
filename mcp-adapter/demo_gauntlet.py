#!/usr/bin/env python3
"""
BTQuant Full Gauntlet — v2
Importiert echte BTQuant-Strategien + backtrader fork + CCXT data (proven working approach).
"""
import sys, os, signal as _signal, time as _time
BTQ = "/home/alca/projects/PubBTQuant"
sys.path.insert(0, BTQ); sys.path.insert(0, f"{BTQ}/dependencies")
sys.path.insert(0, f"{BTQ}/dependencies/backtrader")
sys.path.insert(0, f"{BTQ}/dependencies/backtrader/feeds/mssql")

# Backtrader import mit circular-import workaround (wie in demo_btquant_aggro.py)
try:
    import backtrader as bt
except Exception:
    for k in list(sys.modules.keys()):
        if 'backtrader' in k: del sys.modules[k]
    sys.path = [p for p in sys.path if f"{BTQ}/dependencies/backtrader" not in p]
    sys.path.insert(0, f"{BTQ}/dependencies/backtrader")
    sys.path.insert(0, f"{BTQ}/dependencies/backtrader/feeds/mssql")
    import backtrader as bt
sys.modules['signal'] = _signal

from backtrader.feeds.pandafeed import PandasData
import ccxt
import pandas as pd

# ── Alle Strategien ─────────────────────────────────────────────────
STRATEGIES = [
    ("SMA_Cross_Simple",          "SMA_Cross_Simple",          {}),
    ("SuperTrend_Scalp",          "SuperSTrend_Scalp",         {"adx_period":13,"adxth":20,"st_fast":2,"st_fast_multiplier":3,"st_slow":6,"st_slow_multiplier":7,"take_profit":2,"trailing_stop_pct":0.4,"dca_deviation":1.5}),
    ("Vumanchu_A",                "VuManchCipher_A",           {"take_profit":3,"dca_deviation":4}),
    ("Vumanchu_B",                "VuManchCipher_B",           {"take_profit":3,"dca_deviation":4}),
    ("ST_RSX_ASI",                "STrend_RSX_AccumulativeSwingIndex", {"take_profit":2,"trailing_stop_pct":0.5}),
    ("Aligator_supertrend",       "AliG_STrend",               {"take_profit":3}),
    ("QQE_Hullband_VolumeOsc",    "QQE_Example",               {"take_profit":3}),
    ("SineWeightZeroLagQQEVolMesaAdaptive", "FastSineWeightZeroLagQQEVolMesaAdaptive", {}),
    ("NearestNeighbors_RationalQuadraticKernel", "NRK",        {"take_profit":3,"period":14}),
    ("SMA_Cross_MESAdaptive_Prime", "SMA_Cross_MESAdaptivePrime", {}),
    ("StagedConvergenceStrategy", "StagedConvergenceStrategy",  {"take_profit":3}),
]

# ── Daten ────────────────────────────────────────────────────────────
print("📡 Fetching BTC/USDT 1m ...")
exchange = ccxt.binance()
since = exchange.parse8601("2026-05-01T00:00:00Z")
all_c = []
for i in range(35):
    chunk = exchange.fetch_ohlcv("BTC/USDT", "1m", since=since, limit=1000)
    if not chunk: break
    all_c.extend(chunk)
    since = chunk[-1][0] + 60000
    _time.sleep(0.25)

df = pd.DataFrame(all_c, columns=["timestamp","open","high","low","close","volume"])
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)
print(f"   {len(df):,} candles | {df.index[0].strftime('%d.%m %H:%M')} – {df.index[-1].strftime('%d.%m %H:%M')}")

# ── Gauntlet ────────────────────────────────────────────────────────
INITIAL = 100.0
SIZER = 0.35

print(f"\n🧪 Testing {len(STRATEGIES)} strategies (${INITIAL:.0f} · {SIZER*100:.0f}% sizer) ...\n")

results = []
for fname, cname, params in STRATEGIES:
    try:
        mod = __import__(f"strategies.{fname}", fromlist=["_"])
        cls = getattr(mod, cname)
    except Exception as e:
        print(f"  💥 {fname:35s} IMPORT FAIL: {e}")
        continue

    p = {"percent_sizer": SIZER, "debug": False, "backtest": True}
    p.update(params)

    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(cls, **p)
        fresh_data = PandasData(dataname=df.copy())
        cerebro.adddata(fresh_data)
        cerebro.broker.setcash(INITIAL)
        cerebro.broker.setcommission(commission=0.001)

        cerebro.run()
        end_val = cerebro.broker.getvalue()
        ret = (end_val / INITIAL - 1) * 100

        # Trades zählen
        strat = cerebro.runstrats[0][0]
        trades = 0
        for attr in ['trades', 'wins', 'trade_count', 'ordercount']:
            v = getattr(strat, attr, None)
            if v:
                if isinstance(v, (int, float)): trades += int(v)
                elif hasattr(v, '__len__'): trades += len(v)

        arrow = "📈" if ret > 0 else "📉" if ret < 0 else "➡️"
        print(f"  {arrow} {fname:35s} ${end_val:>7.2f}  {ret:>+7.2f}%  Trades:{trades:>4d}")
        results.append((fname, end_val, ret, trades))
    except Exception as e:
        print(f"  💥 {fname:35s} FAIL: {str(e)[:80]}")
        results.append((fname, 0, -999, 0))

# ── Rangliste ───────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  🏆 RANGLISTE — {INITIAL:.0f}$ · {SIZER*100:.0f}% Sizer")
print(f"{'='*75}")
print(f"  {'#':4s} {'Strategie':35s} {'End':>8s} {'Return':>8s} {'Trades':>6s}")
print(f"  {'────':4s} {'─'*35} {'─'*8} {'─'*8} {'─'*6}")
sorted_r = sorted([r for r in results if isinstance(r[2], (int,float)) and r[2] > -500], key=lambda x: -x[2])
for i, (name, end, ret, trades) in enumerate(sorted_r, 1):
    m = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    a = "📈" if ret > 0 else "📉" if ret < 0 else "➡️"
    print(f"  {i:3d}. {m} {name:35s} ${end:>7.2f}  {a}{ret:>+7.2f}%  {trades:>4d}")

print(f"\n  📊 {len(df):,} Candles 1m BTC/USDT · {df.index[0].strftime('%d.%m')} – {df.index[-1].strftime('%d.%m')}")