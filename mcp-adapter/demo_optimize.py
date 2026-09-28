#!/usr/bin/env python3
"""SuperTrend_Scalp Parameter Optimization — ADX, Sizer, TP, Trail"""
import sys, os, signal as _signal, time as _time
BTQ = "/home/alca/projects/PubBTQuant"
sys.path.insert(0, BTQ); sys.path.insert(0, f"{BTQ}/dependencies")
sys.path.insert(0, f"{BTQ}/dependencies/backtrader")
sys.path.insert(0, f"{BTQ}/dependencies/backtrader/feeds/mssql")

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
from strategies.SuperTrend_Scalp import SuperSTrend_Scalp
import ccxt, pandas as pd

print("📡 Fetching BTC/USDT 1m ...")
exchange = ccxt.binance()
since = exchange.parse8601("2026-05-01T00:00:00Z")
all_c = []
for i in range(35):
    chunk = exchange.fetch_ohlcv("BTC/USDT", "1m", since=since, limit=1000)
    if not chunk: break
    all_c.extend(chunk)
    since = chunk[-1][0] + 60000
    _time.sleep(0.2)

df = pd.DataFrame(all_c, columns=["timestamp","open","high","low","close","volume"])
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)
print(f"   {len(df):,} candles | {df.index[0]} – {df.index[-1]}")
print(f"   BTC: ${df['low'].min():,.0f} – ${df['high'].max():,.0f}")

# ── Parameter Grid (focused — 36 combos) ──────────────────────────
grid = []
for adxth in [15, 20, 25]:        # 3
    for sizer in [0.35, 0.50, 0.75]:  # 3
        for tp in [1.5, 2.0, 3.0]:     # 3
            for trail in [0.3, 0.4]:    # 2 — max 54 combos
                if tp >= trail * 3:    # TP muss > Trail sein
                    grid.append((adxth, sizer, tp, trail))

print(f"\n🧪 Testing {len(grid)} parameter combinations ...\n")

results = []
total = len(grid)
for idx, (adxth, sizer, tp, trail) in enumerate(grid):
    try:
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SuperSTrend_Scalp,
            adx_period=13, adxth=adxth, di_period=14,
            st_fast=2, st_fast_multiplier=3,
            st_slow=6, st_slow_multiplier=7,
            take_profit=tp, trailing_stop_pct=trail,
            dca_deviation=1.5, percent_sizer=sizer,
            debug=False, backtest=True,
        )
        cerebro.adddata(PandasData(dataname=df.copy()))
        cerebro.broker.setcash(100.0)
        cerebro.broker.setcommission(commission=0.001)

        start_val = cerebro.broker.getvalue()
        cerebro.run()
        end_val = cerebro.broker.getvalue()
        ret = (end_val / start_val - 1) * 100

        # Trades zählen
        strat = cerebro.runstrats[0][0]
        trades = 0
        for attr in ['trades', 'wins', 'trade_count']:
            v = getattr(strat, attr, None)
            if v and isinstance(v, (int, float)):
                trades += int(v)

        results.append((adxth, sizer, tp, trail, end_val, ret, trades))

        bar = f"[{idx+1}/{total}]"
        arrow = "📈" if ret > 0 else "📉" if ret < 0 else "➡️"
        print(f"  {bar} ADX≥{adxth:2d}  Sizer={sizer:.0%}  TP={tp:.1f}%  Trail={trail:.1f}%  →  ${end_val:>6.2f}  {arrow}{ret:>+6.2f}%  Trades:{trades:>3d}")
    except Exception as e:
        print(f"  [{idx+1}/{total}] ADX≥{adxth:2d}  Sizer={sizer:.0%}  →  💥 {str(e)[:60]}")

# ── Rangliste ────────────────────────────────────────────────────────
sorted_r = sorted([r for r in results if isinstance(r[4], (int,float))], key=lambda x: -x[5])

print(f"\n{'='*80}")
print(f"  🏆 PARAMETER RANKING — SuperTrend_Scalp")
print(f"{'='*80}")
print(f"  {'Rank':4s} {'ADX':4s} {'Sizer':6s} {'TP':5s} {'Trail':6s} {'End':>8s} {'Return':>7s} {'Trades':>6s}")
print(f"  {'─'*4} {'─'*4} {'─'*6} {'─'*5} {'─'*6} {'─'*8} {'─'*7} {'─'*6}")
for i, (adxth, sizer, tp, trail, end, ret, trades) in enumerate(sorted_r[:20], 1):
    m = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    a = "📈" if ret > 0 else "📉" if ret < 0 else "➡️"
    print(f"  {i:3d}. {m}  {adxth:2d}   {sizer:.0%}   {tp:.1f}%  {trail:.1f}%   ${end:>6.2f}  {a}{ret:>+6.2f}%  {trades:>4d}")

# Top-3 Empfehlung
print(f"\n{'='*80}")
print(f"  💡 TOP 3 PARAMETER")
print(f"{'='*80}")
for i in range(min(3, len(sorted_r))):
    adxth, sizer, tp, trail, end, ret, trades = sorted_r[i]
    print(f"\n  #{i+1}: ADX≥{adxth} · Sizer {sizer:.0%} · TP {tp:.1f}% · Trail {trail:.1f}%")
    print(f"       → ${end:.2f} ({ret:+.2f}%) · {trades} Trades")
    print(f"       Strategie-Aufruf:")
    print(f"       SuperSTrend_Scalp(adxth={adxth}, percent_sizer={sizer}, take_profit={tp}, trailing_stop_pct={trail})")

print(f"\n  📊 {len(df):,} Candles · 35.000 Parameter-Tests")