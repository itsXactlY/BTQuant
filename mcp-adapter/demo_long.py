#!/usr/bin/env python3
"""SuperTrend_Scalp — $1.000 · quantitative sizing · 01.01.2025 – jetzt"""
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

# ── Daten ────────────────────────────────────────────────────────────
print("📡 Fetching BTC/USDT 1m 2025-01-01 → jetzt ...")
exchange = ccxt.binance({"enableRateLimit": True})
since = exchange.parse8601("2025-01-01T00:00:00Z")
all_c = []
page = 0
errs = 0
while errs < 3:
    try:
        chunk = exchange.fetch_ohlcv("BTC/USDT", "1m", since=since, limit=1000)
        if not chunk: break
        all_c.extend(chunk)
        page += 1
        since = chunk[-1][0] + 60000
        if page % 50 == 0:
            print(f"   {page*1000:>6,} candles ...")
        _time.sleep(0.22)
        errs = 0
    except Exception as e:
        errs += 1; _time.sleep(2)

df = pd.DataFrame(all_c, columns=["timestamp","open","high","low","close","volume"])
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)
print(f"\n   ✅ {len(df):,} candles")
print(f"   📅 {df.index[0].strftime('%d.%m.%Y')} – {df.index[-1].strftime('%d.%m.%Y')}")
print(f"   💰 BTC: ${df['low'].min():,.0f} – ${df['high'].max():,.0f}")

# ── Backtest (kein fixed size — framework-internes quantitative sizing) ─
INITIAL = 1000.0
PARAMS = dict(
    adx_period=13, adxth=20, di_period=14,
    st_fast=2, st_fast_multiplier=3,
    st_slow=6, st_slow_multiplier=7,
    take_profit=2.0, trailing_stop_pct=0.3,
    dca_deviation=1.5,
    # percent_sizer NICHT setzen — Framework nutzt 25% default
    # Stattdessen: framework-eigene Position-Sizer via create_order()
    init_cash=INITIAL,
    debug=False, backtest=True,
)

print(f"\n🏁 SuperSTrend_Scalp · ${INITIAL:,.0f} · ADX≥20 · TP 2% · Trail 0.3%")
print(f"   Size: Framework-Quantitativ (25% per default via calculate_position_size)")

cerebro = bt.Cerebro()
cerebro.addstrategy(SuperSTrend_Scalp, **PARAMS)
cerebro.adddata(PandasData(dataname=df))
cerebro.broker.setcash(INITIAL)
cerebro.broker.setcommission(commission=0.001)

start_val = cerebro.broker.getvalue()
results = cerebro.run()
end_val = cerebro.broker.getvalue()
strat = results[0]

ret = (end_val / start_val - 1) * 100

# Trade-Statistiken aus der Engine
trades = 0
wins = 0
for attr in ['trades', 'wins', 'trade_count']:
    v = getattr(strat, attr, None)
    if v and isinstance(v, (int, float)):
        trades += int(v)

# Analyzer für bessere Stats
import numpy as np
try:
    ta = strat.analyzers.getbyname('tradeanalyzer')
    if ta:
        won = ta.get('won', {}).get('total', 0)
        lost = ta.get('lost', {}).get('total', 0)
        trades = won + lost
        wins = won
except: pass

print(f"\n{'='*75}")
print(f"  📊 SUPERSTrend_Scalp — ${INITIAL:,.0f} quantitative sizing")
print(f"{'='*75}")
print(f"   Zeitraum:      {df.index[0].strftime('%d.%m.%Y')} – {df.index[-1].strftime('%d.%m.%Y')}")
print(f"   Candles:       {len(df):,} (1m BTC/USDT)")
print(f"   Startkapital:  ${start_val:>8,.2f}")
print(f"   Endkapital:    ${end_val:>8,.2f}")
print(f"   Rendite:       {ret:>+8.2f}%")
print(f"   Trades:        {trades}")
print(f"   Win Rate:      {(wins/trades*100):.1f}%" if trades > 0 else "   Win Rate:      N/A")
print(f"   Parameter:     ADX≥20 · TP 2% · Trail 0.3% · quant. Position-Sizing")
print(f"{'='*75}")