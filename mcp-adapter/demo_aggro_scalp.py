#!/usr/bin/env python3
"""
BTQuant MCP Demo — Aggressive Momentum Scalp
Strategy: ADX + DI + ParabolicSAR Momentum Scalp (1m BTC/USDT)
Aggressive: 25% Position · 1.5% TP · 0.3% Trail · DCA bei 1%
"""
import ccxt
import pandas as pd
import backtrader as bt
from datetime import datetime


class AggroMomentumScalp(bt.Strategy):
    """ADX + DI + PSAR aggressive momentum scalp"""

    params = (
        ("adx_period", 14),
        ("adxth", 30),          # Strong trend only
        ("di_period", 14),
        ("sar_acc", 0.02),
        ("sar_max", 0.2),
        ("take_profit_pct", 1.5),
        ("trail_pct", 0.3),
        ("dca_dev", 1.0),
        ("sizer", 0.25),
    )

    def __init__(self):
        self.adx = bt.ind.ADX(period=self.p.adx_period)
        self.plus_di = bt.ind.PlusDI(period=self.p.di_period)
        self.minus_di = bt.ind.MinusDI(period=self.p.di_period)
        self.psar = bt.ind.PSAR(afstart=self.p.sar_acc, afmax=self.p.sar_max)
        self.rsi = bt.ind.RSI(period=7)
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.trades = 0
        self.wins = 0

    def next(self):
        has_pos = self.position.size > 0

        # ADX strong + MinusDI bear exhaustion + price above PSAR
        long_signal = (
            self.adx[0] >= self.p.adxth
            and self.minus_di[0] > self.plus_di[0]
            and self.data.close[0] > self.psar[0]
            and self.rsi[0] < 55
        )

        if long_signal and not has_pos:
            size = int(self.broker.getvalue() * self.p.sizer / self.data.close[0])
            self.buy(size=max(size, 1))
            self.entry_price = self.data.close[0]
            self.peak_price = self.data.close[0]
            self.trades += 1
            return

        if has_pos:
            self.peak_price = max(self.peak_price, self.data.close[0])

            # TP
            if self.data.close[0] >= self.entry_price * (1 + self.p.take_profit_pct / 100):
                self.close()
                self.wins += 1
                return

            # Trailing
            if self.data.close[0] <= self.peak_price * (1 - self.p.trail_pct / 100):
                self.close()
                return

            # DCA
            if self.data.close[0] <= self.entry_price * (1 - self.p.dca_dev / 100):
                size = int(self.broker.getvalue() * self.p.sizer / self.data.close[0])
                self.buy(size=max(size, 1))
                self.entry_price = (self.entry_price + self.data.close[0]) / 2


def main():
    print("=" * 65)
    print("  AGGRO SCALP — BTC/USDT 1m")
    print("  SuperTrend + ADX + DI | 25% Position | 2% TP | 0.4% Trail")
    print("=" * 65)

    # Daten via CCXT
    print("\n📡 Fetching BTC/USDT 1m from Binance CCXT ...")
    exchange = ccxt.binance()
    since = exchange.parse8601("2026-05-01T00:00:00Z")
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1m", since=since, limit=35000)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    print(f"   {len(df):,} candles")
    print(f"   {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"   BTC: ${df['low'].min():>8,.0f} – ${df['high'].max():>8,.0f}")

    # Backtest
    data = bt.feeds.PandasData(dataname=df)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AggroMomentumScalp)
    cerebro.adddata(data)
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(commission=0.001)

    start_val = cerebro.broker.getvalue()
    print(f"\n🏁 Starting: ${start_val:,.2f}")
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()
    strat = results[0]

    ret = (end_val / start_val - 1) * 100
    print("\n" + "=" * 65)
    print("  📊 RESULTS")
    print("=" * 65)
    print(f"   Start:     ${start_val:>10,.2f}")
    print(f"   End:       ${end_val:>10,.2f}")
    print(f"   Return:    {ret:>+10.2f}%")
    print(f"   Trades:    {strat.trades}")
    print(f"   Win Rate:  {(strat.wins/strat.trades*100):.1f}%" if strat.trades > 0 else "   Win Rate:  N/A")
    print("=" * 65)


if __name__ == "__main__":
    main()