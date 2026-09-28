#!/usr/bin/env python3
"""
Standalone BTC/USDT 15m Backtest — SMA Crossover
Uses only: pip backtrader + ccxt + pandas (no BTQuant dependencies)
"""
import ccxt
import pandas as pd
import backtrader as bt
from datetime import datetime


class SMA_Cross(bt.Strategy):
    params = (('short', 10), ('long', 30))

    def __init__(self):
        self.sma_short = bt.ind.SMA(period=self.p.short)
        self.sma_long = bt.ind.SMA(period=self.p.long)
        self.crossover = bt.ind.CrossOver(self.sma_short, self.sma_long)
        self.trades = 0

    def next(self):
        if self.crossover[0] > 0 and not self.position:
            self.buy()
            self.trades += 1
        elif self.crossover[0] < 0 and self.position:
            self.close()
            self.trades += 1


def main():
    print("=" * 60)
    print("BTQuant MCP Demo — SMA Crossover Backtest")
    print("BTC/USDT 15m · 1. Mai – 25. Mai 2026")
    print("=" * 60)

    # Fetch data
    print("\n📡 Fetching BTC/USDT 15m from Binance CCXT ...")
    exchange = ccxt.binance()
    since = exchange.parse8601("2026-05-01T00:00:00Z")
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "15m", since=since, limit=2500)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    print(f"   {len(df)} candles")
    print(f"   {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"   BTC: ${df['low'].min():,.0f} – ${df['high'].max():,.0f}")

    # Create data feed
    data = bt.feeds.PandasData(dataname=df)

    # Set up cerebro
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SMA_Cross)
    cerebro.adddata(data)
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%

    # Run
    print("\n🏁 Running backtest ...")
    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    end_val = cerebro.broker.getvalue()

    # Results
    strat = results[0]
    ret = (end_val / start_val - 1) * 100

    print("\n" + "=" * 60)
    print("📊 RESULTAT")
    print("=" * 60)
    print(f"   Startkapital:    ${start_val:>8,.2f}")
    print(f"   Endkapital:      ${end_val:>8,.2f}")
    print(f"   Rendite:         {ret:>+7.2f}%")
    print(f"   Trades:          {strat.trades}")
    print(f"   Strategie:       SMA(10) × SMA(30) Crossover")
    print("=" * 60)


if __name__ == "__main__":
    main()