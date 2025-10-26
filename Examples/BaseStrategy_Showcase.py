"""Comprehensive BaseStrategy example
=====================================

This script demonstrates how to build a production-ready strategy on top of
``BaseStrategy``.  It walks through:

* Wiring the strategy into a :class:`backtrader.cerebro.Cerebro` engine for
  backtesting.
* Re-using the same strategy with CCXT-based exchanges (e.g. Binance, MEXC).
* Preparing the configuration for Web3 compatible exchanges such as
  PancakeSwap via the built-in PancakeSwap order bridge.
* Tapping into the order tracking utilities that persist active positions.

The example is intentionally verbose and heavily documented so that new users
can use it as a template for their own ideas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import backtrader as bt

from dependencies.backtrader.strategies.base import BaseStrategy, OrderTracker


class FullyDocumentedStrategy(BaseStrategy):
    """Example strategy showcasing BaseStrategy features.

    The strategy trades a simple moving-average cross over the ``close`` prices
    of the first data feed, but the surrounding infrastructure demonstrates how
    to work with ``BaseStrategy``'s order tracking, position sizing, and
    exchange helpers.

    Highlights
    ----------
    * Uses ``self.calculate_position_size`` to respect exchange minimum order
      sizes.
    * Demonstrates how to persist orders with :class:`OrderTracker`.
    * Triggers alerts via the BaseStrategy hooks when ``enable_alerts`` is set.
    * Works both for standard CCXT exchanges and Web3 integrations.
    """

    params = dict(
        short_period=10,
        long_period=30,
        percent_sizer=0.05,
        take_profit=1.0,
        backtest=True,
        exchange="binance",
        asset="BTCUSDT",
    )

    def __init__(self):
        super().__init__()
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.short_period)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.long_period)

    def next(self):
        if not self.position and self.short_ma > self.long_ma and self.buy_or_short_condition() is False:
            self.calculate_position_size()
            size = self.amount or 0.001
            order = self.buy(size=size)
            self.entry_prices.append(self.data.close[0])
            self.sizes.append(size)
            self.calc_averages()
            OrderTracker(self.data.close[0], size, self.p.take_profit, symbol=self.p.asset, backtest=self.p.backtest)
            self.send_alert(f"Opened long position via BaseStrategy at {self.data.close[0]:.2f}")
            return order

        if self.position and self.short_ma < self.long_ma and self.sell_or_cover_condition() is False:
            self.sell(size=self.position.size)
            self.send_alert(f"Closed long position via BaseStrategy at {self.data.close[0]:.2f}")


@dataclass
class StrategyConfig:
    """Configuration container for the showcase example."""

    data: Iterable[bt.feeds.PandasData]
    cash: float = 100_000.0
    exchange: str = "binance"
    asset: str = "BTCUSDT"
    enable_alerts: bool = False
    alert_channel: Optional[str] = None
    pancakeswap_coin: Optional[str] = None
    pancakeswap_collateral: Optional[str] = None


def run_backtest(config: StrategyConfig) -> bt.Cerebro:
    """Run a standard backtest using ``FullyDocumentedStrategy``.

    Parameters
    ----------
    config:
        High level configuration describing the data feeds and exchange flavour.
    """

    cerebro = bt.Cerebro()
    for data_feed in config.data:
        cerebro.adddata(data_feed)

    cerebro.broker.setcash(config.cash)

    cerebro.addstrategy(
        FullyDocumentedStrategy,
        exchange=config.exchange,
        asset=config.asset,
        enable_alerts=config.enable_alerts,
        alert_channel=config.alert_channel,
        coin=config.pancakeswap_coin,
        collateral=config.pancakeswap_collateral,
        backtest=True,
    )

    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    return cerebro


def example_ccxt_usage(data_feed: bt.feeds.PandasData):
    """Illustrate how to configure the strategy for a CCXT exchange."""

    config = StrategyConfig(data=[data_feed], exchange="binance", asset="BTCUSDT")
    cerebro = run_backtest(config)
    results = cerebro.run()
    analyzer = results[0].analyzers.trades.get_analysis()
    print("Trade analysis:", analyzer)


def example_web3_usage(data_feed: bt.feeds.PandasData):
    """Illustrate how to configure the strategy for PancakeSwap/Web3 trading."""

    config = StrategyConfig(
        data=[data_feed],
        exchange="pancakeswap",
        asset="CAKEBUSD",
        pancakeswap_coin="CAKE",
        pancakeswap_collateral="BUSD",
        enable_alerts=True,
    )
    cerebro = run_backtest(config)
    cerebro.run()
    print("Web3 example completed. Orders will be routed through the PancakeSwap bridge.")


def load_csv_data(path: Path) -> bt.feeds.PandasData:
    """Load OHLCV data from CSV using pandas."""

    import pandas as pd

    dataframe = pd.read_csv(path, parse_dates=True, index_col="datetime")
    return bt.feeds.PandasData(dataname=dataframe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BaseStrategy showcase example")
    parser.add_argument("--csv", type=Path, help="Path to a CSV file containing OHLCV data", required=False)
    parser.add_argument("--web3", action="store_true", help="Demonstrate PancakeSwap/Web3 setup")
    parser.add_argument("--alerts", action="store_true", help="Enable alert integrations")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.csv is None:
        raise SystemExit("Please provide a CSV file via --csv to run the example.")

    data_feed = load_csv_data(args.csv)

    if args.web3:
        example_web3_usage(data_feed)
    else:
        example_ccxt_usage(data_feed)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
