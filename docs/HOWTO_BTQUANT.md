# How To BTQuant

Based on the actual codebase at /home/alca/projects/PubBTQuant/.

## What BTQuant Is

BTQuant is a fork of the backtrader backtesting framework, extended for crypto trading.
It adds live trading (via JackRabbitRelay, CCXT, PancakeSwap), HotSpine shared memory
for market data, SQL Server storage, a CLI, 20+ strategies, and indicator transparency.

The core lives in `dependencies/backtrader/`. Everything else builds on it.

## Installation

Requirements: Linux, Python 3.12+, GCC 7+, CMake 3.15+, 8GB RAM.

git clone --recurse-submodules https://github.com/ItsXactlY/BTQuant.git
cd BTQuant
bash Installers/install_all.sh

The installer: detects your distro, installs system deps, installs SQL Server (optional),
creates a venv, installs Python packages, builds CCAPI from source.

After install, configure `dependencies/backtrader/dontcommit.py` with your credentials
(JRR webhook URL, Telegram/Discord, SQL Server connection).

## The btq CLI

Usage: btq MODE [options]

Modes: backtest, bulk, optimize, list, live (live not yet implemented in CLI).

Backtest a coin:
  btq backtest --coin BTC --interval 15m --start 2024-01-01 --end 2025-01-01 --plot

Bulk backtest:
  btq bulk --interval 1h --workers 8

Optimize:
  btq optimize --coin BTC --strategy VuManchCipher_A --trials 200 --workers 8

Common flags: --coin, --coins, --strategy, --collateral (default USDT),
--interval (default 15m), --start, --end, --cash (default 1000),
--commission (default 0.00075), --plot, --quantstats, --debug.

## Writing Strategies

All strategies inherit from BaseStrategy (dependencies/backtrader/strategies/base.py).

Override these 3 methods:

- buy_or_short_condition() - entry logic. Call self.create_order("BUY") to enter.
- dca_or_short_condition() - DCA entry. Called when already in position and DCA enabled.
- sell_or_cover_condition() - exit logic. Call self.close_order() to exit.

Example (SMA crossover):

from backtrader.strategies.base import BaseStrategy, bt

class MySMA(BaseStrategy):
    params = (("short_period", 10), ("long_period", 30))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sma_short = bt.ind.SMA(period=self.p.short_period)
        self.sma_long = bt.ind.SMA(period=self.p.long_period)
        self.crossover = bt.ind.CrossOver(self.sma_short, self.sma_long)

    def buy_or_short_condition(self):
        if self.crossover > 0:
            self.create_order(action="BUY")
            return True

    def sell_or_cover_condition(self):
        if self.crossover < 0 and self.buy_executed:
            for o in self.active_orders[:]:
                self.close_order(o, self.data.close[0])

Key BaseStrategy params: init_cash, exchange, account, asset, coin, collateral,
backtest, debug, take_profit, stop_loss, percent_sizer, order_cooldown,
enable_alerts, DCA (set in __init__).

OrderTracker persists orders to CSV in .OrderTracker_Live/ (live only).

## Data Sources

- CCXT: from backtrader.feeds.ccxt import CCXT (100+ exchanges)
- MSSQL: from backtrader.feeds.mssql_crypto import MsSqlCrypto
- CSV/Polars: from backtrader.feeds.polarfeed import PolarsFeed
- HotSpine: from backtrader.feeds.hotspine_feed import HotSpineData
- TradingView: from backtrader.feeds.tv_feed import TradingViewFeed

CCXT config: loaded from <venv>/ccxt/{exchange}_{account}.json or BTQ_ env vars.

## Live Trading

Via JackRabbitRelay (most common):
  from backtrader.livetrading import livetrade_ccxt
  livetrade_ccxt(coin="BTC", collateral="USDT", exchange="mimic",
                 account="default", asset="BTC/USDT", strategy=MyStrategy)

Via CCXT direct:
  livetrade_ccxt(coin="BTC", exchange="binance", ...)

Via HotSpine (lowest latency):
  from backtrader.livetrading import livetrade_hotspine
  livetrade_hotspine(symbol_id=123, strategy=MyStrategy)

Functions: livetrade_ccxt, livetrade_binance, livetrade_mexc, livetrade_bitget,
livetrade_web3 (PancakeSwap), livetrade_hotspine, livetrade_tv (TradingView).

JrrBroker sends orders as HTTP POST to http://127.0.0.1:80 with Exchange,
Market, Account, Action (Buy/Close), Asset, USD, Identity fields.

## HotSpine

Shared memory for reading market data from C++ collectors.
Path: /dev/shm/btquant_hotspine

from backtrader.hotspine.reader import HotSpineReader
reader = HotSpineReader(shm_name="/btquant_hotspine")
trade = reader.poll_trade()  # Returns HotTrade or None

HotTrade struct (32 bytes): ts_exchange (u64), ts_local (u64), price (f64),
size (f64), symbol_id (u32), side (u8, 0=buy 1=sell), padding (3 bytes).

SQL integration (HotSpineSQLIntegration) stores trades to SQL Server async.

## Available Strategies

SMA_Cross_Simple, SMA_Cross_MESAdaptive_Prime, VuManchCipher_A, VuManchCipher_B,
NearestNeighbors_RationalQuadraticKernel (ML/KNN), QQE_Hullband_VolumeOsc,
ST_RSX_ASI, SuperTrend_Scalp, MACD_ADX, Aligator_supertrend,
OrderChain, Order_Chain_Kioseff_Trading, StagedConvergenceStrategy,
SineWeightZeroLagQQEVolMesaAdaptive, pancakeswap_orders,
pancakeswap_dca_marketmaker, jrr_orders, __TEMPLATE__.

## Examples

Backtest_CCXT.py - backtest with CCXT data
Backtest_CSV.py - backtest with Polars CSV
Backtest_MsSQL.py - backtest with SQL Server
Backtest_Bulk_MsSQL.py - bulk backtest
Trading_CCXT.py - live via CCXT
Trading_Websocket_Binance.py - Binance websocket
Trading_Websocket_Bitget.py - Bitget websocket
Trading_Crypto_Tradingview.py - TradingView + JRR
Live_Trading_HotSpine_SMA.py - HotSpine + SMA

## Indicator Transparency

from backtrader.strategies.base import activate_patch, capture_patch
activate_patch()
# In strategy next(): capture_patch(self)
# Exports all indicator values to Polars DataFrame / Parquet / CSV.
