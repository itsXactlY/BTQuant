# BTQuant Quick Start Guide

## Prerequisites

- BTQuant installed (see [Installation Guide](installation.md))
- Virtual environment activated: `source ~/.btq/bin/activate`
- A data source: either CCXT exchange access (internet) or SQL Server with market data

## Your First Backtest

### Option A: CCXT Data (no database required)

Fetch data directly from any CCXT-supported exchange:

```python
from backtrader import backtest, get_crypto_data
from backtrader.strategies.Vumanchu_A import VuManchCipher_A

# Fetch 1 week of BTC/USDT 15-minute candles from Binance
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-08', '15m', 'binance')

backtest(VuManchCipher_A,
         data=data,
         init_cash=1000,
         quantstats=True,
         plot=True,
         asset_name='BTC/USDT')
```

### Option B: SQL Server Data

If you have MSSQL set up with market data:

```python
from backtrader.utils.backtest import backtest
from backtrader.strategies.ST_RSX_ASI import STrend_RSX_AccumulativeSwingIndex

backtest(STrend_RSX_AccumulativeSwingIndex,
         coin='BTC',
         collateral='USDT',
         start_date='2024-01-01',
         end_date='2024-02-15',
         interval='1m',
         init_cash=1000,
         plot=True,
         quantstats=False)
```

When `data` is not provided, BTQuant fetches from SQL Server via `PolarsDataLoader` and caches the result as Parquet in `.btq_cache/`.

## Using the CLI

The `btq` command runs backtests from the terminal:

```bash
# Single coin backtest
btq backtest --coin BTC --strategy VuManchCipher_A --interval 15m --start 2024-01-01 --end 2024-01-08 --plot

# Multiple coins
btq backtest --coins BTC,ETH,BNB --strategy Order_Chain_Kioseff_Trading --interval 1h

# List available strategies
btq list strategies

# List coins in the database
btq list coins --collateral USDT
```

## Writing a Strategy

All BTQuant strategies extend `BaseStrategy` from `backtrader.strategies.base`. You override four methods to define your trading logic:

### The Strategy Interface

```python
from backtrader.strategies.base import BaseStrategy
import backtrader as bt

class MyStrategy(BaseStrategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('take_profit', 2.0),
        ('percent_sizer', 0.1),      # Use 10% of capital per trade
    )

    def __init__(self):
        super().__init__()           # Always call super().__init__()
        self.sma_fast = bt.indicators.SMA(self.data, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data, period=self.p.slow_period)

    def buy_or_short_condition(self):
        """Called each bar. Return True if you placed an order."""
        if not self.buy_executed and self.sma_fast[0] > self.sma_slow[0]:
            self.create_order('BUY')
            return True
        return False

    def dca_or_short_condition(self):
        """Called each bar when already in a position. For adding to position."""
        return False

    def sell_or_cover_condition(self):
        """Called each bar when in a position. Return True if you closed."""
        if self.buy_executed and self.sma_fast[0] < self.sma_slow[0]:
            for ot in self.active_orders[:]:
                self.close_order(ot)
            return True
        return False

    def check_stop_loss(self):
        """Custom stop loss logic. Return True if stop was hit."""
        if self.buy_executed and self.average_entry_price:
            if self.data.close[0] <= self.average_entry_price * 0.95:
                for ot in self.active_orders[:]:
                    self.close_order(ot)
                return True
        return False
```

### Key BaseStrategy Attributes

These are available inside your strategy:

| Attribute | Description |
|---|---|
| `self.buy_executed` | True if currently in a long position |
| `self.entry_prices` | List of entry prices for all open legs |
| `self.average_entry_price` | Weighted average of all entry prices |
| `self.first_entry_price` | Price of the first entry |
| `self.take_profit_price` | Calculated take profit target |
| `self.active_orders` | List of `OrderTracker` instances |
| `self.sizes` | List of position sizes per leg |
| `self.position_count` | Number of open order legs |
| `self.dataclose` | Reference to close price line |
| `self.p` / `self.params` | Access to strategy parameters |

### Creating and Closing Orders

```python
# Create a market buy order (auto-calculates size from percent_sizer)
order_tracker = self.create_order('BUY')

# Create with specific size and price
order_tracker = self.create_order('BUY', size=0.5, price=42000.0)

# Close a specific order
self.close_order(order_tracker)

# Close with specific exit price
self.close_order(order_tracker, exit_price=43000.0)
```

### Position Sizing

When `percent_sizer` is set (e.g., 0.1), each trade uses 10% of available cash:

```python
size = (available_cash * percent_sizer) / current_close_price
```

## Data Sources

### CCXT Data

Fetch from any CCXT exchange:

```python
from backtrader import get_crypto_data

data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')
# Parameters: (asset, start_date, end_date, time_resolution, exchange)
```

Supported timeframes depend on the exchange. Common ones: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`.

### SQL Server Data

Requires MSSQL with market data:

```python
from backtrader.feeds.mssql_crypto import get_database_data

df = get_database_data(
    ticker='BTC',
    start_date='2024-01-01',
    end_date='2024-01-31',
    time_resolution='1h',
    pair='USDT'
)
```

### Custom Data (Polars)

Pass a Polars DataFrame or use `PolarsFeed`:

```python
import polars as pl
from backtrader.feeds.polarfeed import PolarsFeed

df = pl.read_csv('my_data.csv')  # Must have OHLCV columns
data = PolarsFeed(df=df)
backtest(MyStrategy, data=data, init_cash=1000)
```

## Backtest Output

When you run a backtest, BTQuant prints:

```
==================================================
BACKTEST RESULTS - BTC/USDT
==================================================
Total Trades: 42
Winning Trades: 28
Losing Trades: 14
Win Rate: 66.7%
Net P&L: $156.32
Max Drawdown: 8.45%
Final Portfolio Value: $1156.32
Total P/L: $156.32
Return: 15.63%
==================================================
```

With `--plot` / `plot=True`: a candlestick chart with buy/sell arrows.

With `--quantstats` / `quantstats=True`: an HTML report saved to `QuantStats/` directory.

## Bulk Backtesting

Test a strategy across many coins in parallel:

### Python API

```python
from backtrader.utils.backtest import bulk_backtest

results = bulk_backtest(
    MyStrategy,
    coins=['BTC', 'ETH', 'ADA', 'SOL'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    interval='1h',
    init_cash=1000,
    max_workers=8
)
```

If `coins=None`, BTQuant auto-discovers all coins from the SQL Server database.

### CLI

```bash
btq bulk --strategy MyStrategy --interval 1h --workers 8
btq bulk --strategy MyStrategy --coins BTC,ETH --interval 15m --save
```

## Optimization

Optimize strategy parameters using Optuna:

### CLI

```bash
# Default parameter space
btq optimize --coin BTC --strategy VuManchCipher_A --trials 200

# Aggressive (more trades, higher risk)
btq optimize --coin BTC --strategy MyStrategy --trials 200 --aggressive

# Conservative (tighter drawdown control)
btq optimize --coin BTC --strategy MyStrategy --trials 200 --conservative

# Multiple coins (creates separate study per coin)
btq optimize --coins BTC,ETH,DOGE --strategy MyStrategy --trials 100

# Custom study name
btq optimize --coin BTC --strategy MyStrategy --study-name my_study --trials 150
```

### Optimization Options

```
--trials / -n          Number of Optuna trials (default: 200)
--opt-workers          Parallel optimization workers
--aggressive           Use aggressive param space (if strategy defines param_space_aggressive)
--conservative         Use conservative param space (if strategy defines param_space_conservative)
--min-trades           Minimum trades for valid result (default: 30)
--pruner               Pruner algorithm: hyperband, median, none (default: hyperband)
--seed                 Random seed (default: 42)
--multi-period         Run multi-period validation
```

## Live Trading (Experimental)

Live trading supports PancakeSwap (Web3/BSC) and JackRabbitRelay exchanges.

### Live Trading Setup

```python
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK
from backtrader import livetrading

ccxt_config = {
    'apiKey': '',
    'secret': '',
    'enableRateLimit': True,
    'rateLimit': 20,
    'options': {'defaultType': 'spot'}
}

livetrading.livetrade(
    coin='XRP',
    collateral='USDT',
    strategy=NRK,
    asset='XRP/USDT',
    exchange='mexc',
    account='',
    config=ccxt_config
)
```

### BaseStrategy Live Trading Features

When `backtest=False`, BaseStrategy:
- Initializes PancakeSwap Web3 order queue (if exchange is "pancakeswap")
- Initializes JackRabbitRelay broker (if exchange is "mimic")
- Can load existing positions from CSV or exchange API
- Supports Telegram and Discord alert notifications

The `btq live` CLI mode is not yet implemented.

## Multi-Timeframe Resampling

The `backtest()` function supports adding resampled timeframes:

```python
backtest(MyStrategy,
         coin='BTC',
         start_date='2024-01-01',
         end_date='2024-06-01',
         interval='1m',
         add_mtf_resamples=True)  # Adds 5m, 15m, 60m resamples
```

## Caching

Data fetched from SQL Server is cached as Parquet files in `.btq_cache/`:

- Cache key is derived from symbol, interval, collateral, and date range
- Parquet files use zstd compression
- Disable with `--no-cache` flag
- Clear with `--clear-cache` flag
- Custom cache directory via `BTQ_CACHE_DIR` environment variable

## Backtest Function Reference

```python
backtest(
    strategy,             # Strategy class (required)
    data=None,            # Pre-loaded data (DataFrame or Backtrader feed)
    coin=None,            # Coin symbol (e.g., 'BTC')
    start_date="1970-01-01",
    end_date="2030-12-31",
    interval=None,        # Timeframe (e.g., '1h')
    collateral="USDT",
    commission=0.00075,   # Commission rate
    init_cash=100000.0,   # Starting capital
    plot=False,           # Show chart
    quantstats=False,     # Generate QuantStats report
    asset_name=None,      # Display name
    bulk=False,           # Bulk mode flag
    exchange=None,        # Exchange name (e.g., 'mexc' for shorting)
    slippage_bps=5,       # Slippage in basis points
    params=None,          # Dict of strategy parameters
    add_mtf_resamples=False,  # Add multi-timeframe resamples
    **kwargs              # Additional strategy parameters
)
```

Returns the final portfolio value as a float.