# BTQuant API Reference

## Overview

This document provides comprehensive API reference documentation for BTQuant modules and classes. It covers the core components, utilities, and extension points available in the framework.

## Core Modules

### backtrader Package

#### Main Classes

##### `bt.Cerebro`
The main engine for running backtests and live trading.

```python
class Cerebro(
    runonce=True,
    maxcpus=None,
    stdstats=True,
    oldbuysell=False,
    oldtrades=False,
    preload=True,
    runnext=None,
    exactbars=False,
    optdatas=True,
    optreturn=True,
    oldsync=False,
    timeout=None,
    tradehistory=False,
    strict=False,
    writer=None,
    writercsv=False,
    csvwriter=False,
    replaying=False,
    quicknotify=False,
    live=False
)
```

**Methods:**
- `addstrategy(strategy, *args, **kwargs)` - Add a strategy to the engine
- `adddata(data, name=None)` - Add data feed to the engine
- `addobserver(observer, *args, **kwargs)` - Add observer for monitoring
- `addanalyzer(analyzer, *args, **kwargs)` - Add performance analyzer
- `run(stop=False, runonce=None, preload=None, oldsync=None, exactbars=None, stdstats=None, writer=None, tradehistory=None, **kwargs)` - Run the backtest
- `plot(plotter=None, numfigs=1, iplot=True, start=None, end=None, width=16, height=9, dpi=300, tight=True, use=None, **kwargs)` - Plot results

**Example:**
```python
cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)
cerebro.adddata(data)
results = cerebro.run()
cerebro.plot()
```

##### `bt.Strategy`
Base class for all trading strategies.

```python
class Strategy(
    strategyname=None,
    parent=None,
    params=None,
    **kwargs
)
```

**Key Methods:**
- `__init__()` - Initialize strategy and indicators
- `next()` - Called for each new bar of data
- `notify_order(order)` - Handle order notifications
- `notify_trade(trade)` - Handle trade notifications
- `notify_data(data, status, *args, **kwargs)` - Handle data status changes
- `stop()` - Called when strategy stops

**Example:**
```python
class MyStrategy(bt.Strategy):
    params = (('period', 20),)
    
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=self.params.period)
    
    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.data.close[0] < self.sma[0]:
            self.sell()
```

##### `bt.DataBase`
Base class for all data feeds.

**Key Methods:**
- `start()` - Start data feed
- `stop()` - Stop data feed
- `islive()` - Check if data feed is live
- `prenext()` - Called before next() when not enough data
- `nextstart()` - Called when enough data is available

### backtrader.strategies.base Module

#### BaseStrategy Class

The `BaseStrategy` provides enhanced functionality over the standard `bt.Strategy`.

```python
class BaseStrategy(bt.Strategy):
    params = (
        ('init_cash', 1000.0),
        ('exchange', None),
        ('account', None),
        ('asset', None),
        ('amount', None),
        ('coin', None),
        ('collateral', None),
        ('debug', False),
        ('capture_data', False),
        ('backtest', True),
        ('bulk', False),
        ('optuna', False),
        ('quantstats', None),
        ('use_stoploss', None),
        ('pnl', None),
        ('final_value', None),
        ('channel', ''),
        ('symbol', ''),
        ('stop_loss', 0),
        ('stop_trail', 0),
        ('take_profit', 0),
        ('percent_sizer', 0),
        ('order_cooldown', 0),
        ('enable_alerts', False),
        ('alert_channel', None)
    )
```

**Key Methods:**

##### Position Management
- `create_order(action='BUY', size=None, price=None)` - Create buy/sell order
- `close_order(order_tracker, exit_price=None)` - Close specific order
- `close_all_positions()` - Close all active positions
- `reset_position_state()` - Reset position tracking state

##### Strategy Logic (Override in subclasses)
- `buy_or_short_condition()` - Implement buy/short entry logic
- `dca_or_short_condition()` - Implement DCA logic
- `sell_or_cover_condition()` - Implement sell/cover exit logic
- `check_stop_loss()` - Implement custom stop loss logic

##### Live Trading
- `init_live_trading()` - Initialize live trading components
- `send_alert(message)` - Send alert message if enabled

**Example:**
```python
class MyAdvancedStrategy(BaseStrategy):
    params = (('take_profit', 2.0), ('stop_loss', 1.0))
    
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=20)
    
    def buy_or_short_condition(self):
        if not self.buy_executed and self.data.close[0] > self.sma[0]:
            self.create_order('BUY')
            return True
        return False
    
    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] < self.sma[0]:
            self.close_all_positions()
            return True
        return False
```

#### OrderTracker Class

Tracks individual orders with automatic CSV persistence.

```python
class OrderTracker:
    def __init__(
        self,
        entry_price,
        size,
        take_profit_pct,
        symbol=None,
        order_type="BUY",
        backtest=False,
        bulk=False,
        optuna=False,
        data_datetime=None
    )
```

**Key Methods:**
- `close_order(exit_price, exit_datetime=None)` - Close the order
- `save_to_csv()` - Save order to CSV file
- `update_csv()` - Update existing CSV record
- `load_active_orders_from_csv(symbol, backtest=False, bulk=False, optuna=False)` - Load active orders from CSV

### backtrader.utils Module

#### backtest Module

##### `backtest()` Function
Main backtesting function with comprehensive features.

```python
def backtest(
    strategy,
    data=None,
    coin=None,
    start_date="1970-01-01",
    end_date="2030-12-31",
    interval=None,
    collateral="USDT",
    commission=0.00075,
    init_cash=100000.0,
    plot=False,
    quantstats=False,
    asset_name=None,
    bulk=False,
    show_progress=True,
    exchange=None,
    slippage_bps=5,
    min_qty=0.0,
    qty_step=1.0,
    price_tick=None,
    params=None,
    add_mtf_resamples=False,
    **kwargs
)
```

**Parameters:**
- `strategy` - Strategy class to backtest
- `data` - Data feed or None to fetch automatically
- `coin` - Coin symbol (e.g., 'BTC')
- `start_date` - Start date for backtest
- `end_date` - End date for backtest
- `interval` - Timeframe (e.g., '1m', '1h', '1d')
- `collateral` - Quote currency (e.g., 'USDT')
- `commission` - Trading commission rate
- `init_cash` - Initial capital
- `plot` - Generate plot
- `quantstats` - Generate QuantStats report
- `bulk` - Bulk mode (disables CSV persistence)
- `show_progress` - Show progress bar
- `params` - Strategy parameters

**Returns:**
- Final portfolio value

##### `bulk_backtest()` Function
Run backtests on multiple assets in parallel.

```python
def bulk_backtest(
    strategy,
    coins=None,
    start_date="1970-01-01",
    end_date="2030-01-01",
    interval=None,
    collateral="USDT",
    init_cash=1000,
    max_workers=8,
    save_results=True,
    output_file='backtest_results.json',
    params_mode="mtf",
    **backtest_kwargs
)
```

**Parameters:**
- `coins` - List of coin symbols or None for auto-discovery
- `max_workers` - Number of parallel workers
- `save_results` - Save results to JSON file
- `output_file` - Output filename

**Returns:**
- List of backtest results

##### `optimize_backtest()` Function
Run parameter optimization using multiprocessing.

```python
def optimize_backtest(
    strategy,
    data,
    init_cash=1000,
    max_workers=4,
    show_progress=True,
    **param_ranges
)
```

**Parameters:**
- `param_ranges` - Parameter ranges for optimization (e.g., `fast_period=[10, 20, 30]`)

**Returns:**
- List of optimization results

#### ccxt_data Module

##### `get_crypto_data()` Function
Fetch cryptocurrency data from CCXT exchanges.

```python
def get_crypto_data(
    asset,
    start_date,
    end_date,
    timeframe,
    exchange,
    retries=3,
    retry_delay=1
)
```

**Parameters:**
- `asset` - Trading pair (e.g., 'BTC/USDT')
- `start_date` - Start date
- `end_date` - End date
- `timeframe` - Timeframe (e.g., '1m', '1h', '1d')
- `exchange` - Exchange name (e.g., 'binance', 'bybit')

**Returns:**
- Polars DataFrame with OHLCV data

### backtrader.feeds Module

#### Data Feed Classes

##### `MSSQLData`
SQL Server data feed for high-performance data access.

```python
class MSSQLData(bt.feeds.DataBase):
    params = (
        ('connection_string', None),
        ('symbol', None),
        ('start_date', None),
        ('end_date', None),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),
        ('datetime', 'TimestampStart'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None)
    )
```

**Methods:**
- `get_all_pairs(connection_string)` - Get all available trading pairs
- `get_database_data()` - Get data from database

##### `PolarsData`
Polars DataFrame data feed.

```python
class PolarsData(bt.feeds.DataBase):
    params = (
        ('dataname', None),
        ('datetime', 'datetime'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None)
    )
```

##### Exchange-Specific Feeds
- `BinanceFeed` - Binance WebSocket feed
- `BitgetFeed` - Bitget WebSocket feed
- `MEXCFeed` - MEXC WebSocket feed
- `PancakeSwapFeed` - PancakeSwap Web3 feed

### backtrader.analyzers Module

#### Performance Analyzers

##### `TimeReturn`
Calculate time-based returns.

```python
class TimeReturn(bt.Analyzer):
    params = (('timeframe', bt.TimeFrame.NoTimeFrame),)
```

##### `SharpeRatio`
Calculate Sharpe ratio.

```python
class SharpeRatio(bt.Analyzer):
    params = (('timeframe', bt.TimeFrame.NoTimeFrame), ('riskfreerate', 0.01))
```

##### `DrawDown`
Calculate drawdown statistics.

```python
class DrawDown(bt.Analyzer):
    params = ()
```

##### `TradeAnalyzer`
Analyze individual trades.

```python
class TradeAnalyzer(bt.Analyzer):
    params = ()
```

##### `CustomSQN`
Custom SQN (System Quality Number) analyzer.

```python
class CustomSQN(bt.Analyzer):
    params = (('compressionFactor', 1e6),)
```

### backtrader.observers Module

#### Observer Classes

##### `BuySellArrows`
Display buy/sell arrows on plots.

```python
class BuySellArrows(bt.observers.BuySell):
    plotlines = dict(
        buy=dict(marker='$⇧$', markersize=8.0),
        sell=dict(marker='$⇩$', markersize=8.0)
    )
```

##### `Value`
Track portfolio value.

##### `DrawDown`
Track drawdown.

##### `Cash`
Track cash balance.

### backtrader.brokers Module

#### Broker Classes

##### `JrrOrderBase`
JackRabbitRelay broker for live trading.

```python
class JrrOrderBase(bt.brokers.BrokerBase):
    def __init__(self, alert_manager=None):
        super().__init__()
        self.alert_manager = alert_manager
```

##### `PancakeSwapV2DirectOrderBase`
PancakeSwap Web3 broker.

```python
class PancakeSwapV2DirectOrderBase(bt.brokers.BrokerBase):
    def __init__(self, coin, collateral):
        super().__init__()
        self.coin = coin
        self.collateral = collateral
```

### backtrader.stores Module

#### Store Classes

##### `BinanceStore`
Binance exchange store.

```python
class BinanceStore(bt.stores.Store):
    params = (
        ('coin', None),
        ('collateral', None),
        ('config', None),
    )
```

##### `BitgetStore`
Bitget exchange store.

##### `MEXCStore`
MEXC exchange store.

##### `PancakeSwapStore`
PancakeSwap Web3 store.

### backtrader.indicators Module

#### Built-in Indicators

BTQuant includes all standard Backtrader indicators plus custom ones:

##### Moving Averages
- `SimpleMovingAverage` - Simple moving average
- `ExponentialMovingAverage` - Exponential moving average
- `WeightedMovingAverage` - Weighted moving average
- `BollingerBands` - Bollinger Bands

##### Oscillators
- `RelativeStrengthIndex` - RSI
- `StochasticSlow` - Stochastic oscillator
- `MACD` - MACD indicator

##### Volume Indicators
- `OnBalanceVolume` - On-balance volume
- `VolumeWeightedAveragePrice` - VWAP

### backtrader.comminfo Module

#### CommissionInfo
Commission and slippage configuration.

```python
class CommissionInfo(bt.CommInfoBase):
    params = (
        ('commission', 0.00075),
        ('mult', 1.0),
        ('margin', None),
        ('automargin', False),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
        ('percabs', False),
        ('interest', 0.0),
        ('interest_long', False),
        ('leverage', 1.0),
        ('margin_interest', 0.0),
        ('slippage', 0.0),
    )
```

## Utility Functions

### Configuration Utilities

```python
# Configuration validation
def validate_configuration(config):
    """Validate configuration settings"""
    pass

def test_configuration():
    """Test configuration with sample data"""
    pass
```

### Data Utilities

```python
# Data loading with caching
class PolarsDataLoader:
    def load_data(self, spec: DataSpec, use_cache: bool = True) -> pl.DataFrame:
        """Load data with caching support"""
        pass
    
    def make_backtrader_feed(self, df: pl.DataFrame, spec: DataSpec):
        """Convert Polars DataFrame to Backtrader feed"""
        pass
```

### Strategy Utilities

```python
# Strategy parameter extraction
def _strategy_param_keys(cls):
    """Extract parameter keys from strategy class"""
    pass

# Progress tracking
def create_progress_bar(total, description):
    """Create rich progress bar"""
    pass
```

## Error Handling

### Common Exceptions

```python
class BTQuantError(Exception):
    """Base exception for BTQuant errors"""
    pass

class ConfigurationError(BTQuantError):
    """Configuration-related errors"""
    pass

class DataError(BTQuantError):
    """Data-related errors"""
    pass

class StrategyError(BTQuantError):
    """Strategy-related errors"""
    pass
```

### Error Handling Patterns

```python
try:
    # BTQuant operations
    result = backtest(strategy, data=data)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except DataError as e:
    print(f"Data error: {e}")
except StrategyError as e:
    print(f"Strategy error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### Import Organization
```python
# Standard library imports
import os
import sys
import logging

# Third-party imports
import backtrader as bt
import polars as pl
import pandas as pd

# BTQuant imports
from backtrader.strategies.base import BaseStrategy
from backtrader.utils.backtest import backtest
from backtrader.feeds.mssql_crypto import get_database_data
```

### Class Inheritance
```python
# Always inherit from BaseStrategy for new strategies
class MyStrategy(BaseStrategy):
    # Define parameters
    params = (
        ('period', 20),
        ('threshold', 0.02),
    )
    
    # Initialize indicators
    def __init__(self):
        super().__init__()
        # Strategy initialization
    
    # Implement strategy logic
    def buy_or_short_condition(self):
        # Entry logic
        pass
    
    def sell_or_cover_condition(self):
        # Exit logic
        pass
```

### Error Handling
```python
def safe_backtest(strategy, data, **kwargs):
    """Run backtest with comprehensive error handling"""
    try:
        return backtest(strategy, data=data, **kwargs)
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        raise BTQuantError(f"Backtest failed: {e}")
```

### Configuration Management
```python
def load_configuration(env='development'):
    """Load configuration based on environment"""
    if env == 'production':
        return load_production_config()
    elif env == 'development':
        return load_development_config()
    else:
        raise ConfigurationError(f"Unknown environment: {env}")
```

This API reference provides comprehensive documentation for all major BTQuant components. For more detailed information about specific classes or methods, refer to the source code comments and docstrings.