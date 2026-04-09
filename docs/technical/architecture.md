# BTQuant Technical Architecture

## Overview

BTQuant is a professional backtesting and live trading framework built on top of a modified Backtrader engine. It provides multiple data sources, broker integrations, and strategy execution modes for cryptocurrency trading.

## Core Components

### 1. Package Structure

```
dependencies/backtrader/
├── __init__.py          # Package exports and module registration
├── btquant.py           # CLI interface (btq command)
├── cerebro.py           # Core backtesting engine
├── strategy.py          # Strategy base class
├── feed.py              # Data feed base classes
├── broker.py            # Broker base classes
├── order.py             # Order management
├── position.py          # Position tracking
├── indicator.py         # Indicator base class
├── analyzer.py          # Performance analyzers
├── strategies/          # Trading strategies
├── feeds/               # Data feed implementations
├── brokers/             # Broker implementations
├── stores/              # Exchange store implementations
├── indicators/          # Technical indicators
├── hotspine/            # Shared memory market data system
├── bigbraincentral/     # SQL Server data storage
├── utils/               # Utility functions
└── config/              # Configuration management
```

### 2. BaseStrategy Class

Located in `dependencies/backtrader/strategies/base.py`, the `BaseStrategy` class extends `bt.Strategy` with:

**Parameters:**
```python
params = (
    ('init_cash', 1000.0),       # float - Initial capital
    ('exchange', None),          # str - Exchange name
    ('account', None),           # str - Account identifier
    ('asset', None),             # str - Asset symbol
    ('amount', None),            # float - Trade amount
    ('coin', None),              # str - Coin identifier
    ('collateral', None),        # str - Collateral currency
    ('debug', False),            # bool - Debug output flag
    ('capture_data', False),     # bool - Indicator transparency capture
    ('backtest', True),          # bool - Backtest mode flag
    ('bulk', False),             # bool - Bulk operation flag
    ('optuna', False),           # bool - Optuna optimization flag
    ('quantstats', None),        # QuantStats config
    ('use_stoploss', None),      # bool - Enable stop loss
    ('pnl', None),               # float - P&L tracking
    ('final_value', None),       # float - Final portfolio value
    ('channel', ""),             # str - Alert channel
    ('symbol', ""),              # str - Trading symbol
    ('stop_loss', 0),            # float - Stop loss percentage
    ('stop_trail', 0),           # float - Trailing stop percentage
    ('take_profit', 0),          # float - Take profit percentage
    ('percent_sizer', 0),        # float - Position size as % of capital
    ('order_cooldown', 0),       # float - Order cooldown in seconds
    ('enable_alerts', False),    # bool - Enable alert system
    ('alert_channel', None),     # str - Alert channel ID
)
```

**Strategy Interface Methods (Override in subclasses):**
```python
def buy_or_short_condition(self) -> bool:
    """Return True if buy/short order was placed"""

def dca_or_short_condition(self) -> bool:
    """Return True if DCA order was placed"""

def sell_or_cover_condition(self) -> bool:
    """Return True if sell/cover order was placed"""

def check_stop_loss(self) -> bool:
    """Return True if stop loss was triggered"""
```

**Helper Methods:**
```python
def create_order(self, action='BUY', size=None, price=None) -> OrderTracker:
    """Create order with automatic tracking"""

def close_order(self, order_tracker, exit_price=None) -> None:
    """Close specific order and update tracking"""

def calc_averages(self) -> None:
    """Calculate average entry price and take profit"""

def reset_position_state(self) -> None:
    """Reset all position tracking after full exit"""
```

### 3. OrderTracker Class

Internal order tracking structure used by BaseStrategy:

```python
class OrderTracker:
    entry_price: float       # Entry price
    size: float              # Position size
    take_profit_pct: float   # Take profit percentage
    symbol: str              # Trading symbol
    order_type: str          # 'BUY' or 'SELL'
    backtest: bool           # Backtest mode flag
    bulk: bool               # Bulk operation flag
    optuna: bool             # Optimization flag
    data_datetime: datetime  # Order timestamp
```

### 4. Data Feed Architecture

**Available Feeds:**

| Feed | Module | Source |
|------|--------|--------|
| CCXT | `feeds/ccxt.py` | CCXT library exchanges |
| HotSpine | `feeds/hotspine_feed.py` | Shared memory (live) |
| Binance | `feeds/binance_feed.py` | Binance Store |
| Bitget | `feeds/bitget_feed.py` | Bitget Store |
| MEXC | `feeds/mexc_feed.py` | MEXC Store |
| PancakeSwap | `feeds/pancakeswap_feed.py` | Web3/DEX |
| DatabaseOHLCV | `feeds/db_ohlcv_mssql.py` | SQL Server |
| Yahoo | `feeds/yahoo.py` | Yahoo Finance |
| CSV | `feeds/csvgeneric.py` | CSV files |
| Pandas | `feeds/pandafeed.py` | Pandas DataFrame |

**Data Feed Base Class Methods:**
```python
class DataBase:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def _load(self) -> bool: ...     # Load next bar
    def islive(self) -> bool: ...    # True for live feeds
    def haslivedata(self) -> bool: ...
    def qcheck(self, limit) -> bool: ...
```

### 5. Broker Architecture

**Available Brokers:**

| Broker | Module | Description |
|--------|--------|-------------|
| JrrBroker | `brokers/jrrbroker.py` | JackRabbitRelay webhook broker |
| CCXTBroker | `brokers/ccxtbroker.py` | Direct CCXT exchange broker |
| PancakeSwap | `brokers/pancakeswap_orders.py` | Web3 DEX orders |
| IB | `brokers/ibbroker.py` | Interactive Brokers |
| Oanda | `brokers/oandabroker.py` | Oanda forex |

**JrrBroker Parameters:**
```python
params = (
    ('cash', 10000.0),       # float - Starting cash
    ('exchange', 'mimic'),   # str - Exchange identifier
    ('account', 'default'),  # str - Account name
    ('debug', True),         # bool - Debug output
)
```

**CCXTBroker Methods:**
```python
def buy(owner, data, size, price=None, ...) -> Order:
def sell(owner, data, size, price=None, ...) -> Order:
def cancel(order) -> bool:
def get_orders_open(safe=False) -> List[Order]:
def get_order_status(order) -> str:
def check_orders() -> None:
def set_initial_position(data, size) -> Order:
def load_initial_positions(data) -> List[Order]:
def fetch_trades_history(symbol, since=None, limit=100) -> List:
```

### 6. Live Trading Functions

Located in `dependencies/backtrader/livetrading.py`:

```python
def livetrade_ccxt(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy_class: str,
    config: Optional[Dict[str, Any]] = None,
) -> None: ...

def livetrade_web3(
    coin: str,
    collateral: str,
    web3ws: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    timezone: str = 'Europe/Berlin',
    start_hours_ago: int = 2,
    enable_alerts: bool = False,
) -> None: ...

def livetrade_hotspine(
    symbol_id: int,
    strategy_class,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001,
    **strategy_params
) -> None: ...

def livetrade_hotspine_multi_symbol(
    symbol_ids: list,
    strategy,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001,
    **strategy_params
) -> None: ...

def livetrade_binance(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = "",
) -> None: ...

def livetrade_mexc(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = "",
) -> None: ...

def livetrade_bitget(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = "",
) -> None: ...
```

### 7. CLI Interface (btq)

The CLI is defined in `dependencies/backtrader/btquant.py`:

```bash
# Modes
btq backtest --coin BTC --strategy MyStrat --interval 15m
btq bulk --strategy MyStrat --interval 1h --workers 8
btq optimize --coin BTC --strategy MyStrat --trials 200
btq list strategies
btq list coins --collateral USDT
```

**CLI Arguments:**
```
Mode: backtest | bulk | optimize | list | live

Common:
  --coin, --symbol        Single coin (e.g., BTC)
  --coins, --symbols      Comma-separated coins (BTC,ETH,BNB)
  --strategy, --strat     Strategy class name
  --collateral            Collateral currency (default: USDT)
  --interval, --tf        Timeframe: 1m, 5m, 15m, 1h, 4h, 1d
  --start                 Start date YYYY-MM-DD
  --end                   End date YYYY-MM-DD (default: 2025-01-01)

Capital & Risk:
  --cash                  Initial capital (default: 1000)
  --commission            Commission rate (default: 0.00075)
  --leverage              Leverage multiplier (default: 1)
  --slippage              Slippage in basis points (default: 5.0)

Output:
  --plot, -p              Show plot
  --quantstats, -q        Generate QuantStats report
  --debug, -d             Debug output
  --save                  Save results to file

Optimization:
  --trials                Optimization trials (default: 200)
  --aggressive            Aggressive parameter space
  --conservative          Conservative parameter space
  --pruner                hyperband | median | none
  --min-trades            Minimum trades (default: 30)
```

### 8. TransparencyPatch

The `TransparencyPatch` module enables indicator chain transparency for debugging:

```python
from backtrader import transparencypatch

patch = transparencypatch.TransparencyPatch()
patch.debug = True
patch.apply_indicator_patch()

# In strategy:
patch.capture_patch_fast(strategy)
```

## Data Flow

```
[Exchange API] --> [Store] --> [Feed] --> [Cerebro] --> [Strategy]
                                                      |
[Broker] <-- [Order] <-- [Strategy.next()]

HotSpine Flow:
[C++ Collector] --> [Shared Memory] --> [HotSpineReader] --> [HotSpineData] --> [Strategy]

SQL Flow:
[Exchange] --> [Collector] --> [MarketDataStorage] --> [SQL Server]
                                                        |
[ReadOnlyOHLCV] <-- [DatabaseOHLCVData] <-- [Strategy]
```

## Module Dependencies

```
backtrader (core engine)
├── ccxt (exchange connectivity)
├── fast_mssql (C++ SQL driver)
├── telethon (Telegram alerts)
├── requests (HTTP/webhooks)
├── optuna (optimization)
├── quantstats (reporting)
├── colorama (terminal colors)
├── rich (CLI formatting)
└── pytz (timezone handling)
```
