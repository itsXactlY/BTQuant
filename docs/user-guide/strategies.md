# Strategy Development Guide

BTQuant strategies extend `BaseStrategy` which itself extends `bt.Strategy`.
Every strategy overrides a small set of hook methods; the base class handles
order lifecycle, DCA tracking, position sizing, CSV persistence, live-trading
bootstrap, and reporting.

## BaseStrategy

```python
from backtrader.strategies.base import BaseStrategy
```

### Params (all inherited by every strategy)

```python
params = (
    ('init_cash',      1000.0),
    ('exchange',       None),
    ('account',        None),
    ('asset',          None),
    ('amount',         None),
    ('coin',           None),
    ('collateral',     None),
    ('debug',          False),
    ('capture_data',   False),   # Indicator TransparencyPatch
    ('backtest',       True),
    ('bulk',           False),   # set True for bulk backtests
    ('optuna',         False),   # set True during Optuna runs
    ('quantstats',     None),
    ('use_stoploss',   None),
    ('pnl',            None),
    ('final_value',    None),
    ('channel',        ''),
    ('symbol',         ''),
    ('stop_loss',      0),
    ('stop_trail',     0),
    ('take_profit',    0),
    ('percent_sizer',  0),       # fraction of cash per order (e.g. 0.05 = 5%)
    ('order_cooldown', 0),
    ('enable_alerts',  False),
    ('alert_channel',  None),
)
```

### Override hooks

| Method | Return | Purpose |
|---|---|---|
| `buy_or_short_condition()` | bool | Entry signal. Return True if an order was placed. |
| `dca_or_short_condition()` | bool | DCA / additional-entry logic. |
| `sell_or_cover_condition()` | bool | Exit signal. |
| `check_stop_loss()` | bool | Custom stop-loss logic. |

The base `next()` calls `_execute_strategy_logic()` which automatically
sequences these hooks:

1. If flat -> `buy_or_short_condition()`
2. If in position and DCA enabled -> `sell_or_cover_condition()` first, then
   `dca_or_short_condition()`
3. If in position without DCA -> `sell_or_cover_condition()`

### Key helper methods

| Method | Signature | Description |
|---|---|---|
| `create_order` | `(action='BUY', size=None, price=None)` | Creates an OrderTracker, appends to `active_orders`, places a market order. Auto-sizes if size is None. Returns `OrderTracker`. |
| `close_order` | `(order_tracker, exit_price=None)` | Closes one specific OrderTracker, removes it from `active_orders`, executes sell. |
| `calc_averages` | `()` | Recomputes `average_entry_price` and `take_profit_price` from `active_orders`. |
| `reset_position_state` | `()` | Resets all long tracking vars. Called automatically after full exit. |
| `_determine_size` | `()` | Returns position size: backtest uses `broker.get_cash() * percent_sizer / close`; live calls `calculate_position_size()`. |

### Position tracking attributes (auto-initialized)

`entry_price`, `entry_prices`, `sizes`, `first_entry_price`,
`average_entry_price`, `take_profit_price`, `stop_loss_price`,
`buy_executed`, `DCA`, `active_orders`, `position_count`, `stake`,
`short_entry_prices`, `short_sizes`, `average_short_price`, etc.

### Live trading

When `backtest=False`, `__init__` calls `init_live_trading()` which detects:

- **PancakeSwap** (`exchange='pancakeswap'`): initializes
  `PancakeSwapV2DirectOrderBase`, starts a Web3 order queue thread.
- **Mimic / JRR** (`exchange='mimic'`): uses `JrrBroker` for live orders
  through JackRabbitRelay.
- **Alerts**: set `enable_alerts=True` for Telegram/Discord notifications via
  `send_alert(message)`.

## OrderTracker

```python
from backtrader.strategies.base import OrderTracker
```

Tracks individual positions with automatic CSV persistence.

```python
OrderTracker(
    entry_price,          # float
    size,                 # float
    take_profit_pct,      # float  (e.g. 2.0 means 2%)
    symbol=None,          # str
    order_type="BUY",     # "BUY" or "SELL"
    backtest=False,       # disables CSV when True
    bulk=False,           # disables CSV when True
    optuna=False,         # disables CSV when True
    data_datetime=None,   # datetime for the bar
)
```

Key attributes: `entry_price`, `size`, `take_profit_price`,
`executed`, `closed`, `exit_price`, `profit_pct`, `timestamp`,
`tracker_id`, `symbol`, `order_type`.

CSV files are stored in `<venv_root>/.OrderTracker_Live/<symbol>_order_tracker.csv`.
CSV is automatically disabled during backtest, bulk, and optuna runs.

Class methods:
- `OrderTracker.load_active_orders_from_csv(symbol)` - loads open orders
- `OrderTracker.disable_persistence()` / `enable_persistence()`
- `OrderTracker.set_base_dir(path)` / `get_base_dir()`

## Available Strategies (20 files)

| Strategy Class | File | Description |
|---|---|---|
| `SMA_Cross_Simple` | `SMA_Cross_Simple.py` | SMA crossover with stop-loss / take-profit |
| `NRK` | `NearestNeighbors_RationalQuadraticKernel.py` | KNN + Rational Quadratic Kernel ML strategy |
| `Aligator_supertrend` | `Aligator_supertrend.py` | Williams Alligator + SuperTrend |
| `SMA_Cross_MESAdaptive_Prime` | `SMA_Cross_MESAdaptive_Prime.py` | SMA cross with MESA adaptive MA |
| `MACD_ADX` | `MACD_ADX.py` | MACD + ADX trend strength |
| `ST_RSX_ASI` | `ST_RSX_ASI.py` | SuperTrend + RSX + Accumulative Swing Index |
| `QQE_Hullband_VolumeOsc` | `QQE_Hullband_VolumeOsc.py` | QQE + Hull MA band + Volume Oscillator |
| `Vumanchu_A` | `Vumanchu_A.py` | VuManChu Market Cipher A |
| `Vumanchu_B` | `Vumanchu_B.py` | VuManChu Market Cipher B |
| `SuperTrend_Scalp` | `SuperTrend_Scalp.py` | SuperTrend scalping |
| `SineWeightZeroLagQQEVolMesaAdaptive` | `SineWeightZeroLagQQEVolMesaAdaptive.py` | Multi-indicator adaptive |
| `StagedConvergenceStrategy` | `StagedConvergenceStrategy.py` | Staged convergence |
| `OrderChain` | `OrderChain.py` | Order chain indicator strategy |
| `Order_Chain_Kioseff_Trading` | `Order_Chain_Kioseff_Trading.py` | Kioseff order chain |
| `pancakeswap_dca_marketmaker` | `pancakeswap_dca_marketmaker.py` | PancakeSwap DCA market maker |
| `pancakeswap_orders` | `pancakeswap_orders.py` | PancakeSwap order management |
| `jrr_orders` | `jrr_orders.py` | JackRabbitRelay order management |

Use `btq list strategies` to see the current list at runtime.

## Creating a Custom Strategy

### Step 1: Copy the template

The file `__TEMPLATE__.py` shows the skeleton:

```python
from backtrader.strategies.base import BaseStrategy, OrderTracker
from datetime import datetime
import backtrader as bt

class MyStrategy(BaseStrategy):
    params = (
        ('dca_deviation', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.05),   # 5% of cash per order
        ('debug', False),
        ('backtest', None),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buy_executed = False
        self.DCA = True

    def buy_or_short_condition(self):
        if not self.buy_executed:
            if self.some_indicator[0] > self.some_threshold:
                self.create_order(action='BUY')
                return True
        return False

    def dca_or_short_condition(self):
        if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * 0.985:
            self.create_order(action='BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        for order_tracker in list(self.active_orders):
            if self.data.close[0] >= order_tracker.take_profit_price:
                self.close_order(order_tracker)
                return True
        return False
```

### Step 2: Example -- SMA_Cross_Simple

```python
from backtrader.strategies.base import BaseStrategy, bt

class SMA_Cross_Simple(BaseStrategy):
    params = (
        ('short_period', 10),
        ('long_period', 30),
        ('stop_loss_pct', 0.02),
        ('take_profit_pct', 0.05),
        ('size', 0.1),
        ('debug', False),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sma_short = bt.ind.SMA(period=self.p.short_period)
        self.sma_long = bt.ind.SMA(period=self.p.long_period)
        self.crossover = bt.ind.CrossOver(self.sma_short, self.sma_long)
        self.in_position = False

    def buy_or_short_condition(self):
        if self.crossover > 0 and not self.in_position:
            self.create_order(action='BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        if not self.in_position:
            return False
        current_price = self.data.close[0]
        if self.entry_price > 0:
            drawdown = (current_price - self.entry_price) / self.entry_price
            if drawdown <= -self.p.stop_loss_pct:
                self.create_order(action='SELL')
                return True
            if drawdown >= self.p.take_profit_pct:
                self.create_order(action='SELL')
                return True
        if self.crossover < 0:
            self.create_order(action='SELL')
            return True
        return False

    def next(self):
        super().next()
        if self.buy_or_short_condition():
            return
        if self.sell_or_cover_condition():
            return
```

### Step 3: ML example -- NRK (NearestNeighbors + RationalQuadraticKernel)

See `NearestNeighbors_RationalQuadraticKernel.py` for a full example using
`sklearn.neighbors.NearestNeighbors`, a custom `RationalQuadraticKernel`
indicator, and features from RSI, Williams %R, CCI, and ADX.

## Running Strategies

### CLI

```bash
# Single backtest
btq backtest --coin BTC --strategy SMA_Cross_Simple --interval 15m --plot

# Multiple coins
btq backtest --coins BTC,ETH --strategy MACD_ADX --interval 1h --start 2024-01-01

# Bulk (all coins)
btq bulk --interval 1h --workers 8 --strategy Aligator_supertrend

# Optimization
btq optimize --coin BTC --strategy QQE_Hullband_VolumeOsc --trials 200 --aggressive
```

### Python API

```python
from backtrader.utils.backtest import backtest
from backtrader.strategies.SMA_Cross_Simple import SMA_Cross_Simple

final_value = backtest(
    SMA_Cross_Simple,
    coin='BTC',
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1h',
    init_cash=10000,
    commission=0.00075,
    plot=True,
    quantstats=True,
    params={'short_period': 10, 'long_period': 30},
)
```

## Best Practices

1. Always call `super().__init__(**kwargs)` first.
2. Use `self.create_order()` instead of `self.buy()` / `self.sell()` directly.
3. Use `self.close_order(order_tracker)` for partial exits.
4. Set `self.DCA = True` in `__init__` if the strategy uses dollar-cost averaging.
5. Set `percent_sizer` as a fraction (e.g. 0.05 for 5% of available cash).
6. Use `self.p.debug` for conditional logging.
