# Strategy Development Guide

This guide covers strategy development in BTQuant, including the BaseStrategy framework, available pre-built strategies, and best practices for creating custom strategies.

## Table of Contents

- [BaseStrategy Framework](#basestrategy-framework)
- [Pre-built Strategies](#pre-built-strategies)
- [Creating Custom Strategies](#creating-custom-strategies)
- [Strategy Parameters](#strategy-parameters)
- [Order Management](#order-management)
- [Risk Management](#risk-management)
- [Testing and Optimization](#testing-and-optimization)
- [Best Practices](#best-practices)

## BaseStrategy Framework

BTQuant's `BaseStrategy` provides a comprehensive foundation for trading strategies, handling order management, position tracking, DCA (Dollar-Cost Averaging), and telemetry.

### Key Features

- **Unified Order Management**: Consistent order creation and tracking
- **DCA Engine**: Built-in dollar-cost averaging with configurable parameters
- **Position Tracking**: Detailed entry/exit tracking with P&L calculations
- **Risk Management**: Configurable stop-loss and take-profit levels
- **Telemetry**: Built-in logging, notifications, and performance metrics
- **Multi-Exchange Support**: Compatible with all supported exchanges

### Basic Structure

```python
from backtrader.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    params = (
        ('take_profit', 2.0),    # 2% take profit
        ('stop_loss', 5.0),      # 5% stop loss
        ('dca_levels', 3),       # Number of DCA levels
        ('dca_percentage', 0.5), # DCA percentage per level
    )

    def __init__(self):
        super().__init__()
        # Initialize your indicators here
        self.sma = bt.indicators.SimpleMovingAverage(self.data, period=20)

    def buy_or_short_condition(self):
        """Override to implement entry logic"""
        if not self.buy_executed and self.data.close[0] > self.sma[0]:
            self.create_order('BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        """Override to implement exit logic"""
        if self.buy_executed and self.data.close[0] < self.sma[0]:
            self.close_all_positions()
            return True
        return False
```

### Core Methods

#### Entry Conditions
- `buy_or_short_condition()`: Define when to enter long/short positions
- `create_order(order_type, size=None)`: Create buy/sell orders with DCA support

#### Exit Conditions
- `sell_or_cover_condition()`: Define when to exit positions
- `close_all_positions()`: Close all active positions
- `check_stop_loss()`: Custom stop-loss logic

#### Order Management
- `active_orders`: List of current active orders
- `buy_executed`: Boolean indicating if a buy position is active
- `average_entry_price`: Average entry price across DCA levels

## Pre-built Strategies

BTQuant includes a comprehensive library of pre-built strategies covering various trading styles and market conditions.

### Trend Following Strategies

#### Aligator SuperTrend
**File**: `Aligator_supertrend.py`

Combines Williams Alligator and SuperTrend indicators for trend identification.

**Features:**
- Multi-timeframe trend analysis
- Dynamic support/resistance levels
- DCA integration

**Parameters:**
- `jaw_period`, `teeth_period`, `lips_period`: Alligator periods
- `supertrend_period`, `supertrend_multiplier`: SuperTrend settings

#### SMA Cross MESAdaptive Prime
**File**: `SMA_Cross_MESAdaptive_Prime.py`

Uses Simple Moving Averages with MESA Adaptive Moving Average for trend confirmation.

**Features:**
- Adaptive trend detection
- Multiple timeframe analysis
- Hilbert transform integration

### Momentum Strategies

#### MACD ADX
**File**: `MACD_ADX.py`

Combines MACD oscillator with ADX trend strength indicator.

**Features:**
- Momentum and trend strength confirmation
- Multiple EMA combinations
- Trailing stop integration

#### ST RSX ASI
**File**: `ST_RSX_ASI.py`

Combines SuperTrend, RSX (Relative Strength Index), and Accumulative Swing Index.

**Features:**
- Multi-indicator convergence
- Swing analysis
- Oversold/overbought detection

### Mean Reversion Strategies

#### QQE Hullband Volume Oscillator
**File**: `QQE_Hullband_VolumeOsc.py`

Uses QQE indicator with Hull Moving Average and Volume Oscillator.

**Features:**
- Quantitative qualitative estimation
- Volume confirmation
- Band-based signals

#### Vumanchu Market Cipher Variants
**Files**: `Vumanchu_A.py`, `Vumanchu_B.py`

Advanced market cipher implementations with multiple signal types.

**Features:**
- WaveTrend analysis
- Stochastic RSI
- Multiple timeframe signals
- Market structure analysis

### Machine Learning Strategies

#### Nearest Neighbors Rational Quadratic Kernel
**File**: `NearestNeighbors_RationalQuadraticKernel.py`

Machine learning-based strategy using kernel methods.

**Features:**
- ML signal generation
- Multiple technical indicators as features
- Adaptive signal processing

### Specialized Strategies

#### Order Chain Kioseff Trading
**File**: `Order_Chain_Kioseff_Trading.py`

Uses Order Chain indicator for market microstructure analysis.

**Features:**
- Order flow analysis
- Market maker detection
- Volume profile integration

#### PancakeSwap DCA Marketmaker
**File**: `pancakeswap_dca_marketmaker.py`

DEX market making strategy with DCA.

**Features:**
- Automated liquidity provision
- DCA-based position management
- Gas-optimized execution

## Creating Custom Strategies

### Basic Template

```python
import backtrader as bt
from backtrader.strategies.base import BaseStrategy

class CustomStrategy(BaseStrategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('take_profit', 2.0),
        ('stop_loss', 5.0),
    )

    def __init__(self):
        super().__init__()

        # Initialize indicators
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.slow_period)

    def buy_or_short_condition(self):
        """Entry logic"""
        if (not self.buy_executed and
            self.fast_ma[0] > self.slow_ma[0] and
            self.fast_ma[-1] <= self.slow_ma[-1]):  # Crossover

            self.create_order('BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        """Exit logic"""
        if (self.buy_executed and
            self.fast_ma[0] < self.slow_ma[0] and
            self.fast_ma[-1] >= self.slow_ma[-1]):  # Crossunder

            self.close_all_positions()
            return True
        return False

    def check_stop_loss(self):
        """Custom stop loss logic"""
        if self.buy_executed and self.average_entry_price:
            current_price = self.data.close[0]
            stop_price = self.average_entry_price * (1 - self.p.stop_loss / 100)

            if current_price <= stop_price:
                self.close_all_positions()
                return True
        return False
```

### Advanced Template with Multiple Indicators

```python
class AdvancedStrategy(BaseStrategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('atr_period', 14),
        ('risk_per_trade', 0.02),  # 2% risk per trade
    )

    def __init__(self):
        super().__init__()

        # Momentum indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )

        # Volatility
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # Custom indicators
        self.bollinger = bt.indicators.BollingerBands(self.data.close)

    def buy_or_short_condition(self):
        """Multi-condition entry"""
        rsi_oversold = self.rsi[0] < self.p.rsi_oversold
        macd_bullish = self.macd.macd[0] > self.macd.signal[0]
        price_near_lower = self.data.close[0] <= self.bollinger.bot[0] * 1.01

        if (not self.buy_executed and
            rsi_oversold and macd_bullish and price_near_lower):

            self.create_order('BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        """Multi-condition exit"""
        rsi_overbought = self.rsi[0] > self.p.rsi_overbought
        macd_bearish = self.macd.macd[0] < self.macd.signal[0]
        price_near_upper = self.data.close[0] >= self.bollinger.top[0] * 0.99

        if (self.buy_executed and
            (rsi_overbought or macd_bearish or price_near_upper)):

            self.close_all_positions()
            return True
        return False
```

## Strategy Parameters

### Common Parameters

```python
params = (
    # Entry/Exit
    ('take_profit', 2.0),      # Take profit percentage
    ('stop_loss', 5.0),        # Stop loss percentage
    ('trailing_stop', 1.0),    # Trailing stop percentage

    # DCA Settings
    ('dca_levels', 3),         # Number of DCA levels
    ('dca_percentage', 0.5),   # DCA percentage per level
    ('dca_spacing', 2.0),      # Price spacing between DCA levels

    # Risk Management
    ('risk_per_trade', 0.02),  # Risk per trade (2%)
    ('max_risk_per_trade', 0.05),  # Maximum risk per trade
    ('max_open_trades', 5),    # Maximum concurrent trades

    # Indicator Parameters
    ('fast_period', 10),       # Fast MA period
    ('slow_period', 30),       # Slow MA period
    ('rsi_period', 14),        # RSI period
    ('rsi_overbought', 70),    # RSI overbought level
    ('rsi_oversold', 30),      # RSI oversold level
)
```

### Dynamic Parameters

```python
def nextstart(self):
    """Called once before the strategy starts"""
    # Adjust parameters based on market conditions
    if self.data.volume[0] > self.data.volume[-1] * 1.5:
        self.p.take_profit = 1.5  # Reduce target in high volume
    else:
        self.p.take_profit = 2.5  # Increase target in low volume
```

## Order Management

### Basic Order Types

```python
# Market Order
self.create_order('BUY')  # Uses default sizing

# Limit Order
self.create_order('BUY', price=limit_price)

# Custom Size
risk_amount = self.broker.getcash() * self.p.risk_per_trade
size = risk_amount / (entry_price * (1 + self.p.stop_loss / 100))
self.create_order('BUY', size=size)
```

### Advanced Order Management

```python
def manage_orders(self):
    """Advanced order management"""
    # Cancel pending orders if conditions change
    for order in self.active_orders[:]:
        if self.should_cancel_order(order):
            self.cancel_order(order)

    # Adjust stop loss based on volatility
    if self.buy_executed:
        current_atr = self.atr[0]
        new_stop = self.average_entry_price - (current_atr * 2)
        self.adjust_stop_loss(new_stop)

def should_cancel_order(self, order_tracker):
    """Determine if order should be cancelled"""
    # Cancel if price moved too far
    if order_tracker.order_type == 'BUY':
        if self.data.close[0] > order_tracker.price * 1.02:  # 2% slippage
            return True
    return False
```

## Risk Management

### Position Sizing

```python
def calculate_position_size(self, entry_price, stop_price):
    """Calculate position size based on risk"""
    risk_amount = self.broker.getcash() * self.p.risk_per_trade
    risk_per_share = abs(entry_price - stop_price)
    size = risk_amount / risk_per_share

    # Apply maximum position limits
    max_size = self.broker.getcash() * 0.1  # Max 10% of capital
    size = min(size, max_size / entry_price)

    return size
```

### Portfolio Risk Controls

```python
def check_portfolio_risk(self):
    """Global portfolio risk management"""
    total_risk = sum(order_tracker.risk for order_tracker in self.active_orders)

    if total_risk > self.p.max_portfolio_risk:
        # Reduce exposure
        self.reduce_exposure()

    # Check drawdown
    current_value = self.broker.getvalue()
    peak_value = max(self.portfolio_values)
    drawdown = (peak_value - current_value) / peak_value

    if drawdown > self.p.max_drawdown:
        # Emergency stop
        self.emergency_stop()
```

### Stop Loss Strategies

```python
def implement_stop_loss(self):
    """Multiple stop loss strategies"""
    if not self.buy_executed:
        return

    current_price = self.data.close[0]
    entry_price = self.average_entry_price

    # Fixed Percentage Stop Loss
    fixed_stop = entry_price * (1 - self.p.stop_loss / 100)
    if current_price <= fixed_stop:
        self.close_all_positions()
        return

    # ATR-based Stop Loss
    atr_stop = entry_price - (self.atr[0] * 2)
    if current_price <= atr_stop:
        self.close_all_positions()
        return

    # Trailing Stop Loss
    if hasattr(self, 'trailing_stop_price'):
        if current_price > self.trailing_stop_price:
            # Update trailing stop
            self.trailing_stop_price = current_price * (1 - self.p.trailing_stop / 100)
        elif current_price <= self.trailing_stop_price:
            self.close_all_positions()
            return
    else:
        # Initialize trailing stop
        self.trailing_stop_price = current_price * (1 - self.p.trailing_stop / 100)
```

## Testing and Optimization

### Backtesting

```python
from backtrader import backtest

# Single backtest
result = backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    commission=0.001,  # 0.1% commission
    quantstats=True,
    plot=True
)

# Multiple assets
results = bulk_backtest(
    strategy=MyStrategy,
    coins=['BTC', 'ETH', 'ADA'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    init_cash=10000
)
```

### Optimization

```python
from backtrader.utils.backtest import optimize_backtest

# Parameter optimization
results = optimize_backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    fast_period=[5, 10, 15, 20],
    slow_period=[20, 30, 40, 50],
    take_profit=[1.0, 2.0, 3.0],
    max_workers=4  # Parallel optimization
)

# Analyze results
best_params = results[0]['params']
best_return = results[0]['return']
```

### Walk-Forward Analysis

```python
def walk_forward_optimization(data, strategy, window_size=252):
    """Walk-forward optimization"""
    results = []

    for i in range(window_size, len(data), 21):  # Monthly retraining
        train_data = data[i-window_size:i]
        test_data = data[i:i+21]

        # Optimize on training data
        opt_results = optimize_backtest(
            strategy=strategy,
            data=train_data,
            fast_period=[10, 20, 30],
            slow_period=[30, 50, 70]
        )

        # Test best parameters
        best_params = opt_results[0]['params']
        test_result = backtest(
            strategy=strategy,
            data=test_data,
            **best_params
        )

        results.append(test_result)

    return results
```

## Best Practices

### Code Organization

1. **Separate Concerns**: Keep indicator logic separate from entry/exit logic
2. **Parameter Validation**: Validate parameters in `__init__`
3. **Error Handling**: Use try/except blocks for critical operations
4. **Logging**: Use appropriate log levels for different messages

### Performance Optimization

1. **Indicator Efficiency**: Calculate indicators once and reuse
2. **Avoid Look-Ahead Bias**: Don't use future data in calculations
3. **Memory Management**: Clean up large data structures when possible
4. **Vectorization**: Use vectorized operations where applicable

### Risk Management

1. **Position Limits**: Never risk more than 1-2% per trade
2. **Diversification**: Spread risk across multiple assets/strategies
3. **Stop Losses**: Always use stop losses, never rely on mental stops
4. **Portfolio Limits**: Set maximum drawdown limits

### Testing

1. **Out-of-Sample Testing**: Test on data not used for optimization
2. **Walk-Forward Analysis**: Use walk-forward optimization for realistic results
3. **Monte Carlo Simulation**: Test strategy robustness with random variations
4. **Live Paper Trading**: Test in live conditions before going live

### Documentation

1. **Strategy Description**: Document what the strategy does and why
2. **Parameters**: Explain each parameter and its effect
3. **Assumptions**: Document market assumptions and limitations
4. **Performance**: Track and document performance metrics

### Example: Complete Strategy

```python
"""
Advanced RSI Divergence Strategy

This strategy identifies RSI divergences and combines them with trend analysis
for high-probability entries. It uses ATR for dynamic stop losses and includes
DCA for position management.
"""

import backtrader as bt
from backtrader.strategies.base import BaseStrategy

class RSIDivergenceStrategy(BaseStrategy):
    params = (
        # RSI Parameters
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),

        # Divergence Parameters
        ('divergence_lookback', 20),
        ('divergence_threshold', 0.5),

        # Trend Filter
        ('trend_period', 50),

        # Risk Management
        ('take_profit', 3.0),
        ('stop_loss_atr', 2.0),
        ('risk_per_trade', 0.015),

        # DCA
        ('dca_levels', 2),
        ('dca_percentage', 0.6),
    )

    def __init__(self):
        super().__init__()

        # Core indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.trend_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.trend_period)
        self.atr = bt.indicators.ATR(self.data, period=14)

        # Divergence detection
        self.price_swing = bt.indicators.SwingIndicator(self.data.close)
        self.rsi_swing = bt.indicators.SwingIndicator(self.rsi)

    def find_divergence(self):
        """Detect RSI divergences"""
        # Look for bullish divergence: price makes lower low, RSI makes higher low
        price_lows = []
        rsi_lows = []

        for i in range(-self.p.divergence_lookback, 0):
            if self.price_swing[i] < 0:  # Price low
                price_lows.append((i, self.data.low[i]))
            if self.rsi_swing[i] < 0:  # RSI low
                rsi_lows.append((i, self.rsi[i]))

        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            # Check for divergence
            recent_price_low = min(price_lows, key=lambda x: x[1])
            older_price_low = min(price_lows[:-1], key=lambda x: x[1])

            recent_rsi_low = max(rsi_lows, key=lambda x: x[1])
            older_rsi_low = max(rsi_lows[:-1], key=lambda x: x[1])

            # Bullish divergence: price lower low, RSI higher low
            if (recent_price_low[1] < older_price_low[1] and
                recent_rsi_low[1] > older_rsi_low[1]):
                return 'bullish'

        return None

    def buy_or_short_condition(self):
        """Entry condition with divergence and trend filter"""
        if self.buy_executed:
            return False

        # Trend filter: price above trend MA
        trend_up = self.data.close[0] > self.trend_ma[0]

        # RSI oversold
        rsi_oversold = self.rsi[0] < self.p.rsi_oversold

        # Divergence signal
        divergence = self.find_divergence() == 'bullish'

        if trend_up and rsi_oversold and divergence:
            self.create_order('BUY')
            return True

        return False

    def sell_or_cover_condition(self):
        """Exit conditions"""
        if not self.buy_executed:
            return False

        # Take profit
        if self.average_entry_price:
            profit_pct = (self.data.close[0] - self.average_entry_price) / self.average_entry_price * 100
            if profit_pct >= self.p.take_profit:
                self.close_all_positions()
                return True

        # RSI overbought exit
        if self.rsi[0] > self.p.rsi_overbought:
            self.close_all_positions()
            return True

        return False

    def check_stop_loss(self):
        """ATR-based dynamic stop loss"""
        if not self.buy_executed or not self.average_entry_price:
            return False

        # ATR-based stop loss
        atr_value = self.atr[0]
        stop_price = self.average_entry_price - (atr_value * self.p.stop_loss_atr)

        if self.data.close[0] <= stop_price:
            self.close_all_positions()
            return True

        return False

    def calculate_position_size(self):
        """Risk-based position sizing"""
        if not self.average_entry_price:
            return super().calculate_position_size()

        risk_amount = self.broker.getcash() * self.p.risk_per_trade
        atr_value = self.atr[0]
        risk_per_share = atr_value * self.p.stop_loss_atr

        size = risk_amount / risk_per_share
        max_size = self.broker.getcash() * 0.05 / self.data.close[0]  # Max 5% of capital

        return min(size, max_size)
```

This comprehensive strategy demonstrates advanced concepts including divergence detection, trend filtering, ATR-based stops, and risk management. Use it as a template for building sophisticated trading strategies in BTQuant.