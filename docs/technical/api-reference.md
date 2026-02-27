# API Reference

This comprehensive API reference covers all BTQuant modules, classes, and functions. The API is organized by component for easy navigation.

## Table of Contents

- [Backtrader Extensions](#backtrader-extensions)
- [Data Sources](#data-sources)
- [Strategies](#strategies)
- [Indicators](#indicators)
- [Analyzers](#analyzers)
- [Utilities](#utilities)
- [Live Trading](#live-trading)
- [Transparency](#transparency)

## Backtrader Extensions

### Cerebro Class

Extended Backtrader Cerebro with BTQuant features.

```python
import backtrader as bt

cerebro = bt.Cerebro()

# BTQuant extensions
cerebro.addstrategy(MyStrategy)
cerebro.adddata(data)
cerebro.addanalyzer(bt.analyzers.QuantStats, folder='reports')
```

#### BTQuant-Specific Methods

```python
# Quick backtest execution
result = bt.backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    quantstats=True,
    plot=True
)

# Bulk backtesting
results = bt.bulk_backtest(
    strategy=MyStrategy,
    coins=['BTC', 'ETH', 'ADA'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Parameter optimization
opt_results = bt.optimize_backtest(
    strategy=MyStrategy,
    data=data,
    fast_period=[10, 15, 20],
    slow_period=[30, 40, 50]
)
```

## Data Sources

### CCXT Data

```python
from backtrader.utils.ccxt_data import get_crypto_data

data = get_crypto_data(
    asset='BTC/USDT',           # Trading pair
    start_date='2024-01-01',    # Start date
    end_date='2024-12-31',      # End date
    timeframe='1h',             # Candle interval
    exchange='binance'          # Exchange name
)
```

**Parameters:**
- `asset` (str): Trading pair (e.g., 'BTC/USDT')
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format
- `timeframe` (str): Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')
- `exchange` (str): Exchange name ('binance', 'coinbase', 'kraken', etc.)

**Returns:** Polars DataFrame with OHLCV data

### SQL Server Data

```python
from backtrader.feeds.mssql_crypto import get_database_data

data = get_database_data(
    ticker='BTC',                    # Asset ticker
    start_date='2024-01-01',        # Start date
    end_date='2024-12-31',          # End date
    time_resolution='1h',           # Timeframe
    pair='USDT'                     # Quote currency
)
```

**Parameters:**
- `ticker` (str): Asset ticker ('BTC', 'ETH', etc.)
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format
- `time_resolution` (str): Timeframe ('1m', '5m', '1h', '1d', etc.)
- `pair` (str): Quote currency ('USDT', 'BTC', etc.)

**Returns:** Pandas DataFrame with OHLCV data

### HotSpine Live Data

```python
from backtrader.feeds.hotspine_feed import HotSpineData

data = HotSpineData(
    symbol_id=123,                   # Symbol identifier
    shm_name='/btquant_hotspine',   # Shared memory name
    batch_mode=False,               # Single trade mode
    poll_interval=0.0001            # Polling interval
)
```

**Parameters:**
- `symbol_id` (int): Numeric symbol identifier
- `shm_name` (str): Shared memory segment name
- `batch_mode` (bool): Use batch processing for high throughput
- `poll_interval` (float): Polling interval in seconds

### CSV/Pandas Data

```python
from backtrader.feeds.polarfeed import PolarsData

# From Polars DataFrame
data = PolarsData(dataname=df)

# From CSV file
import polars as pl
df = pl.read_csv('data.csv')
data = PolarsData(dataname=df)
```

## Strategies

### BaseStrategy

BTQuant's comprehensive strategy base class.

```python
from backtrader.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    params = (
        ('take_profit', 2.0),
        ('stop_loss', 5.0),
        ('dca_levels', 3),
    )

    def __init__(self):
        super().__init__()
        # Initialize indicators
        self.sma = bt.indicators.SMA(self.data, period=20)

    def buy_or_short_condition(self):
        """Entry logic - override this method"""
        if self.data.close[0] > self.sma[0]:
            self.create_order('BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        """Exit logic - override this method"""
        if self.data.close[0] < self.sma[0]:
            self.close_all_positions()
            return True
        return False
```

#### Key Methods

**Order Management:**
- `create_order(order_type, size=None)`: Create buy/sell orders with DCA
- `close_all_positions()`: Close all active positions
- `active_orders`: List of current orders
- `buy_executed`: Boolean indicating active long position
- `average_entry_price`: Average entry price across DCA levels

**Risk Management:**
- `check_stop_loss()`: Custom stop loss logic
- `calculate_position_size()`: Risk-based position sizing

**Telemetry:**
- `log(message, dt=None, doprint=False)`: Logging method
- `broker.getvalue()`: Current portfolio value
- `broker.getcash()`: Available cash

### Pre-built Strategies

#### Trend Following
```python
from backtrader.strategies.Aligator_supertrend import AligatorSuperTrend
from backtrader.strategies.SMA_Cross_MESAdaptive_Prime import SMACrossMESAdaptivePrime
from backtrader.strategies.ST_RSX_ASI import STRSXASI
```

#### Momentum
```python
from backtrader.strategies.MACD_ADX import MACDADX
from backtrader.strategies.QQE_Hullband_VolumeOsc import QQEHullbandVolumeOsc
```

#### Mean Reversion
```python
from backtrader.strategies.Vumanchu_A import VuManchCipher_A
from backtrader.strategies.Vumanchu_B import VuManchCipher_B
```

#### Machine Learning
```python
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NearestNeighborsRationalQuadratic
```

## Indicators

### Trend Indicators

#### Moving Averages
```python
# Simple Moving Average
sma = bt.indicators.SMA(data, period=20)

# Exponential Moving Average
ema = bt.indicators.EMA(data, period=20)

# Hull Moving Average
hma = bt.indicators.HullMovingAverage(data, period=20)

# MESA Adaptive Moving Average
mama = bt.indicators.MesaAdaptiveMovingAverage(data, fastlimit=0.5, slowlimit=0.05)
```

#### SuperTrend
```python
supertrend = bt.indicators.SuperTrend(data, period=10, multiplier=3.0)
# Lines: supertrend, direction
```

#### Williams Alligator
```python
alligator = bt.indicators.WilliamsAlligator(data,
    jaw_period=13, teeth_period=8, lips_period=5)
# Lines: jaw, teeth, lips
```

### Momentum Indicators

#### RSI Family
```python
# Standard RSI
rsi = bt.indicators.RSI(data, period=14)

# Laguerre RSI
lrsi = bt.indicators.LRSI(data, period=14)

# Relative Strength Index (Ehlers)
rsx = bt.indicators.RSX(data, period=14)
```

#### MACD Family
```python
# Standard MACD
macd = bt.indicators.MACD(data, period_me1=12, period_me2=26, period_signal=9)
# Lines: macd, signal, histogram

# Schaff Trend Cycle
stc = bt.indicators.SchaffTrendCycle(data)
```

#### Oscillators
```python
# Stochastic
stoch = bt.indicators.Stochastic(data, period=14, period_dfast=3, period_dslow=3)
# Lines: percK, percD

# Commodity Channel Index
cci = bt.indicators.CCI(data, period=20)

# Ultimate Oscillator
ult = bt.indicators.UltimateOscillator(data, period1=7, period2=14, period3=28)
```

### Volatility Indicators

#### ATR Family
```python
# Average True Range
atr = bt.indicators.ATR(data, period=14)

# Standardized ATR
std_atr = bt.indicators.StandardizedATR(data, period=14)
```

#### Bollinger Bands
```python
bb = bt.indicators.BollingerBands(data, period=20, devfactor=2.0)
# Lines: top, mid, bot
```

#### Chaikin Volatility
```python
chaikin_vol = bt.indicators.ChaikinVolatility(data, period=10)
```

### Volume Indicators

#### Volume Oscillators
```python
# Volume Oscillator
vol_osc = bt.indicators.VolumeOscillator(data, period1=5, period2=10)

# Chaikin Money Flow
cmf = bt.indicators.ChaikinMoneyFlow(data, period=21)

# Klinger Oscillator
klinger = bt.indicators.KlingerOscillator(data, period_fast=34, period_slow=55)
# Lines: kvo, signal
```

### Advanced Indicators

#### Ehlers Indicators
```python
# Cyber Cycle
cyber = bt.indicators.CyberCycle(data, period=16)
# Lines: cycle, smooth, trigger

# Adaptive Cyber Cycle
adaptive_cyber = bt.indicators.AdaptiveCyberCycle(data, period=16)

# Decycler Oscillator
decycler = bt.indicators.DecyclerOscillator(data, period=30)

# Roofing Filter
roofing = bt.indicators.RoofingFilter(data)
# Lines: roof, iroof

# Super Smoother Filter
super_smooth = bt.indicators.SuperSmootherFilter(data, period=10)

# Laguerre Filter
laguerre = bt.indicators.LaguerreFilter(data, gamma=0.8)
# Lines: filter, p, L0, L1, L2, L3
```

#### Vumanchu Market Cipher
```python
cipher_a = bt.indicators.VumanchuMarketCipher_A(data)
cipher_b = bt.indicators.VumanchuMarketCipher_B(data)
# Comprehensive signal system with multiple components
```

#### Order Chain
```python
order_chain = bt.indicators.OrderChain(data, period=20)
# Market microstructure analysis
```

#### Ichimoku Cloud
```python
ichimoku = bt.indicators.Ichimoku(data)
# Lines: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
```

## Analyzers

### Performance Analyzers

#### QuantStats Analyzer
```python
from backtrader.analyzers import QuantStats

# Add to cerebro
cerebro.addanalyzer(QuantStats, folder='reports', fileprefix='backtest_')

# Access results
quantstats_result = strategy.analyzers.quantstats.get_analysis()
```

**Generated Reports:**
- HTML tear sheet with comprehensive metrics
- Performance statistics
- Risk analysis
- Benchmark comparison

#### PyFolio Analyzer
```python
from backtrader.analyzers import PyFolio

cerebro.addanalyzer(PyFolio)

# Access results
pyfolio_result = strategy.analyzers.pyfolio.get_analysis()
returns = pyfolio_result['returns']
positions = pyfolio_result['positions']
transactions = pyfolio_result['transactions']
```

#### Standard Backtrader Analyzers
```python
# Returns analysis
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.AnnualReturn)

# Risk analysis
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addanalyzer(bt.analyzers.SharpeRatio)
cerebro.addanalyzer(bt.analyzers.SortinoRatio)

# Trade analysis
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
cerebro.addanalyzer(bt.analyzers.SQN)
```

### Custom Analyzers

```python
class CustomAnalyzer(bt.Analyzer):
    """Custom performance analyzer"""

    def __init__(self):
        self.trades = []
        self.returns = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'pnl': trade.pnl,
                'duration': trade.barlen,
                'return': trade.pnlcomm / trade.price * 100
            })

    def get_analysis(self):
        if not self.trades:
            return {}

        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100,
            'avg_win': sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            'avg_loss': sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'profit_factor': sum(t['pnl'] for t in winning_trades) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        }
```

## Utilities

### Backtest Utilities

```python
from backtrader import backtest, bulk_backtest, optimize_backtest

# Single backtest
result = backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    commission=0.001,
    quantstats=True,
    plot=True
)

# Bulk backtest
results = bulk_backtest(
    strategy=MyStrategy,
    coins=['BTC', 'ETH', 'ADA'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    init_cash=10000
)

# Optimization
opt_results = optimize_backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    fast_period=[10, 15, 20],
    slow_period=[30, 40, 50],
    max_workers=4
)
```

### Date Utilities

```python
from backtrader.utils import date

# Date conversions
timestamp = date.date2num(datetime(2024, 1, 1))
dt = date.num2date(timestamp)

# Timezone handling
utc_time = date.utcnow()
local_time = date.localize(utc_time)

# Business days
next_business_day = date.next_business_day(datetime.now())
```

### CCXT Data Utilities

```python
from backtrader.utils.ccxt_data import get_crypto_data, get_available_exchanges

# Get data
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')

# List exchanges
exchanges = get_available_exchanges()
print("Available exchanges:", exchanges)

# Check symbol availability
from backtrader.utils.ccxt_data import check_symbol
is_available = check_symbol('BTC/USDT', 'binance')
```

## Live Trading

### HotSpine Live Trading

```python
from backtrader.livetrading import livetrade_hotspine

# Single symbol live trading
livetrade_hotspine(
    symbol_id=123,
    strategy=MyStrategy,
    shm_name='/btquant_hotspine',
    batch_mode=False,
    poll_interval=0.0001
)

# Multi-symbol live trading
from backtrader.livetrading import livetrade_hotspine_multi_symbol

livetrade_hotspine_multi_symbol(
    symbol_ids=[123, 456, 789],
    strategy=MyMultiSymbolStrategy,
    shm_name='/btquant_hotspine',
    batch_mode=True
)
```

### Exchange Integration

#### CCXT Live Trading
```python
import ccxt

# Initialize exchange
exchange = ccxt.binance({
    'apiKey': 'your_api_key',
    'secret': 'your_secret',
    'enableRateLimit': True
})

# Place order
order = exchange.create_order(
    symbol='BTC/USDT',
    type='limit',
    side='buy',
    amount=0.001,
    price=50000
)
```

#### JackRabbit Relay
```python
# Configure in dontcommit.py
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = "/path/to/jrr/history/"
identify = "your_jrr_identify_string"
```

### Broker Integration

```python
# Custom broker setup
broker = bt.brokers.BackBroker()
broker.set_cash(10000)
broker.set_commission(commission=0.001)

cerebro.setbroker(broker)
```

## Transparency

### Transparency Patch

```python
from backtrader import transparencypatch

# Activate transparency
patch = transparencypatch.TransparencyPatch()
patch.debug = True
patch.apply_indicator_patch()

# All indicator calculations now visible
strategy = MyStrategy()
# Transparency logging enabled
```

### Real-time Monitoring

```python
def next(self):
    if self.p.capture_data:
        # Monitor indicator calculations
        self.log(f"RSI[0]: {self.rsi[0]:.4f}")
        self.log(f"MACD: {self.macd.macd[0]:.6f}")
        self.log(f"ATR: {self.atr[0]:.6f}")

        # Monitor strategy state
        self.log(f"Position: {self.position.size}")
        self.log(f"Cash: {self.broker.getcash():.2f}")
        self.log(f"Value: {self.broker.getvalue():.2f}")
```

### Calculation Transparency

```python
# Example: MACD calculation visibility
self.macd = bt.indicators.MACD(self.data.close,
    period_me1=12, period_me2=26, period_signal=9)

# With transparency, you can see:
# - EMA12 calculation on close prices
# - EMA26 calculation on close prices
# - MACD line = EMA12 - EMA26
# - Signal line = EMA9 of MACD line
# - Histogram = MACD - Signal
```

### Performance Monitoring

```python
import time

def monitor_performance(self):
    """Monitor calculation performance"""
    start_time = time.time()

    # Execute indicator calculations
    rsi_value = self.rsi[0]
    macd_value = self.macd.macd[0]

    calc_time = time.time() - start_time

    if calc_time > 0.001:  # Log slow calculations
        self.log(f"Slow calculation: {calc_time:.6f}s")

    # Memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    if memory_mb > 1000:  # Log high memory usage
        self.log(f"High memory usage: {memory_mb:.1f}MB")
```

## Error Handling

### Exception Types

```python
try:
    data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')
except ccxt.NetworkError as e:
    print(f"Network error: {e}")
except ccxt.ExchangeError as e:
    print(f"Exchange error: {e}")
except ValueError as e:
    print(f"Data validation error: {e}")
```

### Validation Functions

```python
def validate_data(data):
    """Validate data integrity"""
    required_columns = ['open', 'high', 'low', 'close', 'volume']

    # Check columns
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required columns")

    # Check for NaN values
    if data.isnull().any().any():
        raise ValueError("Data contains NaN values")

    # Check price validity
    if (data['close'] <= 0).any():
        raise ValueError("Invalid closing prices")

    return True

# Usage
try:
    validate_data(data)
    print("Data validation passed")
except ValueError as e:
    print(f"Data validation failed: {e}")
```

This API reference provides comprehensive coverage of BTQuant's capabilities. For more detailed examples and tutorials, see the user guides and examples directory.