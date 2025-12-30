# Indicators Reference

BTQuant provides the most comprehensive indicator library available, featuring complete transparency in all calculations. This guide covers all available indicators, their usage, and transparency features.

## Table of Contents

- [Transparency Features](#transparency-features)
- [Trend Indicators](#trend-indicators)
- [Momentum Indicators](#momentum-indicators)
- [Volatility Indicators](#volatility-indicators)
- [Volume Indicators](#volume-indicators)
- [Oscillators](#oscillators)
- [Advanced Indicators](#advanced-indicators)
- [Custom Indicators](#custom-indicators)

## Transparency Features

BTQuant's transparency system allows you to see every calculation step for any indicator.

### Activating Transparency

```python
from backtrader import transparencypatch

# Activate transparency
optimized_patch = transparencypatch.TransparencyPatch()
optimized_patch.debug = True  # Enable debug logging
optimized_patch.apply_indicator_patch()

# Your strategy code here
# All indicator calculations will now be visible
```

### Transparency in Action

```python
# Example: RSI calculation transparency
self.rsi = bt.indicators.RSI(self.data.close, period=14)

# With transparency enabled, you can see:
# - Raw price data input
# - Gain/loss calculations for each period
# - Average gain/loss computations
# - Final RSI formula: RSI = 100 - (100 / (1 + RS))
# - Where RS = Average Gain / Average Loss
```

### Real-time Monitoring

```python
def next(self):
    if self.p.capture_data:
        # Log indicator values in real-time
        self.log(f"RSI[0]: {self.rsi[0]:.4f}")
        self.log(f"MACD: {self.macd.macd[0]:.6f}")
        self.log(f"ATR: {self.atr[0]:.6f}")
```

## Trend Indicators

### Moving Averages

#### Simple Moving Average (SMA)
```python
sma = bt.indicators.SimpleMovingAverage(data, period=20)
```
**Calculation**: `SMA = (P1 + P2 + ... + Pn) / n`

#### Exponential Moving Average (EMA)
```python
ema = bt.indicators.ExponentialMovingAverage(data, period=20)
```
**Calculation**: `EMA(today) = (Price(today) * Multiplier) + EMA(yesterday)`

#### Hull Moving Average (HMA)
```python
hma = bt.indicators.HullMovingAverage(data, period=20)
```
**Calculation**: Combines weighted MA with square root of period for reduced lag.

### SuperTrend
```python
supertrend = bt.indicators.SuperTrend(data, period=10, multiplier=3.0)
```
**Features**:
- Dynamic support/resistance levels
- Trend direction indication
- Customizable ATR period and multiplier

**Calculation**:
```
Basic Upper Band = (High + Low) / 2 + (Multiplier * ATR)
Basic Lower Band = (High + Low) / 2 - (Multiplier * ATR)
Final Band = Previous Final Band (if trend continues) or Basic Band (if trend reverses)
```

### Williams Alligator
```python
alligator = bt.indicators.WilliamsAlligator(data,
    jaw_period=13, teeth_period=8, lips_period=5)
```
**Lines**: `jaw` (13-period SMA), `teeth` (8-period SMA), `lips` (5-period SMA)

### MESA Adaptive Moving Average (MAMA)
```python
mama = bt.indicators.MesaAdaptiveMovingAverage(data, fastlimit=0.5, slowlimit=0.05)
```
**Features**: Self-adjusting based on market cycle using Hilbert transforms.

## Momentum Indicators

### RSI (Relative Strength Index)
```python
rsi = bt.indicators.RSI(data, period=14)
```
**Calculation**:
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

### Stochastic Oscillator
```python
stoch = bt.indicators.Stochastic(data, period=14, period_dfast=3, period_dslow=3)
```
**Lines**: `%K` (fast), `%D` (slow)

### MACD (Moving Average Convergence Divergence)
```python
macd = bt.indicators.MACD(data,
    period_me1=12, period_me2=26, period_signal=9)
```
**Lines**: `macd`, `signal`, `histogram`

### Commodity Channel Index (CCI)
```python
cci = bt.indicators.CCI(data, period=20)
```

### Relative Momentum Index (RMI)
```python
rmi = bt.indicators.RMI(data, period=14, momentum_period=4)
```

## Volatility Indicators

### Average True Range (ATR)
```python
atr = bt.indicators.ATR(data, period=14)
```
**Calculation**:
```
TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
ATR = EMA(TR, period)
```

### Bollinger Bands
```python
bbands = bt.indicators.BollingerBands(data, period=20, devfactor=2.0)
```
**Lines**: `top`, `mid`, `bot`

### Chaikin Volatility
```python
chaikin_vol = bt.indicators.ChaikinVolatility(data, period=10, movav=bt.indicators.ExponentialMovingAverage)
```

### Damiani Volatmeter
```python
volatmeter = bt.indicators.DamianiVolatmeter(data, period=20)
```

## Volume Indicators

### Volume Oscillator
```python
vol_osc = bt.indicators.VolumeOscillator(data, period1=5, period2=10)
```
**Calculation**: `(Short EMA - Long EMA) / Long EMA * 100`

### Chaikin Money Flow
```python
cmf = bt.indicators.ChaikinMoneyFlow(data, period=21)
```
**Calculation**: Sum of money flow over period divided by sum of volume.

### Klinger Oscillator
```python
klinger = bt.indicators.KlingerOscillator(data, period_fast=34, period_slow=55)
```
**Lines**: `kvo` (volume force), `sig` (signal line)

### Accumulation/Distribution (AD)
```python
ad = bt.indicators.AccumulationDistribution(data)
```

## Oscillators

### QQE (Quantitative Qualitative Estimation)
```python
qqe = bt.indicators.QQE(data, rsi_period=14, smoothing=5, qqe_factor=4.236)
```
**Features**: RSI-based oscillator with dynamic bands.

### Ultimate Oscillator
```python
ult_osc = bt.indicators.UltimateOscillator(data,
    period1=7, period2=14, period3=28, weight1=4.0, weight2=2.0, weight3=1.0)
```

### Williams %R
```python
williams_r = bt.indicators.WilliamsR(data, period=14)
```

### Awesome Oscillator
```python
ao = bt.indicators.AwesomeOscillator(data, period1=5, period2=34)
```

### Pretty Good Oscillator (PGO)
```python
pgo = bt.indicators.PrettyGoodOscillator(data, period=14)
```

## Advanced Indicators

### Ehlers Indicators Suite

BTQuant includes the complete John Ehlers indicator collection with full transparency.

#### Cyber Cycle
```python
cyber_cycle = bt.indicators.CyberCycle(data, period=16)
```
**Lines**: `cycle`, `smooth`, `trigger`

#### Adaptive Cyber Cycle
```python
adaptive_cyber = bt.indicators.AdaptiveCyberCycle(data, period=16)
```

#### Decycler Oscillator
```python
decycler = bt.indicators.DecyclerOscillator(data, period=30)
```

#### Roofing Filter
```python
roofing = bt.indicators.RoofingFilter(data)
```
**Lines**: `roof`, `iroof`

#### Super Smoother Filter
```python
super_smooth = bt.indicators.SuperSmootherFilter(data, period=10)
```

#### Laguerre Filter
```python
laguerre = bt.indicators.LaguerreFilter(data, gamma=0.8)
```
**Lines**: `filter`, `p`, `L0`, `L1`, `L2`, `L3`

#### Adaptive Laguerre Filter
```python
adaptive_laguerre = bt.indicators.AdaptiveLaguerreFilter(data)
```

#### RSX (Ehlers RSI)
```python
rsx = bt.indicators.RSX(data, period=14)
```

#### iFisher
```python
ifisher = bt.indicators.iFisher(data, period=10)
```

#### iDecycler
```python
idecycler = bt.indicators.iDecycler(data, period=30)
```

### Vumanchu Market Cipher
```python
cipher_a = bt.indicators.VumanchuMarketCipher_A(data)
cipher_b = bt.indicators.VumanchuMarketCipher_B(data)
```
**Features**: Comprehensive signal system with WaveTrend, RSI, MFI, and pattern analysis.

### Order Chain Indicator
```python
order_chain = bt.indicators.OrderChain(data, period=20)
```
**Purpose**: Market microstructure analysis using order flow data.

### Accumulative Swing Index (ASI)
```python
asi = bt.indicators.AccumulativeSwingIndex(data)
```
**Purpose**: Measures cumulative swing to identify trends.

### Ichimoku Cloud
```python
ichimoku = bt.indicators.Ichimoku(data)
```
**Lines**: `tenkan_sen`, `kijun_sen`, `senkou_span_a`, `senkou_span_b`, `chikou_span`

### Fibonacci Levels
```python
fib_levels = bt.indicators.FibonacciLevels(data, period=20)
```
**Levels**: 23.6%, 38.2%, 50%, 61.8%, 78.6% retracements

### Pivot Points
```python
pivots = bt.indicators.PivotPoint(data)
```
**Variants**: Standard, Fibonacci, and DeMark pivot points.

## Custom Indicators

### Creating Custom Indicators

```python
import backtrader as bt

class CustomIndicator(bt.Indicator):
    """
    Example custom indicator with full transparency
    """
    lines = ('custom', 'signal')

    params = (
        ('period', 20),
        ('multiplier', 2.0),
    )

    def __init__(self):
        # With transparency, all these calculations are visible
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.period)
        self.std = bt.indicators.StandardDeviation(
            self.data.close, period=self.p.period)

        # Final calculation
        self.lines.custom = self.sma + (self.std * self.p.multiplier)
        self.lines.signal = bt.indicators.ExponentialMovingAverage(
            self.lines.custom, period=5)

    def next(self):
        # Additional calculations can be logged with transparency
        if self.p.debug:
            print(f"Custom[{len(self)-1}]: {self.lines.custom[0]:.4f}")
```

### Advanced Custom Indicator

```python
class AdaptiveRSI(bt.Indicator):
    """
    RSI with adaptive period based on volatility
    """
    lines = ('rsi', 'adaptive_period')

    params = (
        ('base_period', 14),
        ('volatility_period', 20),
        ('max_period', 50),
    )

    def __init__(self):
        # Calculate volatility
        self.volatility = bt.indicators.ATR(self.data, period=self.p.volatility_period)

        # Adaptive period calculation
        avg_volatility = bt.indicators.SimpleMovingAverage(
            self.volatility, period=self.p.volatility_period)

        # Scale period based on volatility (higher volatility = shorter period)
        self.lines.adaptive_period = bt.Max(
            self.p.base_period - (avg_volatility / self.data.close * 100),
            self.p.base_period / 2
        )
        self.lines.adaptive_period = bt.Min(
            self.lines.adaptive_period, self.p.max_period)

        # Adaptive RSI
        self.lines.rsi = bt.indicators.RSI(self.data.close, period=self.lines.adaptive_period)

    def next(self):
        # Log transparency information
        current_period = int(self.lines.adaptive_period[0])
        print(f"Adaptive Period: {current_period}, RSI: {self.lines.rsi[0]:.2f}")
```

## Indicator Transparency in Practice

### Real-Time Calculation Monitoring

```python
class TransparentStrategy(bt.Strategy):
    def __init__(self):
        # Enable transparency
        from backtrader import transparencypatch
        self.patch = transparencypatch.TransparencyPatch()
        self.patch.debug = True
        self.patch.apply_indicator_patch()

        # Initialize indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)
        self.atr = bt.indicators.ATR(self.data)

    def next(self):
        # Monitor calculations in real-time
        self.log(f"RSI Calculation Details:")
        self.log(f"  Current RSI: {self.rsi[0]:.4f}")
        self.log(f"  Average Gain: {self.rsi.avg_gain[0]:.6f}")
        self.log(f"  Average Loss: {self.rsi.avg_loss[0]:.6f}")

        self.log(f"MACD Details:")
        self.log(f"  MACD Line: {self.macd.macd[0]:.6f}")
        self.log(f"  Signal Line: {self.macd.signal[0]:.6f}")
        self.log(f"  Histogram: {self.macd.histo[0]:.6f}")

        # Trading logic
        if self.rsi[0] < 30 and self.macd.macd[0] > self.macd.signal[0]:
            self.buy()
        elif self.rsi[0] > 70 and self.macd.macd[0] < self.macd.signal[0]:
            self.sell()
```

### Performance Impact Analysis

```python
def analyze_indicator_performance(self):
    """Analyze indicator calculation performance"""
    import time

    start_time = time.time()
    rsi_value = self.rsi[0]  # Trigger calculation
    calc_time = time.time() - start_time

    self.log(f"RSI calculation time: {calc_time:.6f} seconds")
    self.log(f"Data points processed: {len(self.data)}")

    # Memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    self.log(f"Memory usage: {memory_mb:.2f} MB")
```

## Best Practices

### Indicator Selection
1. **Match to Strategy**: Choose indicators that align with your trading style
2. **Avoid Over-optimization**: Don't use too many indicators
3. **Consider Lag**: Some indicators have inherent lag - account for this
4. **Test Combinations**: Test indicator combinations thoroughly

### Performance Optimization
1. **Calculate Once**: Compute indicators in `__init__`, not `next()`
2. **Reuse Calculations**: Store intermediate results when possible
3. **Profile Performance**: Use transparency to identify slow indicators
4. **Memory Management**: Clean up unused indicators

### Transparency Usage
1. **Development**: Use transparency to understand indicator behavior
2. **Debugging**: Enable transparency when signals behave unexpectedly
3. **Validation**: Verify calculations match your expectations
4. **Education**: Learn how indicators work internally

### Common Pitfalls
1. **Look-ahead Bias**: Ensure indicators don't use future data
2. **Data Alignment**: Verify all indicators use the same data source
3. **Parameter Sensitivity**: Test indicators with various parameters
4. **Market Conditions**: Indicators perform differently in various markets

## Complete Indicator List

| Category | Indicator | File | Key Parameters |
|----------|-----------|------|----------------|
| Trend | SimpleMovingAverage | `basicops.py` | period |
| Trend | ExponentialMovingAverage | `ema.py` | period |
| Trend | HullMovingAverage | `hma.py` | period |
| Trend | SuperTrend | `SuperTrend.py` | period, multiplier |
| Trend | WilliamsAlligator | `WilliamsAligator.py` | jaw_period, teeth_period, lips_period |
| Momentum | RSI | `rsi.py` | period |
| Momentum | Stochastic | `stochastic.py` | period, period_dfast, period_dslow |
| Momentum | MACD | `macd.py` | period_me1, period_me2, period_signal |
| Momentum | CCI | `cci.py` | period |
| Volatility | ATR | `atr.py` | period |
| Volatility | BollingerBands | `bollinger.py` | period, devfactor |
| Volume | VolumeOscillator | `VolumeOscillator.py` | period1, period2 |
| Volume | ChaikinMoneyFlow | `ChaikinMoneyFlow.py` | period |
| Oscillators | QQE | `qqe.py` | rsi_period, smoothing, qqe_factor |
| Advanced | CyberCycle | `CyberCycle.py` | period |
| Advanced | RSX | `RSX.py` | period |
| Advanced | VumanchuMarketCipher_A | `VumanchuMarketCipher_A.py` | Various |

This comprehensive indicator library, combined with complete transparency, makes BTQuant the most powerful and auditable quantitative trading framework available.