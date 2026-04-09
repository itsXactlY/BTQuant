# Indicators Reference

BTQuant wraps the full backtrader indicator library and adds a TransparencyPatch
system that captures every intermediate indicator value per bar. Indicators are
created via `bt.indicators.*` or `bt.ind.*` inside strategy `__init__`.

## TransparencyPatch

The `TransparencyPatch` singleton patches `bt.Indicator.__setattr__` to record
every line-assignment inside any indicator. It is activated per-strategy via the
`capture_data` parameter.

### Activation

```python
from backtrader import transparencypatch

# In BaseStrategy.__init__:
if self.p.capture_data:
    from backtrader.strategies.base import activate_patch
    activate_patch(debug=False)

# Or manually:
patch = transparencypatch.TransparencyPatch()
patch.apply_indicator_patch()
```

### Capturing data per bar

```python
from backtrader.strategies.base import capture_patch

def next(self):
    if self.p.capture_data:
        capture_patch(self)   # records OHLCV + all indicator values
```

### Exporting

```python
from backtrader.transparencypatch import export_data, print_patch

# After run:
print_patch(auto_export=True, filename='my_run')
# Writes exports/my_run.parquet and exports/my_run.csv
```

`TransparencyPatch` class methods:
- `apply_indicator_patch()` - monkey-patches `bt.Indicator.__setattr__`
- `capture_patch_fast(strategy)` - records one bar of OHLCV + all registered indicator values
- `get_dataframe() -> pl.DataFrame` - returns captured data
- `export_data(filename, export_dir)` - writes Parquet + CSV
- `print_summary()` - prints bar count, column count, sample rows

## Backtrader Built-in Indicators

All indicators are accessed as `bt.indicators.<Name>` or `bt.ind.<Name>`.

### Trend

| Indicator | Usage | Lines |
|---|---|---|
| `SMA` | `bt.ind.SMA(data, period=20)` | `sma` |
| `EMA` | `bt.ind.EMA(data, period=20)` | `ema` |
| `HMA` | `bt.ind.HullMovingAverage(data, period=20)` | `hma` |
| `WMA` | `bt.ind.WeightedMovingAverage(data, period=20)` | `wma` |
| `CrossOver` | `bt.ind.CrossOver(data1, data2)` | `cross` |
| `CrossDown` | `bt.ind.CrossDown(data1, data2)` | `crossdown` |

### Momentum

| Indicator | Usage | Lines |
|---|---|---|
| `RSI` | `bt.ind.RSI(data, period=14)` | `rsi` |
| `MACD` | `bt.ind.MACD(data, period_me1=12, period_me2=26, period_signal=9)` | `macd`, `signal`, `histogram` |
| `Stochastic` | `bt.ind.Stochastic(data, period=14)` | `percK`, `percD` |
| `CCI` | `bt.ind.CCI(data, period=20)` | `cci` |
| `WilliamsR` | `bt.ind.WilliamsR(data, period=14)` | `percR` |
| `Momentum` | `bt.ind.Momentum(data, period=10)` | `momentum` |
| `RateOfChange` | `bt.ind.RateOfChange(data, period=10)` | `roc` |
| `AwesomeOscillator` | `bt.ind.AwesomeOscillator(data)` | `ao` |

### Volatility

| Indicator | Usage | Lines |
|---|---|---|
| `ATR` | `bt.ind.ATR(data, period=14)` | `atr` |
| `AverageTrueRange` | `bt.indicators.AverageTrueRange(data, period=14)` | `atr` |
| `BollingerBands` | `bt.ind.BollingerBands(data, period=20, devfactor=2.0)` | `top`, `mid`, `bot` |
| `StandardDeviation` | `bt.ind.StandardDeviation(data, period=20)` | `stddev` |

### Volume

| Indicator | Usage | Lines |
|---|---|---|
| `VolumeOscillator` | `bt.ind.VolumeOscillator(data)` | `volosc` |
| `ChaikinMoneyFlow` | `bt.ind.ChaikinMoneyFlow(data, period=21)` | `cmf` |
| `AccumulationDistribution` | `bt.ind.AccumulationDistribution(data)` | `ad` |
| `OnBalanceVolume` | `bt.ind.OnBalanceVolume(data)` | `obv` |
| `KlingerOscillator` | custom file | `kvo`, `sig` |

### Oscillators

| Indicator | Usage | Lines |
|---|---|---|
| `UltimateOscillator` | `bt.ind.UltimateOscillator(data)` | `uo` |
| `AwesomeOscillator` | `bt.ind.AwesomeOscillator(data)` | `ao` |
| `PrettyGoodOscillator` | custom | `pgo` |

## BTQuant Custom Indicators

These are in `dependencies/backtrader/indicators/` and used by strategies.

### Ehlers Indicators

| File | Class | Lines | Purpose |
|---|---|---|---|
| `AdaptiveCyberCycle.py` | `AdaptiveCyberCycle` | `cycle`, `smooth`, `signal`, `trigger` | Adaptive cycle analysis |
| `CyberCycle.py` | `CyberCycle` | `cycle`, `smooth`, `trigger` | Classical cycle |
| `DecyclerOscillator.py` | `DecyclerOscillator` | `osc`, `decycle`, `hp` | Trend extraction |
| `ElhersHighPass.py` | `EhlersHighPass` | `hp` | High-pass filter |
| `LaguerreFilter.py` | `LaguerreFilter` | `filter`, `p`, `L0`-`L3` | 4-stage Laguerre |
| `AdaptiveLaguerreFilter.py` | `AdaptiveLaguerreFilter` | filter lines | Self-adjusting Laguerre |
| `MesaAdaptiveMovingAverage.py` | `MesaAdaptiveMovingAverage` | `MAMA`, `FAMA` | Hilbert-based adaptive MA |
| `RSX.py` | `RSX` | `rsx` | Smoothed RSI |
| `RoofingFilter.py` | `RoofingFilter` | `roof`, `iroof` | HP + super-smooth |
| `SuperSmoothFilter.py` | `SuperSmootherFilter` | `ssf` | Butterworth smoothing |
| `ButterWorth.py` | `ButterworthFilter` | `bf` | N-pole Butterworth |
| `iFisher.py` | `iFisher` | `ifisher` | Inverse Fisher transform |
| `iDecycler.py` | `iDecycler` | `idec` | Inverse decycler |

### Other Custom Indicators

| File | Class | Purpose |
|---|---|---|
| `SuperTrend.py` | `SuperTrend` | ATR-based trend bands |
| `WilliamsAligator.py` | `WilliamsAlligator` | Jaw/teeth/lips MAs |
| `macd.py` | `MACD` | Extended MACD |
| `rsi.py` | `RSI` | Extended RSI |
| `stochastic.py` | `Stochastic` | Extended Stochastic |
| `atr.py` | `ATR` | ATR wrapper |
| `bollinger.py` | `BollingerBands` | Bollinger wrapper |
| `cci.py` | `CCI` | Commodity Channel Index |
| `rmi.py` | `RMI` | Relative Momentum Index |
| `lrsi.py` | `LRSI` | Laguerre RSI |
| `qqe.py` | `QQE` | Quantitative Qualitative Estimation |
| `ichimoku.py` | `Ichimoku` | Full Ichimoku cloud |
| `pivotpoint.py` | `PivotPoint` | Standard/Fibo/DeMark pivots |
| `FibonacciLevels.py` | `FibonacciLevels` | Fib retracements |
| `ChaikinMoneyFlow.py` | `ChaikinMoneyFlow` | Volume-weighted flow |
| `ChaikinVolatility.py` | `ChaikinVolatility` | Price range vol |
| `SchaffTrendCycle.py` | `SchaffTrendCycle` | MACD-based cycle |
| `StandarizedATR.py` | `StandarizedATR` | Normalized ATR |
| `VolumeOscillator.py` | `VolumeOscillator` | Short/long EMA vol |
| `Klingeroscillator.py` | `KlingerOscillator` | Volume-price momentum |
| `ultimateoscillator.py` | `UltimateOscillator` | 3-period weighted |
| `williams.py` | `WilliamsR` | Williams %R |
| `VumanchuMarketCipher_A.py` | `VumanchuMarketCipher_A` | WaveTrend + RSI + MFI signals |
| `VumanchuMarketCipher_B.py` | `VumanchuMarketCipher_B` | Advanced WaveTrend cycle |
| `WaddahAttarExplosion.py` | `WaddahAttarExplosion` | Trend/vol explosion |
| `TrendTriggerFactor.py` | `TrendTriggerFactor` | Long-term trend trigger |
| `Stochcastic_Generic.py` | `Stochastic_Generic` | Generic data source stochastic |
| `SMA_Cross_MESAdaptive_Prime.py` | (in strategies) | SMA + MESA crossover |

## functions.py -- Logic Combinators

`backtrader.functions` provides line-based logic operators:

| Class | Usage | Description |
|---|---|---|
| `bt.And(a, b, ...)` | `bt.And(self.rsi > 30, self.macd > self.macd.signal)` | Boolean AND across lines |
| `bt.Or(a, b, ...)` | `bt.Or(cond1, cond2)` | Boolean OR |
| `bt.If(cond, a, b)` | `bt.If(condition, value_if_true, value_if_false)` | Conditional |
| `bt.Max(a, b)` | element-wise max | |
| `bt.Min(a, b)` | element-wise min | |
| `bt.Sum(a, b, ...)` | element-wise sum | |
| `bt.Cmp(a, b)` | returns -1, 0, 1 | |
| `bt.DivByZero(a, b, zero=0.0)` | safe division | |
| `bt.Any(a, b, ...)` | any truthy | |
| `bt.All(a, b, ...)` | all truthy | |

## Custom Indicator Example

```python
import backtrader as bt

class MyIndicator(bt.Indicator):
    lines = ('myline', 'signal')
    params = (('period', 20), ('mult', 2.0))

    def __init__(self):
        self.sma = bt.ind.SMA(self.data.close, period=self.p.period)
        self.std = bt.ind.StandardDeviation(self.data.close, period=self.p.period)
        self.lines.myline = self.sma + self.std * self.p.mult
        self.lines.signal = bt.ind.EMA(self.lines.myline, period=5)
```

When `capture_data=True`, all intermediate assignments inside `__init__` are
automatically recorded by the TransparencyPatch and exported per bar.
