# BTQuant Indicator Transparency

## What It Does

TransparencyPatch is a singleton that monkey-patches `bt.Indicator.__setattr__` to capture
every intermediate indicator calculation during backtesting. It records OHLCV data plus all
indicator attribute values per bar, then exports them as a Polars DataFrame.

## How It Works

1. `apply_indicator_patch()` replaces `bt.Indicator.__setattr__` with a version that
   registers any non-private, array-like attribute assigned to an indicator.

2. `capture_patch_fast(strategy)` is called each bar in `strategy.next()`. It reads
   OHLCV from `strategy.data` and iterates the registry to extract `[0]` values from
   each registered indicator attribute.

3. Duplicate values (same numeric value across indicators) are deduplicated.

4. Data is batched (100 bars default) then accumulated in `all_captured_data`.

## API

```python
from backtrader.strategies.base import activate_patch, capture_patch

# Before running:
activate_patch(debug=False)

# In strategy.next():
capture_patch(self)
```

Or directly:

```python
from backtrader import transparencypatch

patch = transparencypatch.TransparencyPatch()
patch.apply_indicator_patch()

# Each bar in strategy:
patch.capture_patch_fast(strategy)

# After backtest:
df = patch.get_dataframe()          # Returns Polars DataFrame
patch.export_data("my_run")         # Writes to exports/my_run.parquet + .csv
```

## TransparencyPatch Class

Singleton. Thread-safe via `threading.Lock`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `debug` | bool | Print detailed capture info |
| `batch_size` | int | Bars batched before extending (default: 100) |
| `all_captured_data` | list[dict] | All captured bar data |
| `indicator_registry` | dict | Registered indicator attributes |
| `bar_counter` | int | Total bars captured |

### Methods

| Method | Description |
|--------|-------------|
| `apply_indicator_patch()` | Patch bt.Indicator.__setattr__ (idempotent) |
| `capture_patch_fast(strategy)` | Capture current bar OHLCV + indicators |
| `get_dataframe()` | Returns Polars DataFrame sorted by bar |
| `export_data(filename, export_dir)` | Export to Parquet + CSV in export_dir |
| `flush_remaining_batch()` | Flush incomplete batch to all_captured_data |

## Output Format

The DataFrame has columns:

- `bar` - bar index
- `datetime` - ISO format (if available)
- `open`, `high`, `low`, `close`, `volume` - OHLCV
- `{IndicatorClass}_{attribute}` - each unique indicator intermediate value

Example columns: `SMA_sma`, `RSI_rsi`, `MACD_macd`, `MACD_signal`, `CrossOver_crossover`

## Usage in Strategy

```python
from backtrader.strategies.base import BaseStrategy, activate_patch, capture_patch

class MyStrategy(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sma = bt.ind.SMA(period=20)
        self.rsi = bt.ind.RSI(period=14)
        # ...

    def next(self):
        capture_patch(self)  # Must be called each bar
        # ... strategy logic
```

Or via the `capture_data` param:

```python
cerebro.addstrategy(MyStrategy, capture_data=True)
```

BaseStrategy automatically calls `activate_patch()` and `capture_patch(self)` each bar
when `capture_data=True`.

## What Gets Captured

The patch captures any attribute assigned in an indicator's `__init__` that:
- Does not start with `_`
- Is not a reserved name (lines, params, plotinfo, datas, data, close, open, high, low, etc.)
- Has `__getitem__` and `__len__` (i.e., a backtrader Line object)

This means intermediate calculations like `self.sma`, `self.rsi`, `self.macd`, `self.signal`
inside any indicator are all captured.

## Console Output

BTQuant provides colored console helpers (in `strategies/base.py`):

```python
from backtrader.strategies.base import cinfo, cgood, cwarn, cerr, chead, csep

print(cinfo("Starting"))      # Cyan [i]
print(cgood("Order filled"))   # Green [OK]
print(cwarn("Low balance"))    # Yellow [!]
print(cerr("Order failed"))    # Red [x]
```

Requires `colorama` package. Falls back to plain text without it.
