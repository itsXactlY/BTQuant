---
name: data-management
description: Data sourcing, preparation, and validation for BTQuant strategies
---

# BTQuant Data Management

When working with trading data:

## Data Sources

### MSSQL Database

```python
from btquant import Database

db = Database.connect_mssql(
    server='localhost',
    database='btquant_data'
)

df = db.query_candles(
    symbol='BTCUSDT',
    exchange='binance',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

### CCXT Integration

```python
from btquant import CCXTDataSource

data = CCXTDataSource(
    exchange='binance',
    symbols=['BTC/USDT', 'ETH/USDT'],
    timeframe='1h'
)

candles = data.fetch_ohlcv(days_back=365)
```

### CSV Import

```python
import pandas as pd
from btquant import DataValidator

df = pd.read_csv('data/BTCUSDT.csv')
df = DataValidator.validate_and_clean(df)
```

## Data Validation

### OHLCV Integrity Checks

```python
validator = DataValidator()

# Check for missing values
assert not df[['open', 'high', 'low', 'close', 'volume']].isna().any()

# Check price relationships
assert (df['high'] >= df['low']).all()
assert (df['high'] >= df['open']).all()
assert (df['high'] >= df['close']).all()
assert (df['low'] <= df['open']).all()
assert (df['low'] <= df['close']).all()

# Check for duplicates and monotonic timestamps
assert not df.index.duplicated().any()
assert df.index.is_monotonic_increasing
```

## Data Quality Metrics

- **Completeness**: % of expected candles present
- **Gaps**: Missing time periods
- **Outliers**: Unusual price movements
- **Volume consistency**: Detect reporting errors
- **Time alignment**: Proper UTC timestamps

## Data Preparation

### Resampling

```python
# Aggregate to higher timeframe
df_daily = df.resample('D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

### Feature Engineering

```python
import polars as pl

df = pl.DataFrame(df)
df = df.with_columns([
    pl.col('close').rolling_mean(window=20).alias('sma_20'),
    pl.col('volume').rolling_mean(window=20).alias('avg_volume')
])
```

## Multi-Symbol Organization

```python
data = {
    'BTCUSDT': {'1h': df_1h, '4h': df_4h},
    'ETHUSDT': {'1h': eth_1h, '4h': eth_4h}
}
```

## Storage and Caching

- Cache data locally for fast access
- Store last update timestamp
- Only fetch new data since last update
- Validate consistency between old and new
