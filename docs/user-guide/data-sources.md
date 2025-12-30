# Data Sources Guide

BTQuant supports multiple data sources for comprehensive market data access, from historical backtesting to live trading. This guide covers all available data feeds and their usage.

## Table of Contents

- [CCXT Historical Data](#ccxt-historical-data)
- [SQL Server Data (BigBrainCentral)](#sql-server-data-bigbraincentral)
- [HotSpine Live Data](#hotspine-live-data)
- [CSV and Custom Data](#csv-and-custom-data)
- [Data Feed Classes](#data-feed-classes)
- [Data Quality and Validation](#data-quality-and-validation)
- [Performance Considerations](#performance-considerations)

## CCXT Historical Data

CCXT (CryptoCurrency eXchange Trading Library) provides access to 100+ cryptocurrency exchanges.

### Basic Usage

```python
from backtrader.utils.ccxt_data import get_crypto_data

# Get Bitcoin data from Binance
data = get_crypto_data(
    asset='BTC/USDT',
    start_date='2024-01-01',
    end_date='2024-12-31',
    timeframe='1h',
    exchange='binance'
)

print(f"Retrieved {len(data)} candles")
print(data.head())
```

### Parameters

- **asset**: Trading pair (e.g., 'BTC/USDT', 'ETH/BTC')
- **start_date**: Start date in 'YYYY-MM-DD' format
- **end_date**: End date in 'YYYY-MM-DD' format
- **timeframe**: Candle interval ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
- **exchange**: Exchange name ('binance', 'coinbase', 'kraken', etc.)

### Advanced Features

```python
# Multiple assets
assets = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
data_dict = {}

for asset in assets:
    data_dict[asset] = get_crypto_data(
        asset=asset,
        start_date='2024-01-01',
        end_date='2024-12-31',
        timeframe='1d',
        exchange='binance'
    )

# Custom exchange configuration
data = get_crypto_data(
    asset='BTC/USDT',
    start_date='2024-01-01',
    end_date='2024-01-31',
    timeframe='1h',
    exchange='binance',
    api_key='your_api_key',
    secret='your_secret'
)
```

### Supported Exchanges

Major exchanges include:
- **Binance** (binance, binanceus)
- **Coinbase Pro** (coinbasepro)
- **Kraken** (kraken)
- **Bitfinex** (bitfinex)
- **Huobi** (huobi)
- **OKX** (okx)
- **Bybit** (bybit)
- **KuCoin** (kucoin)
- And 100+ more...

### Error Handling

```python
try:
    data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')
    if data is None:
        print("Failed to retrieve data")
    else:
        print(f"Success: {len(data)} rows")
except Exception as e:
    print(f"Error: {e}")
```

## SQL Server Data (BigBrainCentral)

BigBrainCentral provides institutional-grade market data storage and retrieval.

### Architecture

BigBrainCentral consists of:
- **C++ Collectors**: Ultra-low latency data ingestion
- **SQL Server Storage**: Microsecond-precision canonical storage
- **Python Adapters**: Seamless Backtrader integration

### Basic Usage

```python
from backtrader.feeds.mssql_crypto import get_database_data

# Retrieve historical data
data = get_database_data(
    ticker='BTC',
    start_date='2024-01-01',
    end_date='2024-12-31',
    time_resolution='1h',
    pair='USDT'
)

print(f"Retrieved {len(data)} records from database")
```

### Advanced Queries

```python
# Multiple tickers
tickers = ['BTC', 'ETH', 'ADA']
data_frames = {}

for ticker in tickers:
    data_frames[ticker] = get_database_data(
        ticker=ticker,
        start_date='2024-01-01',
        end_date='2024-12-31',
        time_resolution='1d',
        pair='USDT'
    )

# Custom timeframes
data_15m = get_database_data(
    ticker='BTC',
    start_date='2024-01-01',
    end_date='2024-01-02',
    time_resolution='15m',
    pair='USDT'
)
```

### Database Schema

BigBrainCentral stores data in optimized tables:

- **Trades**: Raw trade data with microsecond timestamps
- **Order Books**: Bid/ask snapshots
- **OHLCV**: Aggregated candles for different timeframes
- **Metadata**: Exchange and symbol information

### Connection Configuration

Configure database connection in `dependencies/backtrader/dontcommit.py`:

```python
# SQL Server Configuration
server = 'localhost'
candle_database = 'BigBrainCentral'
username = 'SA'
password = 'YourStrong!Passw0rd'
driver = '{ODBC Driver 18 for SQL Server}'

connection_string = (
    f'DRIVER={driver};'
    f'SERVER={server};'
    f'DATABASE={candle_database};'
    f'UID={username};'
    f'PWD={password};'
    f'TrustServerCertificate=yes;'
)
```

## HotSpine Live Data

HotSpine provides ultra-low latency live trading data through shared memory.

### Architecture

- **Shared Memory**: Direct memory access for sub-microsecond latency
- **C++ Core**: High-performance data structures
- **Python Bindings**: Seamless Backtrader integration

### Basic Usage

```python
from backtrader.feeds.hotspine_feed import HotSpineData

# Create live data feed
data = HotSpineData(
    symbol_id=123,  # Symbol identifier
    shm_name="/btquant_hotspine",  # Shared memory name
    batch_mode=False,  # Single trade mode for lowest latency
    poll_interval=0.0001  # Polling interval
)

# Add to cerebro
cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(YourStrategy)

# Run live trading
cerebro.run(live=True)
```

### Multi-Symbol Trading

```python
from backtrader.livetrading import livetrade_hotspine_multi_symbol

# Multi-symbol live trading
livetrade_hotspine_multi_symbol(
    symbol_ids=[123, 456, 789],  # Multiple symbol IDs
    strategy=YourMultiSymbolStrategy,
    shm_name="/btquant_hotspine",
    batch_mode=True  # Batch mode for high throughput
)
```

### Performance Characteristics

- **Single Trade Mode**: 6M+ trades/second, <1µs latency
- **Batch Mode**: 14M+ trades/second, ~0.07µs per trade
- **Memory Efficient**: 32 bytes per trade
- **Thread Safe**: Concurrent read/write operations

### HotSpine Data Structure

```python
# Trade data structure
trade = {
    'ts_exchange': int,    # Exchange timestamp (microseconds)
    'ts_local': int,       # Local receive timestamp (microseconds)
    'price': float,        # Trade price
    'size': float,         # Trade size
    'symbol_id': int,      # Symbol identifier
    'side': int            # 0=BUY, 1=SELL
}
```

## CSV and Custom Data

For custom datasets and research data.

### CSV Data

```python
import pandas as pd
from backtrader.feeds.polarfeed import PolarsData

# Load CSV data
df = pd.read_csv('your_data.csv')

# Convert to Backtrader format
data = PolarsData(dataname=df)

# Add to cerebro
cerebro.adddata(data)
```

### Required CSV Format

```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00,50000.0,51000.0,49500.0,50500.0,100.0
2024-01-01 01:00:00,50500.0,51500.0,50000.0,51000.0,150.0
...
```

### Custom Data Feeds

```python
import backtrader as bt

class CustomDataFeed(bt.feeds.DataBase):
    """Custom data feed example"""

    def __init__(self, custom_data_source):
        super().__init__()
        self.custom_data = custom_data_source

    def start(self):
        super().start()
        self._idx = 0

    def _load(self):
        if self._idx >= len(self.custom_data):
            return False

        row = self.custom_data[self._idx]

        # Set OHLCV data
        self.lines.datetime[0] = bt.date2num(row['datetime'])
        self.lines.open[0] = row['open']
        self.lines.high[0] = row['high']
        self.lines.low[0] = row['low']
        self.lines.close[0] = row['close']
        self.lines.volume[0] = row['volume']

        self._idx += 1
        return True
```

### Polars Integration

```python
# PANDAS Dataframe is fully deprecated. Very few dependencies will fully rewritten sooner than later
import polars as pl
from backtrader.feeds.polarfeed import PolarsData

# Polars DataFrame
df = pl.read_csv('market_data.csv')

# Convert datetime if needed
df = df.with_columns(
    pl.col('datetime').str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S')
)

# Create feed
data = PolarsData(dataname=df)
```

## Data Feed Classes

### Backtrader Feed Classes

| Feed Class | Purpose | Data Source |
|------------|---------|-------------|
| `CCXTData` | Historical crypto data | CCXT exchanges |
| `MSSQLData` | Database-stored data | SQL Server |
| `HotSpineData` | Live trading data | Shared memory |
| `PolarsData` | Custom data | DataFrames |
| `YahooData` | Stock data | Yahoo Finance |
| `CSVData` | CSV files | Local files |

### Feed Configuration

```python
# Generic feed configuration
data = bt.feeds.CSVData(
    dataname='data.csv',
    dtformat='%Y-%m-%d %H:%M:%S',
    timeframe=bt.TimeFrame.Minutes,
    compression=1
)

# Custom data feed
data = CustomDataFeed(custom_data_source=my_data)
```

### Multi-Timeframe Data

```python
# Add multiple timeframes
data_1h = bt.feeds.CCXTData(dataname='BTC/USDT', timeframe=bt.TimeFrame.Minutes, compression=60)
data_4h = bt.feeds.CCXTData(dataname='BTC/USDT', timeframe=bt.TimeFrame.Minutes, compression=240)

cerebro.adddata(data_1h)
cerebro.adddata(data_4h, name='4h')
```

## Data Quality and Validation

### Data Validation

```python
def validate_data(data):
    """Validate data quality"""
    if data is None or len(data) == 0:
        return False, "No data available"

    # Check for missing values
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        return False, f"Found {missing_count} missing values"

    # Check for negative prices/volumes
    if (data['close'] <= 0).any():
        return False, "Negative or zero closing prices found"

    if (data['volume'] < 0).any():
        return False, "Negative volumes found"

    # Check for monotonic timestamps
    if not data['datetime'].is_monotonic_increasing:
        return False, "Timestamps not in chronological order"

    return True, "Data validation passed"

# Usage
is_valid, message = validate_data(data)
if not is_valid:
    print(f"Data validation failed: {message}")
```

### Gap Detection

```python
def detect_data_gaps(data, timeframe_minutes=60):
    """Detect gaps in time series data"""
    expected_interval = pd.Timedelta(minutes=timeframe_minutes)

    # Calculate time differences
    time_diffs = data['datetime'].diff()

    # Find gaps larger than expected
    gaps = time_diffs > expected_interval
    gap_indices = data.index[gaps]

    if len(gap_indices) > 0:
        print(f"Found {len(gap_indices)} data gaps:")
        for idx in gap_indices:
            gap_start = data.loc[idx-1, 'datetime']
            gap_end = data.loc[idx, 'datetime']
            gap_duration = gap_end - gap_start
            print(f"  Gap from {gap_start} to {gap_end} ({gap_duration})")

    return gap_indices
```

### Outlier Detection

```python
def detect_outliers(data, threshold=3.0):
    """Detect price outliers using z-score"""
    # Calculate z-scores
    mean_price = data['close'].mean()
    std_price = data['close'].std()
    z_scores = (data['close'] - mean_price) / std_price

    # Find outliers
    outliers = data[abs(z_scores) > threshold]

    if len(outliers) > 0:
        print(f"Found {len(outliers)} price outliers:")
        for idx, row in outliers.iterrows():
            print(f"  {row['datetime']}: {row['close']} (z-score: {z_scores[idx]:.2f})")

    return outliers.index
```

## Performance Considerations

### Data Loading Optimization

```python
# Use efficient data types
data = data.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'float32'
})

# Index on datetime for fast queries
data = data.set_index('datetime')
```

### Memory Management

```python
# Process large datasets in chunks
chunk_size = 10000

for i in range(0, len(data), chunk_size):
    chunk = data.iloc[i:i+chunk_size]
    # Process chunk
    process_data_chunk(chunk)
```

### Database Optimization

```python
# Use indexed queries for SQL Server
query = """
SELECT * FROM btcusdt_klines
WHERE datetime >= ? AND datetime <= ?
ORDER BY datetime
"""

# Use connection pooling
import pyodbc
pool = pyodbc.connect(connection_string, pool_size=10)
```

### Caching Strategies

```python
import pickle
from pathlib import Path

def get_cached_data(symbol, start_date, end_date, cache_dir='./cache'):
    """Cache data to avoid repeated downloads"""
    cache_file = Path(cache_dir) / f"{symbol}_{start_date}_{end_date}.pkl"

    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Fetch data
    data = get_crypto_data(symbol, start_date, end_date, '1h', 'binance')

    # Cache it
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data
```

## Data Source Comparison

| Feature | CCXT | BigBrainCentral | HotSpine | CSV |
|---------|------|-----------------|----------|-----|
| **Latency** | High (API) | Medium (DB) | Ultra-low (Memory) | N/A |
| **Historical Depth** | Limited | Unlimited | Limited | Unlimited |
| **Live Trading** | No | No | Yes | No |
| **Cost** | Free/Paid APIs | Infrastructure | Infrastructure | Free |
| **Setup Complexity** | Low | High | High | Low |
| **Data Quality** | Variable | High | High | User-dependent |

## Best Practices

### Data Source Selection
1. **Backtesting**: Use CCXT for quick tests, BigBrainCentral for production
2. **Live Trading**: HotSpine for ultra-low latency requirements
3. **Research**: BigBrainCentral for microstructure analysis
4. **Development**: CSV for controlled testing scenarios

### Data Validation
1. **Always validate** data before using in strategies
2. **Check for gaps** in time series data
3. **Monitor data quality** continuously
4. **Handle missing data** gracefully

### Performance
1. **Cache frequently used data** to reduce API calls
2. **Use appropriate data types** for memory efficiency
3. **Index databases** for fast queries
4. **Process data in chunks** for large datasets

### Reliability
1. **Implement retry logic** for API failures
2. **Monitor data feeds** for outages
3. **Have fallback data sources** when possible
4. **Log data issues** for debugging

This comprehensive data sources guide ensures you can access and utilize market data effectively across all BTQuant use cases.
