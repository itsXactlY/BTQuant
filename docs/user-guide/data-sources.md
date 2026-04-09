# Data Sources Guide

BTQuant provides 33 data feed modules under `dependencies/backtrader/feeds/`.
The primary backtesting data path uses a PolarsDataLoader with Parquet caching
on top of an MSSQL database; live trading uses CCXT, HotSpine, or exchange-
specific feeds.

## All Available Data Feeds (33 files)

| Feed | File | Description |
|---|---|---|
| `CCXT` | `ccxt.py` | Generic CCXT exchange data (historical + live) |
| `MSSQLData` | `mssql_crypto.py` | SQL Server (BigBrainCentral) with Polars resampling |
| `MSSQLStocks` | `mssql_stocks.py` | SQL Server equity data |
| `HotSpineData` | `hotspine_feed.py` | Shared-memory live trade feed |
| `PolarsData` | `polarfeed.py` | Polars DataFrame feed |
| `PandasData` | `pandafeed.py` | Pandas DataFrame feed (legacy) |
| `BinanceFeed` | `binance_feed.py` | Binance-specific feed |
| `BitgetFeed` | `bitget_feed.py` | Bitget-specific feed |
| `MEXCFeed` | `mexc_feed.py` | MEXC-specific feed |
| `PancakeSwapFeed` | `pancakeswap_feed.py` | PancakeSwap DEX feed |
| `YahooFinance` | `yahoo.py` | Yahoo Finance equities |
| `IBData` | `ibdata.py` | Interactive Brokers |
| `OandaData` | `oanda.py` | Oanda forex/CFD |
| `Quandl` | `quandl.py` | Quandl/Nasdaq Data Link |
| `InfluxFeed` | `influxfeed.py` | InfluxDB time series |
| `CSVGeneric` | `csvgeneric.py` | Generic CSV |
| `BTCsv` | `btcsv.py` | BTC-specific CSV |
| `MT4CSV` | `mt4csv.py` | MetaTrader 4 CSV |
| `SierraChart` | `sierrachart.py` | Sierra Chart data |
| `VChart` | `vchart.py` | VChart format |
| `VChartCSV` | `vchartcsv.py` | VChart CSV |
| `VChartFile` | `vchartfile.py` | VChart binary |
| `VCData` | `vcdata.py` | VC data format |
| `TVFeed` | `tv_feed.py` | TradingView data |
| `Crypto` | `crypto.py` | Generic crypto feed |
| `Blaze` | `blaze.py` | Blaze data |
| `Chainer` | `chainer.py` | Chain multiple feeds |
| `RollOver` | `rollover.py` | Futures roll-over |

## Primary Backtesting Path: MSSQL + PolarsDataLoader

The `backtest()` function in `utils/backtest.py` uses a `PolarsDataLoader` that:

1. Checks a Parquet cache in `.btq_cache/`
2. Falls back to `get_database_data()` from `mssql_crypto.py`
3. Resamples 1-second/1-minute data from the database to any timeframe
4. Wraps the result as an `MSSQLData` (extends `PolarsData`) feed

```python
from backtrader.utils.backtest import backtest, PolarsDataLoader, DataSpec
import polars as pl

# Automatic (recommended):
final_value = backtest(
    MyStrategy,
    coin='BTC',
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1h',
    collateral='USDT',
    init_cash=10000,
)

# Manual:
loader = PolarsDataLoader()
spec = DataSpec(symbol='BTC', interval='1h', start_date='2024-01-01',
                end_date='2024-12-31', collateral='USDT')
df = loader.load_data(spec, use_cache=True)
feed = loader.make_backtrader_feed(df, spec)
```

### DataSpec

```python
@dataclass(frozen=True)
class DataSpec:
    symbol: str              # e.g. 'BTC'
    interval: str            # e.g. '1h', '15m', '1d'
    start_date: str = None   # 'YYYY-MM-DD'
    end_date: str = None     # 'YYYY-MM-DD'
    ranges: List[Tuple[str, str]] = None  # multiple date ranges
    collateral: str = 'USDT'
```

### Caching

Cache directory: `.btq_cache/` (override with `BTQ_CACHE_DIR` env var).
Cache key is `md5(symbol|interval|collateral|ranges)[:12]` -> `.parquet`.
Use `--no-cache` or `--clear-cache` CLI flags to control caching.

### MSSQL get_database_data

```python
from backtrader.feeds.mssql_crypto import get_database_data

df = get_database_data(
    ticker='BTC',           # coin name
    start_date='2024-01-01',
    end_date='2024-12-31',
    time_resolution='1h',   # '1s', '1m', '5m', '15m', '1h', '1d', '1w', '1M'
    pair='USDT',
)
```

- Fetches 1-second or 1-minute rows from SQL Server
- Resamples to the requested timeframe using `pl.group_by_dynamic`
- Returns a Polars DataFrame with columns: `TimestampStart`, `Open`, `High`, `Low`, `Close`, `Volume`

## CCXT Data Feed

```python
from backtrader.feeds.ccxt import CCXT

feed = CCXT(
    exchange='binance',
    symbol='BTC/USDT',
    ohlcv_limit=1000,
    config={},
    retries=5,
)

# Params inherited from DataBase:
# historical=False, backfill_start=False, fromdate, todate
```

States: LIVE -> fetches new candles; HISTORBACK -> fetches historical; OVER -> done.

### CCXT Configuration

Credentials are loaded by `load_ccxt_config()` in `ccxt_config.py`:

```python
from backtrader.ccxt_config import load_ccxt_config

cfg = load_ccxt_config('binance', 'main')
# cfg = {'apiKey': '...', 'secret': '...', 'enableRateLimit': True, ...}
```

Search order:
1. `<venv>/ccxt/{exchange}_{account}.json`
2. `<venv>/ccxt/{exchange}.json`
3. Environment variables: `BTQ_{EXCHANGE}_{ACCOUNT}_API_KEY`, etc.

## HotSpine Live Data

Ultra-low latency shared-memory feed for live trading.

```python
from backtrader.feeds.hotspine_feed import HotSpineData, HotSpineFeed

data = HotSpineData(
    symbol_id=123,
    symbol='BTC/USDT',
    shm_name='/btquant_hotspine',
    batch_mode=False,         # True for batch reads
    poll_interval=0.0001,     # seconds
)

# Or via factory:
from backtrader.feeds.hotspine_feed import create_hotspine_data_feed
data = create_hotspine_data_feed(symbol_id=123, symbol='BTC/USDT')
```

`HotSpineFeed(bt.feed.FeedBase)` wraps HotSpineData as `DataCls`.

Params:
- `shm_name`: shared memory segment name (default `/btquant_hotspine`)
- `symbol_id`: filter by symbol
- `symbol`: symbol string
- `batch_mode`: batch vs single-trade reads
- `poll_interval`: polling interval in seconds

## PolarsData Feed

For loading Polars DataFrames directly:

```python
from backtrader.feeds.polarfeed import PolarsData
import polars as pl

df = pl.read_csv('data.csv')
data = PolarsData(
    dataname=df,
    datetime=0,    # column index or name
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5,
)
```

Supports ISO datetime strings, epoch seconds, and epoch milliseconds.

## Using Data Feeds with Cerebro

```python
import backtrader as bt
from backtrader.feeds.ccxt import CCXT

cerebro = bt.Cerebro()

# CCXT feed
feed = CCXT(exchange='binance', symbol='BTC/USDT')
feed.p.timeframe = bt.TimeFrame.Minutes
feed.p.compression = 60
cerebro.adddata(feed)

# Or Polars DataFrame
from backtrader.feeds.polarfeed import PolarsData
data = PolarsData(dataname=my_polars_df)
cerebro.adddata(data)

cerebro.addstrategy(MyStrategy)
cerebro.run()
```

## Multi-Timeframe

```python
data_1m = CCXT(exchange='binance', symbol='BTC/USDT')
cerebro.adddata(data_1m)

cerebro.resampledata(data_1m, timeframe=bt.TimeFrame.Minutes, compression=5, name='5m')
cerebro.resampledata(data_1m, timeframe=bt.TimeFrame.Minutes, compression=60, name='1h')
```

Or via `backtest()` with `add_mtf_resamples=True`, which adds 5m, 15m, and 60m
resamples automatically.
