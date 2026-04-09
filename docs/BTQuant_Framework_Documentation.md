# BTQuant Framework

Concise overview of BTQuant's components. For details, see the linked guides.

## What BTQuant Is

A fork of backtrader for crypto trading. Adds:
- CLI (btq) for backtesting, bulk runs, Optuna optimization
- BaseStrategy with DCA, OrderTracker, CSV persistence
- Live trading via JackRabbitRelay, CCXT, PancakeSwap, HotSpine
- SQL Server storage (fast_mssql C++ driver)
- Polars-based data pipeline and transparency tooling
- 17 pre-built strategies

## Core: dependencies/backtrader/

Everything lives here. The rest of the repo supports it.

| Module | Purpose |
|--------|---------|
| `btquant.py` | CLI entry point (btq command) |
| `cerebro.py` | Backtest engine |
| `strategies/base.py` | BaseStrategy + OrderTracker |
| `strategies/*.py` | 17 strategy implementations |
| `brokers/` | JrrBroker, CCXTBroker, BackBroker |
| `feeds/` | 28 data feeds (CCXT, MSSQL, HotSpine, CSV, Polars, TradingView) |
| `hotspine/` | Shared memory reader (HotSpineReader) |
| `bigbraincentral/` | SQL Server storage (MarketDataStorage) |
| `livetrading.py` | Live trading entry points |
| `ccxt_config.py` | Exchange API key loading |
| `TransparencyPatch.py` | Indicator calculation capture |
| `dontcommit.py` | Credentials (gitignored) |

## Strategies

All strategies extend BaseStrategy and override condition methods:

- `buy_or_short_condition()` - entry logic
- `dca_or_short_condition()` - DCA/add to position
- `sell_or_cover_condition()` - exit logic
- `check_stop_loss()` - custom stop loss (optional)

Available: Aligator_supertrend, MACD_ADX, NearestNeighbors_RationalQuadraticKernel,
OrderChain, Order_Chain_Kioseff_Trading, QQE_Hullband_VolumeOsc,
SMA_Cross_MESAdaptive_Prime, SMA_Cross_Simple, ST_RSX_ASI,
StagedConvergenceStrategy, SineWeightZeroLagQQEVolMesaAdaptive,
SuperTrend_Scalp, VuManchCipher_A, VuManchCipher_B,
pancakeswap_dca_marketmaker, pancakeswap_orders, jrr_orders

See [Strategies Guide](user-guide/strategies.md) for details.

## Data Feeds

| Feed | Source |
|------|--------|
| CCXT | 100+ exchanges via ccxt library |
| MsSqlCrypto | SQL Server OHLCV data |
| HotSpineData | Shared memory (live) |
| PolarsFeed | Polars DataFrame |
| PolarsDataLoader | SQL + Parquet caching |
| TradingViewFeed | TradingView data |
| BinanceFeed, BitgetFeed, MexcFeed | Exchange-specific |
| PancakeSwapFeed | DEX data |
| CSV feeds | csvgeneric, btcsv |

See [Data Sources Guide](user-guide/data-sources.md) for details.

## Live Trading

| Function | Backend |
|----------|---------|
| `livetrade_ccxt()` | CCXT exchange via CCXTBroker |
| `livetrade_binance()` | Binance websocket |
| `livetrade_mexc()` | MEXC websocket |
| `livetrade_bitget()` | Bitget websocket |
| `livetrade_web3()` | PancakeSwap DEX |
| `livetrade_hotspine()` | HotSpine shared memory |
| `livetrade_tv()` | TradingView + JRR |

All functions create a Cerebro, add data + strategy with `backtest=False`,
and run with `live=True`.

See [API Reference](technical/api-reference.md) for signatures.

## HotSpine

Shared memory for C++ market data collectors. Python reads via `HotSpineReader`.

Data struct (HotTrade): exchange timestamp, local timestamp, price, size, symbol_id, side.

See [HotSpine Technical](technical/hotspine.md) for struct details and configuration.

## SQL Server

Three databases:
- `BinanceData` - historical candle data
- `OptunaBT` - optimization results
- `BTQ_MarketData` - live market data

Storage via `fast_mssql` C++ driver or `pyodbc`.

See [BigBrainCentral Technical](technical/bigbraincentral.md) for schema.

## Transparency

TransparencyPatch captures all indicator intermediate values per bar.
Output: Polars DataFrame with OHLCV + every indicator attribute.

See [Indicator Transparency](BTQuant_Indicator_Transparency.md) for usage.

## Configuration

- `dontcommit.py` - credentials (JRR, Telegram, Discord, SQL Server)
- `ccxt_config.py` - exchange API keys (JSON files or BTQ_ env vars)
- `hotspine/config.py` - HotSpine shared memory settings

See [Configuration](technical/configuration.md) for all options.
