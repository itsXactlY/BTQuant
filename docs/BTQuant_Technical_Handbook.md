# BTQuant Technical Handbook

> **Scope**: This document captures the complete picture of BTQuant's capabilities, moving parts, and integration points so that quantitative developers can install, extend, test, and run the platform with confidence.

## Table of Contents
1. [Platform Overview](#platform-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Environment Provisioning](#installation--environment-provisioning)
4. [Key Directory Map](#key-directory-map)
5. [Data Ingestion Pipelines](#data-ingestion-pipelines)
6. [Backtesting & Optimization](#backtesting--optimization)
7. [Live & Paper Trading Stack](#live--paper-trading-stack)
8. [Strategy Development Lifecycle](#strategy-development-lifecycle)
9. [Analytics, Reporting & Monitoring](#analytics-reporting--monitoring)
10. [Testing & Quality Gates](#testing--quality-gates)
11. [Security, Secrets & Operational Safety](#security-secrets--operational-safety)
12. [Troubleshooting & Performance Tuning](#troubleshooting--performance-tuning)
13. [Further Reading & External Resources](#further-reading--external-resources)

---

## Platform Overview
BTQuant is an institutional-grade algorithmic trading framework that unifies historical research, forward simulation, and live execution around an extended fork of Backtrader. The repository ships with:

- **Ultra-low-latency market data pipelines** that cover CCXT-compatible exchanges, proprietary WebSocket feeds (Binance, Bitget, MEXC, PancakeSwap), and Microsoft SQL Server tick stores.
- **Turn-key backtesting utilities** that wrap Cerebro with rich analyzers, optimization helpers, and QuantStats reporting.
- **Strategy and store libraries** for discretionary and systematic trading, including custom analyzers, sizers, and feed adapters.
- **Built-in precision DCA engine** embedded in the shared `BaseStrategy`, so authors can focus purely on signal design while BTQuant manages staged execution, position accounting, and live-trading plumbing.
- **Operational tooling** (installers, database scripts, WebSocket runners) required for production deployment.

---

## System Architecture
BTQuant is structured as layered services that can be composed as needed.

### 1. Data Layer
- **Historical CCXT ingestion** via `dependencies/backtrader/utils/ccxt_data.py`, producing Polars DataFrames for arbitrary symbols/timeframes.
- **Tick-level SQL Server access** using `dependencies/backtrader/feeds/mssql_crypto.py` and the C++ extension in `dependencies/MsSQL/fast_mssql.cpp` for microsecond resolution queries.
- **CSV/Polar feeds** in `dependencies/backtrader/feeds/` to import custom datasets (e.g., `csvgeneric.py`, `polarfeed.py`).
- **Real-time WebSocket connectors** implemented in `dependencies/backtrader/feeds/*_feed.py` and matching `stores` for exchange-native protocols.

### 2. Strategy & Analytics Layer
**Base strategy scaffold**
- **Strategies** reside in `dependencies/backtrader/strategies/` and extend the Backtrader API with custom indicators, state machines, and machine learning components.
- `BaseStrategy` (`dependencies/backtrader/strategies/base.py`) centralizes staged order routing, Telegram/Discord alerting, JackRabbitRelay/PancakeSwap exchange bridges, and the precision DCA ledger so individual strategies can remain lean.
- **Analyzers & observers** (`dependencies/backtrader/analyzers`, `dependencies/backtrader/observers`) capture performance metrics such as drawdowns, SQN, and PyFolio artifacts.
- **Commission models** in `dependencies/backtrader/comminfo.py` reproduce venue-specific fee structures.

### 3. Execution & Orchestration Layer
- **Backtesting orchestration** is centralized in `dependencies/backtrader/utils/backtest.py`, wrapping Cerebro configuration, optimization, and QuantStats report generation.
- **Live trading stores** (`dependencies/backtrader/stores/*.py`) encapsulate session management, order routing, and WebSocket subscriptions for each venue.
- **Operational entry points** under `Examples/` demonstrate how to combine feeds, strategies, analyzers, and execution venues.

### 4. Tooling & Automation
- **Installers** (`Installers/*.sh`) automate environment bootstrapping, dependency installation, and database setup.
- **Auxiliary utilities** like `dependencies/backtrader/utils/date.py` and `dateintern.py` provide shared date/clock logic.

---

## Installation & Environment Provisioning
BTQuant targets **Python 3.12+** (per `dependencies/setup.py`) and Linux distributions with GCC/Clang toolchains.

1. **Automated bootstrap**
   ```bash
   cd ~
   bash Installers/install.sh
   ```
   - Detects the host distribution, installs build prerequisites (`build-essential`, `python3-dev`, `unixodbc-dev`, etc.).
   - Clones the repository with submodules, creates a virtualenv (`.btq`), and installs the bundled packages: Backtrader fork, QuantStats fork, and the MSSQL extension.

2. **Manual setup checklist**
   - Install system packages: compiler suite, `unixODBC`, `pybind11`, `tk`, and vendor SDKs for GPU acceleration (optional).
   - Create and activate a virtual environment.
   - `pip install ./dependencies` to build the Backtrader fork (pulls Python deps such as `ccxt`, `polars`, `web3`, `telethon`).
   - Build the `fast_mssql` extension via `pip install ./dependencies/MsSQL` if SQL Server support is required.
   - Populate `dependencies/backtrader/dontcommit.py` with exchange credentials, webhook URLs, and database connection strings.

3. **Database provisioning**
   - Use `Installers/install_database.sh` (MSSQL scripts) to prepare schema/tables for tick ingestion.
   - Ensure SQL Server exposes tables named `<SYMBOL><COLLATERAL>_klines` (e.g., `BTCUSDT_klines`) with microsecond timestamps as expected by `MSSQLData`.

---

## Key Directory Map
| Path | Description |
|------|-------------|
| `Examples/` | Executable samples for CCXT backtests, MSSQL bulk jobs, and native WebSocket trading loops. |
| `Installers/` | Shell scripts for environment setup, updates, and database maintenance. |
| `dependencies/backtrader/` | Core Backtrader fork with custom feeds, stores, analyzers, observers, strategies, and utilities. |
| `dependencies/MsSQL/` | Source for the `fast_mssql` pybind11 extension enabling ultra-fast SQL Server queries. |
| `dependencies/backtrader/utils/` | Shared utilities for CCXT ingestion, sequential/bulk data fetchers, and date helpers. |
| `dependencies/backtrader/feeds/` | Data feed implementations (MSSQL, CCXT, Binance, Bitget, MEXC, PancakeSwap, CSV, Polar). |
| `dependencies/backtrader/stores/` | Live trading stores for each supported venue. |
| `dependencies/backtrader/strategies/` | Strategy catalogue, including machine-learning driven and discretionary systems. |
| `dependencies/backtrader/analyzers/` | Portfolio performance analyzers (Sharpe, DrawDown, PyFolio, CustomSQN). |
| `dependencies/backtrader/dontcommit.py` | Secrets placeholder for private credentials and operational hooks. |

---

## Data Ingestion Pipelines

### CCXT Historical Downloader
- Use `backtrader.utils.ccxt_data.get_crypto_data(asset, start_date, end_date, timeframe, exchange)` to pull OHLCV candles into a Polars DataFrame.
- Handles intraday vs. daily parsing, deduplicates overlapping windows, and stitches multiple CCXT calls together.
- Retries by advancing one day when exchanges respond with empty payloads.

### MSSQL Tick Store
- `backtrader.feeds.mssql_crypto.get_database_data` queries SQL Server for microsecond-resolution candles and resamples them with Polars `group_by_dynamic`.
- Requires `fast_mssql` (pybind11) and a valid ODBC connection string defined in `dontcommit.py`.
- Automatically infers special `_1s` tables for sub-second archives.

### Native WebSocket Streams
- Exchange-specific feeds (`binance_feed.py`, `bitget_feed.py`, `mexc_feed.py`, `pancakeswap_feed.py`) pair with stores to provide real-time bars or ticks.
- Designed for **true** tick-by-tick streaming with reconnection logic and latency-sensitive buffering.

### CSV & Polar Imports
- `feeds/csvgeneric.py`, `feeds/polarfeed.py`, and `feeds/btcsv.py` enable ingestion of local research datasets when proprietary infrastructure is unavailable.

---

## Backtesting & Optimization

### Quick Start
```python
from backtrader.utils.backtest import backtest
from backtrader.strategies.Vumanchu_A import VuManchCipher_A
from backtrader.utils.ccxt_data import get_crypto_data

data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-08', '15m', 'binance')

if __name__ == '__main__':
    backtest(
        strategy=VuManchCipher_A,
        data=data,
        init_cash=1_000,
        quantstats=True,
        plot=True,
        asset_name='BTC/USDT'
    )
```

### Batch Optimization
- Call `optimize_backtest` with iterable keyword arguments (e.g., `breakout_period=[20, 40, 55]`) to spawn Cerebro optimizations with progress feedback.
- Analyzer stack includes TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer, Returns, CustomSQN, and PyFolio hooks for performance introspection.
- When `quantstats=True`, reports are exported to `QuantStats/<asset>_<date>_<time>.html` using the custom QuantStats fork.

### Commission & Cash Management
- Defaults: `INIT_CASH = 100_000`, `COMMISSION_PER_TRANSACTION = 0.00075`. Override via function arguments.
- Commission models can be tailored by adjusting `dependencies/backtrader/comminfo.py` or passing custom `CommissionInfo` instances to the broker.

### Data Provisioning Modes
- **Direct DataFrame input**: pass a Polars/Pandas DataFrame into `backtest`; the helper wraps it with `CustomData`.
- **Backtrader feed objects**: supply `DataBase` subclasses (e.g., `MSSQLData`) for zero-copy ingestion.

---

## Live & Paper Trading Stack

### Exchange Stores
- `dependencies/backtrader/stores/binance_store.py`, `bitget_store.py`, `mexc_store.py`, and `pancakeswap_store.py` expose session factories with authentication, order execution, and WebSocket routing tailored to each venue.
- Stores integrate with strategies via standard Backtrader live-trading APIs (brokers, notifications, observers).

### Operational Examples
- `Examples/Trading_CCXT.py` – CCXT spot trading skeleton with strategy wiring.
- `Examples/Trading_Websocket_Binance.py` / `Trading_Websocket_Bitget.py` / `Trading_Websocket_Tickdata_Mexc.py` – native WebSocket event loops showcasing tick-level execution.
- `Examples/Backtest_MsSQL.py` & `Backtest_Bulk_MsSQL.py` – demonstrate large-scale historical pulls from SQL Server.

### Forward Testing via JackRabbitRelay
- Configure relay parameters in `dontcommit.py` (`identify`, `jrr_webhook_url`, `jrr_order_history`) to mirror exchange behaviour without risking capital.

---

## Strategy Development Lifecycle

1. **Create a strategy** in `dependencies/backtrader/strategies/` inheriting from `BaseStrategy` (for full DCA/live orchestration) or `bt.Strategy` (for minimal research prototypes).
2. **Define indicators** using the rich catalogue in `dependencies/backtrader/indicators/` or add new ones under that directory.
3. **Add analyzers/observers** if additional metrics or telemetry are needed (e.g., custom risk dashboards).
4. **Register parameters** so `optimize_backtest` can sweep ranges simply by passing iterables.
5. **Document usage** and provide sample configuration within `Examples/` for discoverability.

### BaseStrategy advantages
- **Precision DCA for free** – toggle `self.DCA = True` to unlock multi-slice scaling, automatic fill tracking, and rejection handling managed inside `BaseStrategy`.
- **Unified live adapters** – PancakeSwap Web3 queues, JackRabbitRelay order routing, and alert loops are initialized once by the base class, letting new strategies stay strategy-focused.
- **Consistent telemetry** – shared attributes (e.g., `self.buy_executed`, `self.average_entry_price`, win/loss counters) give every derived strategy identical bookkeeping for reporting and analytics.

### Template
```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = dict(
        lookback=20,
        risk_per_trade=0.01,
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.lookback)

    def next(self):
        if not self.position and self.data.close[0] > self.sma[0]:
            size = self.broker.getcash() * self.p.risk_per_trade / self.data.close[0]
            self.buy(size=size)
        elif self.position and self.data.close[0] < self.sma[0]:
            self.close()
```

### Packaging Considerations
- Strategies can import machine learning models (scikit-learn, Keras) already listed in the dependency tree.
- Persist heavy models with `joblib` (pre-imported in `dontcommit.py`) for reuse across sessions.

### From PoC ML scripts to self-aware neural stacks
- Start with the ML primer in `Examples/Polar Dataframe Manipulation/Polars_kNN_Example-Suite.py`, which demonstrates feature engineering in Polars, model training with scikit-learn, and integrating predictions inside a `BaseStrategy` derivative while leaning on built-in DCA orchestration.
- Graduate to advanced neural pipelines by swapping in deep-learning architectures (e.g., PyTorch, TensorFlow) – the modular loader pattern in the examples keeps data preparation, model inference, and order management isolated so you can iterate on each layer independently while the base infrastructure continues to handle execution and alerts.

---

## Analytics, Reporting & Monitoring
- **QuantStats** integration: set `quantstats=True` in `backtest` or `optimize_backtest` to produce HTML tear sheets via `quantstats_lumi` fork.
- **PyFolio Analyzer**: accessible through `strategy_result.analyzers.getbyname('pyfolio')` returning returns, positions, transactions, and leverage series.
- **Rich Console Feedback**: the backtest utilities employ `rich` progress bars for data loading and optimization loops.
- **Telemetry Hooks**: `dontcommit.py` contains placeholders for Discord and Telegram notifications—wire these into strategies or observers for alerting.

---

## Testing & Quality Gates
- **Unit tests** (pytest) can live under `dependencies/backtrader/utils/tests/` or adjacent to target modules.
- Recommended coverage targets:
  - Date conversions (`utils/date.py`, `utils/ccxt_data.py`).
  - Exchange error handling for WebSocket feeds and stores.
  - Strategy logic—use small synthetic datasets to validate entry/exit conditions.
- Continuous integration should run `pytest` plus linters (e.g., `ruff`, `black`) across the repo to guard against regression in the Backtrader fork.

---

## Security, Secrets & Operational Safety
- Store all credentials in `dependencies/backtrader/dontcommit.py` (excluded from VCS). The template defines slots for Web3 private keys, exchange identifiers, and messaging webhooks.
- Restrict file permissions on the virtualenv and database config to prevent credential leakage.
- When deploying to production, move secrets into environment variables or dedicated vaults and import them within `dontcommit.py` instead of hardcoding.
- Audit the `Installers/` scripts before running in sensitive environments—adjust package repositories and TLS settings as required by corporate policy.

---

## Troubleshooting & Performance Tuning
- **Data gaps**: If `get_crypto_data` returns `None`, confirm the symbol/timeframe is supported and the exchange name matches `ccxt.exchanges` exactly.
- **Slow SQL queries**: Verify MSSQL indices and confirm the `ENABLE_PARALLEL_PLAN_PREFERENCE` hint aligns with your server edition; adjust or remove if plans regress.
- **WebSocket drops**: Customize reconnection and heartbeat logic inside the respective `*_feed.py` to honour venue-specific rate limits.
- **Memory pressure**: Use `fetch_all_data_sequential` for staged coin loading and enable Python's `gc.collect()` after large optimization sweeps (already imported in `backtest.py`).
- **Plotting environments**: Matplotlib is forced to `'Agg'` in `backtest.py` for headless execution. Switch to an interactive backend locally if desired.

---

## Further Reading & External Resources
- **Primary Backtrader Documentation**: https://www.backtrader.com/docu/
- **CCXT Manual**: https://docs.ccxt.com/
- **QuantStats Reference**: https://github.com/ranaroussi/quantstats
- **Telethon Guide** (for Telegram integrations): https://docs.telethon.dev/
- **Web3.py Documentation**: https://web3py.readthedocs.io/

---

*BTQuant empowers quants to blend historical research, real-time intelligence, and live execution on top of hardened infrastructure. Use this handbook as the launchpad for bespoke strategies and institutional deployments.*
