# BTQuant Frequently Asked Questions

## General

### What is BTQuant?

BTQuant is a backtesting and optimization framework built on a forked Backtrader engine. It provides a CLI (`btq`), a strategy base class (`BaseStrategy`), data feeds for CCXT exchanges and SQL Server, Optuna-based parameter optimization, and QuantStats reporting.

### What languages and platforms does it support?

BTQuant is written in Python 3.13 with some C++ components (Fast_MSSQL driver, CCAPI). It targets Linux only (Ubuntu/Debian, Fedora, CentOS/RHEL, Arch-based distributions).

### Do I need SQL Server?

No. SQL Server is optional. You can use CCXT data feeds (fetched from any CCXT-supported exchange over the internet) without any database. SQL Server is needed only if you want to use the `bulk_backtest` auto-discovery feature, the `btq list coins` command, or store large amounts of historical data locally.

### Do I need to know C++?

No. The C++ components (Fast_MSSQL driver, CCAPI market data collector) are pre-built or compiled during installation. You only need Python to write strategies and run backtests.

### Is Docker supported?

No. The project FAQ explicitly states Docker is not supported and will not be.

## Installation

### Where does BTQuant get installed?

- Repository: wherever you clone it
- Virtual environment: `~/.btq/` (Python 3.13)
- CCAPI binaries: `~/bin/`
- Data cache: `.btq_cache/` in the working directory

### What does `install_all.sh` actually install?

The script installs:
1. System build dependencies (compiler, cmake, libraries)
2. Microsoft SQL Server (if not already running)
3. SQL Server ODBC driver and sqlcmd tools
4. BTQuant Python package and all dependencies into `~/.btq` venv
5. Fast_MSSQL C++ driver (pre-compiled or built from source)
6. BTQ_MarketData database initialization
7. CCAPI C++ market data collector and hotspine binaries

### What Python packages does BTQuant depend on?

From `dependencies/setup.py`:
- ccxt, pybind11, pyodbc, websockets, websocket-client 1.8.0
- Web3, matplotlib, numpy, polars, pyarrow
- telethon, scikit-learn, keras, pytz, optuna

### How do I activate the environment?

```bash
source ~/.btq/bin/activate
```

### Can I install on Windows or macOS?

No. The project targets Linux only. The installer script uses Linux-specific package managers and system paths.

## Data

### Which exchanges can I get data from?

Any exchange supported by CCXT (100+ exchanges). Common ones: binance, bybit, kraken, coinbase, mexc, kucoin.

```python
from backtrader import get_crypto_data
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')
```

### What timeframes are supported?

Depends on the exchange. Common timeframes: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`.

### How does data caching work?

When fetching from SQL Server, data is cached as Parquet files (zstd compressed) in `.btq_cache/`. The cache key is derived from symbol, interval, collateral, and date range. CCXT data fetched via `get_crypto_data()` is not cached by default.

Clear the cache:
```bash
btq --clear-cache backtest --coin BTC --strategy MyStrategy
```

Disable caching:
```bash
btq --no-cache backtest --coin BTC --strategy MyStrategy
```

Custom cache directory:
```bash
export BTQ_CACHE_DIR=/path/to/cache
```

### Can I use my own CSV data?

Yes. Pass a Polars DataFrame via `PolarsFeed` or use `get_database_data()` for SQL Server data. You can also load CSVs with Polars and pass them to the backtest engine.

## Strategies

### How do I write a strategy?

Extend `BaseStrategy` from `backtrader.strategies.base` and override these methods:

- `buy_or_short_condition()` -- entry logic
- `dca_or_short_condition()` -- dollar-cost-averaging or adding to position
- `sell_or_cover_condition()` -- exit logic
- `check_stop_loss()` -- custom stop loss

Call `self.create_order('BUY')` to enter and `self.close_order(order_tracker)` to exit.

See the [Quick Start Guide](quickstart.md) for a complete example.

### What parameters does BaseStrategy accept?

```
init_cash, exchange, account, asset, amount, coin, collateral,
debug, capture_data, backtest, bulk, optuna, quantstats,
use_stoploss, pnl, final_value, channel, symbol,
stop_loss, stop_trail, take_profit, percent_sizer,
order_cooldown, enable_alerts, alert_channel
```

### How do I list available strategies?

```bash
btq list strategies
```

Or programmatically:
```python
from backtrader.btquant import list_available_strategies
strategies = list_available_strategies()
```

### How does position sizing work?

When `percent_sizer` is set (e.g., 0.1), each trade uses that fraction of available cash:

```
size = (available_cash * percent_sizer) / current_close_price
```

In live trading, minimum order values are enforced per exchange (Binance: $5.50, MEXC: $1.10, PancakeSwap: $0.00001).

## Backtesting

### How do I run a backtest?

Python:
```python
from backtrader.utils.backtest import backtest
result = backtest(MyStrategy, coin='BTC', start_date='2024-01-01',
                  end_date='2024-01-31', interval='1h', init_cash=1000)
```

CLI:
```bash
btq backtest --coin BTC --strategy MyStrategy --interval 1h --start 2024-01-01 --end 2024-01-31
```

### What does the backtest function return?

A float representing the final portfolio value.

### What analyzers are included?

The `backtest()` function automatically adds:
- TimeReturn
- SharpeRatio
- DrawDown
- TradeAnalyzer
- Returns
- CustomSQN
- PyFolio (unless multi-timeframe resampling is enabled)

### How do I generate a QuantStats report?

CLI: `btq backtest --coin BTC --strategy MyStrategy --quantstats`

Python: `backtest(MyStrategy, ..., quantstats=True)`

Reports are saved as HTML files in the `QuantStats/` directory.

### What is bulk backtesting?

Bulk backtesting runs a strategy across multiple coins in parallel using `concurrent.futures`. If no coins are specified, it auto-discovers all available coins from the SQL Server database.

CLI: `btq bulk --strategy MyStrategy --interval 1h --workers 8`

### What does bulk_backtest return?

A list of dictionaries, each containing:
- `coin`, `asset`, `final_value`, `pnl`, `return_pct`, `status` (success/failed/skipped)

## Optimization

### How does optimization work?

BTQuant uses Optuna for hyperparameter optimization. Each trial runs a backtest with sampled parameters. Pruning (hyperband or median) stops unpromising trials early.

### What are the parameter space modes?

- **default**: Strategy's default parameter space
- **aggressive**: Strategy's `param_space_aggressive()` function (if defined)
- **conservative**: Strategy's `param_space_conservative()` function (if defined)

### How do I run optimization?

```bash
btq optimize --coin BTC --strategy MyStrategy --trials 200
btq optimize --coin BTC --strategy MyStrategy --trials 200 --aggressive
btq optimize --coins BTC,ETH --strategy MyStrategy --trials 100 --conservative
```

### What pruning algorithms are available?

- `hyperband` (default)
- `median`
- `none` (disabled)

### What is the minimum trades threshold?

Default: 30 trades. Optimization trials with fewer than `--min-trades` trades are considered invalid. Set with `--min-trades`.

## Live Trading

### Is live trading implemented?

The `btq live` CLI mode exits with "Live trading not yet implemented." However, live trading is partially implemented in Python for:

- **PancakeSwap** (Web3/BSC via `PancakeSwapV2DirectOrderBase`)
- **JackRabbitRelay** (via `JrrBroker`)

### Which exchanges support live trading?

Through JackRabbitRelay: any exchange JRR supports. Through PancakeSwap: Binance Smart Chain DEX trading.

### How do alerts work?

When `enable_alerts=True` on a strategy, BTQuant initializes Telegram and Discord alert services using credentials from `dontcommit.py`. Alerts are sent via `self.send_alert(message)`.

## Configuration

### Where are credentials stored?

In `dependencies/backtrader/dontcommit.py`. This file contains:
- JackRabbitRelay settings
- Web3/Solana wallet keys
- Discord webhook URL
- Telegram API credentials
- SQL Server connection strings

This file should not be committed to version control.

### What is the HotSpine shared memory?

HotSpine is a C++ component that writes market data to shared memory at `/dev/shm/btquant_hotspine` for ultra-low latency access. It is built from the CCAPI example during installation.

## Troubleshooting

### "Could not import strategy: X"

Make sure the strategy class exists in `backtrader.strategies`. Use `btq list strategies` to see available strategies.

### "Error: --strategy required"

The `--strategy` flag is required for backtest, bulk, and optimize modes.

### "No data for BTC 1h 2024-01-01->2024-01-31"

The SQL Server database does not have data for that coin/timeframe/date range. Check available coins with `btq list coins`.

### "Failed to get coins from database"

SQL Server is not running or `dontcommit.py` connection settings are wrong. Verify with:
```bash
sudo systemctl status mssql-server
```

### Backtest returns no trades

Check that your strategy's `buy_or_short_condition()` returns `True` and actually calls `self.create_order()`. Enable debug mode with `--debug`.

### Plot not showing

The `--plot` flag calls `cerebro.plot()`. On headless servers, matplotlib may not have a display backend. The plot will still be generated if a display is available.