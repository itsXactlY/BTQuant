# Backtesting Guide

BTQuant's backtesting engine is built on top of backtrader's `Cerebro` with
custom extensions for caching, analysis, and CLI integration.

## CLI (btq)

```bash
# Single backtest
btq backtest --coin BTC --strategy SMA_Cross_Simple --interval 1h \
    --start 2024-01-01 --end 2025-01-01 --cash 10000 --plot

# Multiple coins
btq backtest --coins BTC,ETH,BNB --strategy MACD_ADX --interval 15m

# Bulk (auto-discover coins)
btq bulk --interval 1h --workers 8 --strategy Aligator_supertrend

# Optimization
btq optimize --coin BTC --strategy QQE_Hullband_VolumeOsc \
    --trials 200 --aggressive --workers 6

# List resources
btq list strategies
btq list coins --collateral USDT
```

### CLI Arguments

**Mode** (required, first arg): `backtest`, `bulk`, `optimize`, `list`, `live`

| Group | Flag | Default | Description |
|---|---|---|---|
| Common | `--coin` / `--symbol` | - | Single coin (e.g. BTC) |
| Common | `--coins` / `--symbols` | - | Comma-separated coins |
| Common | `--strategy` / `--strat` | - | Strategy class name |
| Common | `--collateral` / `--pair` | USDT | Trading pair |
| Common | `--interval` / `--timeframe` / `--tf` | 15m | Timeframe |
| Common | `--start` / `--start-date` | - | Start date YYYY-MM-DD |
| Common | `--end` / `--end-date` | 2025-01-01 | End date YYYY-MM-DD |
| Capital | `--cash` / `--capital` | 1000 | Initial capital |
| Capital | `--commission` | 0.00075 | Commission rate |
| Capital | `--leverage` | 1 | Leverage multiplier |
| Capital | `--slippage` | 5.0 | Slippage in bps |
| Output | `--plot` / `-p` | false | Show plot |
| Output | `--quantstats` / `-q` | false | Generate QuantStats HTML report |
| Output | `--debug` / `-d` | false | Debug output |
| Output | `--verbose` / `-v` | false | Verbose output |
| Output | `--save` | false | Save results to file |
| Output | `--output` / `-o` | auto | Output filename |
| Bulk | `--workers` / `-j` | 8 | Parallel workers |
| Optimize | `--trials` / `-n` | 200 | Optuna trials |
| Optimize | `--opt-workers` | same as --workers | Parallel opt workers |
| Optimize | `--study-name` | auto | Optuna study name |
| Optimize | `--aggressive` | false | Aggressive param space |
| Optimize | `--conservative` | false | Conservative param space |
| Optimize | `--multi-period` | false | Multi-period validation |
| Optimize | `--min-trades` | 30 | Min trades for valid trial |
| Optimize | `--pruner` | hyperband | Pruner: hyperband, median, none |
| Optimize | `--seed` | 42 | Random seed |
| Strategy | `--params` | - | JSON or key=value params |
| Strategy | `--take-profit` / `--tp` | - | Take profit % |
| Strategy | `--stop-loss` / `--sl` | - | Stop loss % |
| Strategy | `--position-size` / `--size` | - | Position size % of capital |
| Misc | `--exchange` | - | Exchange name |
| Misc | `--no-cache` | false | Disable caching |
| Misc | `--clear-cache` | false | Clear cache before run |

## backtest() Function

```python
from backtrader.utils.backtest import backtest

def backtest(
    strategy,                # strategy class
    data=None,               # pre-loaded data feed (optional)
    coin=None,               # coin ticker (e.g. 'BTC')
    start_date="1970-01-01",
    end_date="2030-12-31",
    interval=None,           # timeframe string (e.g. '1h')
    collateral="USDT",
    commission=0.00075,
    init_cash=100000.0,
    plot=False,
    quantstats=False,
    asset_name=None,
    bulk=False,
    show_progress=True,
    exchange=None,
    slippage_bps=5,
    min_qty=0.0,
    qty_step=1.0,
    price_tick=None,
    params=None,             # dict of strategy params
    add_mtf_resamples=False, # add 5m/15m/60m resamples
    **kwargs,
)
```

Returns `final_value` (float) -- the portfolio value after the backtest.

### What it does

1. Loads data via `PolarsDataLoader` (MSSQL + Parquet cache) or uses provided
   `data` feed.
2. Creates `bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)`.
3. Adds the data feed and strategy with merged params.
4. Sets broker: cash, commission, slippage.
5. Adds analyzers: `TimeReturn`, `SharpeRatio`, `DrawDown`, `TradeAnalyzer`,
   `Returns`, `CustomSQN`, and optionally `PyFolio`.
6. Adds observers: `Value`, `DrawDown`, `Cash`.
7. Runs `cerebro.run()`.
8. Prints results table (trades, win rate, P&L, drawdown, return).
9. Optionally generates QuantStats HTML report.
10. Optionally plots with `cerebro.plot()`.
11. Cleans up with `gc.collect()`.

### Results printed

```
BACKTEST RESULTS - BTC/USDT
Total Trades: 42
Winning Trades: 28
Losing Trades: 14
Win Rate: 66.7%
Net P&L: $1234.56
Max Drawdown: 8.32%
Final Portfolio Value: $11234.56
Total P/L: $1234.56
Return: 12.35%
```

## backtest_with_leverage()

Same interface as `backtest()` but with cross-margin simulation:

```python
from backtrader.utils.backtest import backtest_with_leverage

final_value = backtest_with_leverage(
    MyStrategy,
    coin='BTC',
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1h',
    leverage=100,
    max_leverage=100,
    margin_mode='cross',
    maintenance_margin_rate=0.005,
    initial_margin_rate=0.01,
    position_pct=0.0025,
    init_cash=1000.0,
)
```

## bulk_backtest()

```python
from backtrader.utils.backtest import bulk_backtest

results = bulk_backtest(
    MyStrategy,
    coins=['BTC', 'ETH', 'SOL'],    # None = all coins from DB
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1h',
    collateral='USDT',
    init_cash=10000,
    max_workers=8,
    save_results=True,
    output_file='bulk_results.csv',
    commission=0.00075,
    params={'take_profit': 2.0},
)
```

## Optimization

BTQuant uses Optuna for hyperparameter optimization.

```python
from backtrader.utils.optimize import optimize, OptimizationConfig

config = OptimizationConfig(
    strategy_class=MyStrategy,
    coin='BTC',
    interval='1h',
    start_date='2024-01-01',
    end_date='2024-12-31',
    collateral='USDT',
    init_cash=10000,
    commission=0.00075,
    n_trials=200,
    n_jobs=8,
    study_name='MyStrategy_BTC_1h',
    pruner='hyperband',
    seed=42,
    min_trades=30,
    plot_best=True,
    quantstats_best=True,
)

study = optimize(config, param_space_fn=my_param_space_fn)
```

The strategy module can define `param_space_aggressive()` and
`param_space_conservative()` functions that Optuna calls to suggest trial params.

## Analyzers Included

| Analyzer | Name | Output |
|---|---|---|
| `TimeReturn` | `time_return` | Per-bar returns for QuantStats |
| `SharpeRatio` | `sharpe_ratio` | Annualized Sharpe |
| `DrawDown` | `drawdown` | Max/current drawdown |
| `TradeAnalyzer` | `trade_analyzer` | Win/loss counts, P&L |
| `Returns` | `returns` | Total returns |
| `CustomSQN` | `customsqn` | System Quality Number |
| `PyFolio` | `pyfolio` | Full tear sheet (when not MTF) |

## QuantStats Reports

When `--quantstats` or `quantstats=True`:

- Generates `QuantStats/<coin>_<date>_<time>.html`
- Uses `quantstats_lumi` library
- Requires `TimeReturn` analyzer data

## Data Caching

- Cache dir: `.btq_cache/` (override with `BTQ_CACHE_DIR` env var)
- Format: Parquet with zstd compression
- Key: `md5(symbol|interval|collateral|ranges)[:12].parquet`
- CLI: `--no-cache` skips cache, `--clear-cache` deletes all cached data

## Configuration (Cerebro)

The default Cerebro is created with:

```python
cerebro = bt.Cerebro(
    oldbuysell=True,     # buy/sell arrows on chart
    runonce=False,       # step-by-step mode
    stdstats=False,      # custom observers added manually
)
```

Cerebro params (from `cerebro.py`):

| Param | Default | Description |
|---|---|---|
| `preload` | True | Preload data feeds |
| `runonce` | True | Vectorized indicator calc |
| `maxcpus` | None | CPU cores for optimization |
| `stdstats` | True | Default observers |
| `exactbars` | False | Memory-saving mode |
| `live` | False | Force live mode |
| `cheat_on_open` | False | Call next_open before orders |
| `tz` | None | Global timezone |

## Best Practices

1. **Always set start_date and end_date** -- unbounded queries are slow.
2. **Use caching** -- subsequent runs load from Parquet in seconds.
3. **Set min_trades >= 30** for optimization to avoid overfitting.
4. **Use --conservative mode** for strategies you plan to trade live.
5. **Enable --quantstats** for full performance tear sheets.
6. **Profile with --debug** to see indicator warmup and position state.
7. **Use add_mtf_resamples=True** for multi-timeframe strategies.
