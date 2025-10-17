import backtrader as bt
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import gc
import pandas as pd
import polars as pl
import hashlib
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

INIT_CASH = 100_000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"

# Cache directory
CACHE_DIR = Path(os.getenv("BTQ_CACHE_DIR", ".btq_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

@dataclass(frozen=True)
class DataSpec:
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL

class PolarsDataLoader:
    """Polars-based data loader with caching capabilities"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_key(self, spec: DataSpec) -> str:
        ranges = spec.ranges or [(spec.start_date or "", spec.end_date or "")]
        ranges_str = ",".join([f"{s}:{e}" for s, e in ranges])
        raw = f"{spec.symbol}|{spec.interval}|{spec.collateral}|{ranges_str}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{spec.symbol}_{spec.interval}_{spec.collateral}_{h}.parquet"
    
    def _cache_path(self, spec: DataSpec) -> Path:
        return self.cache_dir / self._cache_key(spec)
    
    def _load_from_cache(self, spec: DataSpec) -> Optional[pl.DataFrame]:
        cache_path = self._cache_path(spec)
        if not cache_path.exists():
            return None
        try:
            return pl.scan_parquet(str(cache_path)).collect()
        except Exception as e:
            console.print(f"[yellow]Cache read failed for {spec.symbol}: {e}[/yellow]")
            return None
    
    def _save_to_cache(self, spec: DataSpec, df: pl.DataFrame) -> None:
        cache_path = self._cache_path(spec)
        try:
            df.write_parquet(str(cache_path), compression="zstd", statistics=True)
            console.print(f"[green]Cached {spec.symbol} -> {cache_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Cache write failed for {spec.symbol}: {e}[/yellow]")
    
    def _fetch_from_database(self, spec: DataSpec) -> pl.DataFrame:
        """Fetch data from database using existing MSSQL function"""
        from backtrader.feeds.mssql_crypto import get_database_data
        
        if spec.ranges:
            parts = []
            for start_date, end_date in spec.ranges:
                part = get_database_data(
                    ticker=spec.symbol, 
                    start_date=start_date, 
                    end_date=end_date,
                    time_resolution=spec.interval, 
                    pair=spec.collateral
                )
                if part is None or part.is_empty():
                    raise ValueError(f"No data for {spec.symbol} {spec.interval} {start_date}->{end_date}")
                parts.append(part)
            df = pl.concat(parts).sort("TimestampStart")
        else:
            df = get_database_data(
                ticker=spec.symbol, 
                start_date=spec.start_date, 
                end_date=spec.end_date,
                time_resolution=spec.interval, 
                pair=spec.collateral
            )
        
        if df is None or df.is_empty():
            raise ValueError(f"No data for {spec.symbol} {spec.interval} {spec.start_date}->{spec.end_date}")
        
        return df.sort("TimestampStart")
    
    def load_data(self, spec: DataSpec, use_cache: bool = True) -> pl.DataFrame:
        """Load data with caching support"""
        # Try cache first
        df = None
        if use_cache:
            df = self._load_from_cache(spec)
            if df is not None:
                console.print(f"[cyan]Loaded from cache: {spec.symbol}[/cyan]")
        
        # Fetch from database if not in cache
        if df is None:
            df = self._fetch_from_database(spec)
            if use_cache:
                self._save_to_cache(spec, df)
        
        # Validate required columns
        required = {"TimestampStart", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"{spec.symbol}: missing required columns. Found: {df.columns}")
        
        return df
    
    def make_backtrader_feed(self, df: pl.DataFrame, spec: DataSpec):
        """Convert Polars DataFrame to Backtrader feed"""
        from backtrader.feeds.mssql_crypto import MSSQLData
        
        feed = MSSQLData(
            dataname=df,
            datetime="TimestampStart", 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close", 
            volume="Volume",
            timeframe=bt.TimeFrame.Minutes, 
            compression=1
        )
        
        # Set timeframe attributes safely
        try:
            feed.p.timeframe = bt.TimeFrame.Minutes
            feed.p.compression = 1
        except Exception:
            setattr(feed, "_timeframe", bt.TimeFrame.Minutes)
            setattr(feed, "_compression", 1)
        
        # Set feed identifiers
        try:
            feed._name = f"{spec.symbol}-{spec.interval}"
            feed._dataname = f"{spec.symbol}{spec.collateral}"
        except Exception:
            pass
        
        return feed

def fetch_single_data(coin, start_date, end_date, interval, collateral="USDT"):
    """Legacy compatibility function - now uses PolarsDataLoader"""
    try:
        loader = PolarsDataLoader()
        spec = DataSpec(
            symbol=coin,
            interval=interval, 
            start_date=start_date, 
            end_date=end_date,
            collateral=collateral
        )
        df = loader.load_data(spec)
        feed = loader.make_backtrader_feed(df, spec)
        feed._dataname = f"{coin}{collateral}"
        return feed
    except Exception as e:
        console.print(f"[red]Error fetching data for {coin}: {str(e)}[/red]")
        return None

def backtest(
    strategy,
    data=None,
    coin=None,
    start_date=None,
    end_date="2030-12-31",
    interval=None,
    collateral="USDT",
    commission=COMMISSION_PER_TRANSACTION,
    init_cash=INIT_CASH,
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
    params=None,
    params_mode="mtf",
    add_mtf_resamples=False,
    **kwargs,
):
    from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
    from backtrader.strategies.base import CustomSQN, CustomData

    print("strategy:", strategy)

    # ---------- Data ----------
    if data is None:
        if not all([coin, start_date, end_date, interval]):
            raise ValueError("If data is not provided, coin, start_date, end_date, and interval are required")
        if show_progress and not bulk:
            console.print(f"🔄 [bold blue]Fetching data for {coin}...[/bold blue]")
        
        # Use OOP data loader
        loader = PolarsDataLoader()
        spec = DataSpec(
            symbol=coin,
            interval=interval, 
            start_date=start_date, 
            end_date=end_date,
            collateral=collateral
        )
        df = loader.load_data(spec, use_cache=True)
        data_feed = loader.make_backtrader_feed(df, spec)
        
        if show_progress and not bulk:
            console.print(f"✅ [bold green]Data fetched for {coin}[/bold green]")
    else:
        # Handle existing data input
        from backtrader.feed import DataBase
        if isinstance(data, DataBase):
            data_feed = data
        else:
            data_feed = CustomData(dataname=data)

    # ---------- Engine & feeds ----------
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
    cerebro.adddata(data_feed)

    # Add multi-timeframe resamples
    if add_mtf_resamples:
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=5,  name='5m',  boundoff=1)
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=15, name='15m', boundoff=1)
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=60, name='60m', boundoff=1)

    # ---------- Strategy parameters ----------
    strat_kwargs = {'backtest': True}

    def _strategy_param_keys(cls):
        try:
            return set(cls.params._getkeys())
        except Exception:
            try:
                return set(k for k, _ in cls.params)
            except Exception:
                return set()

    valid_keys = _strategy_param_keys(strategy)

    # Exchange-specific settings
    if 'can_short' in valid_keys and exchange is not None:
        strat_kwargs['can_short'] = (str(exchange).lower() == "mexc")
    
    # Order sizing parameters
    if 'min_qty' in valid_keys:
        strat_kwargs['min_qty'] = min_qty
    if 'qty_step' in valid_keys:
        strat_kwargs['qty_step'] = qty_step
    if 'price_tick' in valid_keys:
        strat_kwargs['price_tick'] = price_tick

    # Handle Optuna parameters
    if isinstance(params, dict):
        # For now, assume MTF format - you can add compat conversion later if needed
        converted_params = dict(params)
        
        # Merge safely
        reserved = set(strat_kwargs.keys())
        for k, v in converted_params.items():
            if k in valid_keys and k not in reserved:
                strat_kwargs[k] = v

    # Allow explicit kwargs
    for k, v in kwargs.items():
        if k not in strat_kwargs:
            strat_kwargs[k] = v

    if 'backtest' not in valid_keys and 'backtest' in strat_kwargs:
        del strat_kwargs['backtest']

    cerebro.addstrategy(strategy, **strat_kwargs)

    # ---------- Broker & analyzers ----------
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    try:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception:
        pass

    # Analyzers
    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(CustomSQN, _name='customsqn')
    
    if not add_mtf_resamples:
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)

    display_name = asset_name or (f"{coin}/{collateral}" if coin else "Asset")

    if show_progress and not bulk:
        console.print(f"🔄 [bold green]Running backtest for {display_name}...")

    strategy_result = cerebro.run()[0]

    # ---------- Metrics ----------
    dd_info = strategy_result.analyzers.drawdown.get_analysis()
    max_drawdown = dd_info.get('max', {}).get('drawdown', 0.0)
    trade_analyzer = strategy_result.analyzers.trade_analyzer.get_analysis()

    # ---------- Returns for QuantStats using Polars ----------
    tr = strategy_result.analyzers.time_return.get_analysis()

    if tr:
        # Create Polars DataFrame from time returns
        dates = list(tr.keys())
        returns_values = list(tr.values())
        
        returns_df = pl.DataFrame({
            'date': dates,
            'returns': returns_values
        })
        
        # Check if dates are already datetime objects or need parsing
        if returns_df.dtypes[0] == pl.String:
            # Parse string dates
            returns_df = returns_df.with_columns([
                pl.col('date').str.strptime(pl.Datetime, format='%Y-%m-%d', strict=False)
            ]).sort('date').drop_nulls()
        elif returns_df.dtypes[0] in [pl.Datetime, pl.Date]:
            # Already datetime, just sort and clean
            returns_df = returns_df.sort('date').drop_nulls()
        else:
            # Try to convert whatever type it is to datetime
            returns_df = returns_df.with_columns([
                pl.col('date').cast(pl.Datetime)
            ]).sort('date').drop_nulls()
        
        # Convert to pandas Series for QuantStats
        if not returns_df.is_empty():
            returns_pd = returns_df.to_pandas().set_index('date')['returns']
        else:
            returns_pd = pd.Series(dtype=float)
    else:
        returns_pd = pd.Series(dtype=float)

    if quantstats and not returns_pd.empty:
        import quantstats_lumi as quantstats
        from datetime import datetime
        import os
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")
        if asset_name:
            coin_name = asset_name.replace('/', '_')
        elif coin:
            coin_name = f"{coin}_{collateral}"
        elif hasattr(data_feed, '_dataname'):
            coin_name = str(data_feed._dataname)
        else:
            coin_name = "Unknown_Asset"
        folder = os.path.join("QuantStats")
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{coin_name}_{current_date}_{current_time}.html")
        try:
            quantstats.reports.html(returns_pd, output=filename, title=f'QuantStats_{coin_name}_{current_date}')
        except Exception as e:
            console.print(f"[yellow]QuantStats report generation failed: {e}[/yellow]")

    if not bulk:
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        pnl_net = trade_analyzer.get('pnl', {}).get('net', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - init_cash

        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS - {display_name}")
        print(f"{'='*50}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {won_trades}")
        print(f"Losing Trades: {lost_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Net P&L: ${pnl_net:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Final Portfolio Value: ${portvalue:.2f}")
        print(f"Total P/L: ${pnl:.2f}")
        print(f"Return: {(pnl / init_cash * 100):.2f}%")
        print(f"{'='*50}")

    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')

    # Store final value before cleanup
    final_value = cerebro.broker.getvalue()

    # Clean up
    del cerebro, data_feed, strategy_result
    gc.collect()

    return final_value

def backtest_with_leverage(
    strategy,
    data=None,
    coin=None,
    start_date=None,
    end_date="2030-12-31",
    interval=None,
    collateral="USDT",
    commission=0.00075,
    init_cash=100000.0,
    leverage=1,                    # NEW: Leverage parameter
    max_leverage=100,              # NEW: Max leverage limit
    margin_mode='isolated',        # NEW: Margin mode
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
    params=None,
    params_mode="mtf",
    add_mtf_resamples=False,
    **kwargs,
):
    """
    Enhanced backtest function with leverage support

    New Parameters:
        leverage: Leverage multiplier (default: 1 = no leverage)
        max_leverage: Maximum allowed leverage (default: 100)
        margin_mode: 'isolated' or 'cross' (default: 'isolated')

    Example:
        backtest_with_leverage(
            MyStrategy,
            coin='BTC',
            start_date='2025-01-01',
            end_date='2025-08-28',
            interval='1m',
            leverage=10,  # 10x leverage
            max_leverage=20
        )
    """
    from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
    from backtrader.strategies.base import CustomSQN, CustomData
    from rich.console import Console

    console = Console()

    # [Keep all your existing data loading logic]
    if data is None:
        if not all([coin, start_date, end_date, interval]):
            raise ValueError("If data is not provided, coin, start_date, end_date, and interval are required")

        from backtrader.feeds.mssql_crypto import get_database_data

        if show_progress and not bulk:
            console.print(f"🔄 [bold blue]Fetching data for {coin}...[/bold blue]")

        df = get_database_data(
            ticker=coin,
            start_date=start_date,
            end_date=end_date,
            time_resolution=interval,
            pair=collateral
        )

        data_feed = CustomData(dataname=df)

        if show_progress and not bulk:
            console.print(f"✅ [bold green]Data fetched for {coin}[/bold green]")
    else:
        from backtrader.feed import DataBase
        if isinstance(data, DataBase):
            data_feed = data
        else:
            data_feed = CustomData(dataname=data)

    # === LEVERAGE INTEGRATION POINT ===
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
    cerebro.adddata(data_feed)

    # Replace broker with LeverageBroker if leverage > 1
    if leverage > 1:
        from backtrader.Leverage import LeverageBroker
        broker = LeverageBroker(
            leverage=leverage,
            max_leverage=max_leverage,
            margin_mode=margin_mode,
        )
        cerebro.broker.setcash(init_cash)
        broker.setcommission(commission=commission)

        # Set slippage if supported
        try:
            broker.set_slippage_perc(perc=slippage_bps / 10000.0)
        except Exception:
            pass

        cerebro.broker = broker

        if show_progress and not bulk:
            console.print(f"💪 [bold yellow]Leverage enabled: {leverage}x ({margin_mode})[/bold yellow]")
    else:
        # Use standard broker
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=commission)
        try:
            cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
        except Exception:
            pass

    # [Rest of your existing backtest logic]
    # Add multi-timeframe resamples
    if add_mtf_resamples:
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=5, name='5m', boundoff=1)
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=15, name='15m', boundoff=1)
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=60, name='60m', boundoff=1)

    # Strategy parameters
    strat_kwargs = {'backtest': True}

    def _strategy_param_keys(cls):
        try:
            return set(cls.params._getkeys())
        except Exception:
            try:
                return set(k for k, _ in cls.params)
            except Exception:
                return set()

    valid_keys = _strategy_param_keys(strategy)

    # Exchange-specific settings
    if 'can_short' in valid_keys and exchange is not None:
        strat_kwargs['can_short'] = (str(exchange).lower() == "mexc")

    # Order sizing parameters
    if 'min_qty' in valid_keys:
        strat_kwargs['min_qty'] = min_qty
    if 'qty_step' in valid_keys:
        strat_kwargs['qty_step'] = qty_step
    if 'price_tick' in valid_keys:
        strat_kwargs['price_tick'] = price_tick

    # Handle Optuna parameters
    if isinstance(params, dict):
        converted_params = dict(params)
        reserved = set(strat_kwargs.keys())
        for k, v in converted_params.items():
            if k in valid_keys and k not in reserved:
                strat_kwargs[k] = v

    # Allow explicit kwargs
    for k, v in kwargs.items():
        if k not in strat_kwargs:
            strat_kwargs[k] = v

    cerebro.addstrategy(strategy, **strat_kwargs)

    # Analyzers
    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(CustomSQN, _name='customsqn')

    if not add_mtf_resamples:
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)

    display_name = asset_name or (f"{coin}/{collateral}" if coin else "Asset")

    if show_progress and not bulk:
        console.print(f"🔄 [bold green]Running backtest for {display_name}...")

    strategy_result = cerebro.run()[0]

    # === LEVERAGE STATISTICS ===
    leverage_stats = {}
    if hasattr(cerebro.broker, 'get_margin_info'):
        leverage_stats = cerebro.broker.get_margin_info()

    # Metrics
    dd_info = strategy_result.analyzers.drawdown.get_analysis()
    max_drawdown = dd_info.get('max', {}).get('drawdown', 0.0)
    trade_analyzer = strategy_result.analyzers.trade_analyzer.get_analysis()

    if not bulk:
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        pnl_net = trade_analyzer.get('pnl', {}).get('net', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - init_cash

        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS - {display_name}")
        print(f"{'='*50}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {won_trades}")
        print(f"Losing Trades: {lost_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Net P&L: ${pnl_net:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Final Portfolio Value: ${portvalue:.2f}")
        print(f"Total P/L: ${pnl:.2f}")
        print(f"Return: {(pnl / init_cash * 100):.2f}%")

        # === PRINT LEVERAGE STATS ===
        if leverage_stats:
            print(f"\n{'='*50}")
            print(f"LEVERAGE STATISTICS")
            print(f"{'='*50}")
            print(f"Leverage: {leverage_stats.get('leverage', 1)}x")
            print(f"Margin Mode: {leverage_stats.get('margin_mode', 'N/A')}")
            print(f"Liquidations: {leverage_stats.get('liquidation_count', 0)}")
            print(f"Margin Calls: {leverage_stats.get('margin_call_count', 0)}")
            print(f"Final Margin Used: ${leverage_stats.get('total_margin_used', 0):.2f}")

        print(f"{'='*50}")

    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')

    final_value = cerebro.broker.getvalue()

    # Clean up
    import gc
    del cerebro, data_feed, strategy_result
    gc.collect()

    return final_value

def run_multi_isolated_pairs(strategy, pairs_config, start_date, end_date, interval, collateral="USDT"):
    """
    Run multiple isolated leveraged backtests at once — each pair has its own leverage, margin, and isolation.
    """
    from rich.console import Console
    from rich.table import Table
    import concurrent.futures
    import gc

    console = Console()
    results = []

    def run_pair(pair_cfg):
        coin = pair_cfg["symbol"]
        lev = float(pair_cfg.get("leverage", 1))
        cap = float(pair_cfg.get("capital", 100))
        iso_margin = cap / lev

        console.print(f"[cyan]Running {coin} with x{lev} isolated (margin {iso_margin:.2f}$)[/cyan]")

        try:
            loader = PolarsDataLoader()
            spec = DataSpec(
                symbol=coin,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                collateral=collateral
            )
            df = loader.load_data(spec, use_cache=True)
            feed = loader.make_backtrader_feed(df, spec)

            # simulate isolated margin by setting broker cash = margin * leverage
            init_cash = iso_margin * lev

            final_value = backtest(
                strategy,
                data=feed,
                coin=coin,
                init_cash=init_cash,
                collateral=collateral,
                bulk=True,
                show_progress=True,
                quantstats=False,
            )

            pnl = final_value - init_cash
            ret_pct = (pnl / init_cash) * 100

            gc.collect()
            return {
                "coin": coin,
                "leverage": lev,
                "capital": cap,
                "iso_margin": iso_margin,
                "final_value": final_value,
                "pnl": pnl,
                "return_pct": ret_pct,
                "status": "success",
            }

        except Exception as e:
            return {"coin": coin, "error": str(e), "status": "failed"}

    # Parallel run all pairs
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs_config)) as executor: # ProcessPoolExecutor
        futures = [executor.submit(run_pair, p) for p in pairs_config]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Print results summary
    table = Table(title="📊 Multi-Isolated Backtest Results")
    table.add_column("Pair", style="cyan")
    table.add_column("Lev", justify="right")
    table.add_column("Margin", justify="right")
    table.add_column("Final Value", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("Return %", justify="right")

    for r in results:
        if r["status"] == "success":
            table.add_row(
                r["coin"],
                f"x{r['leverage']}",
                f"${r['iso_margin']:.2f}",
                f"${r['final_value']:.2f}",
                f"${r['pnl']:.2f}",
                f"{r['return_pct']:.2f}%",
            )
        else:
            table.add_row(r["coin"], "-", "-", "-", "-", f"❌ {r['error']}")

    console.print(table)
    return results

# Keep the rest of your functions unchanged...
def run_backtest_with_data(args):
    """Helper function for parallel execution in bulk_backtest"""
    coin, data, strategy_class, init_cash, backtest_params, collateral = args

    if data is None:
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": "No data available", "status": "skipped"}

    try:
        asset = f'{coin}/{collateral}'
        
        result = backtest(
            strategy_class,
            data=data,
            init_cash=init_cash,
            asset_name=asset,
            bulk=True,
            **backtest_params
        )

        del data
        gc.collect()

        return {
            "coin": coin,
            "asset": asset,
            "result": result,
            "final_value": result,
            "pnl": result - init_cash,
            "return_pct": ((result - init_cash) / init_cash) * 100,
            "status": "success"
        }

    except Exception as e:
        console.print(f"[red]Error in {coin}: {str(e)}[/red]")
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": str(e), "status": "failed"}

def bulk_backtest(strategy, coins=None, start_date="2016-01-01", end_date="2026-01-08", interval=None, 
                 collateral="USDT", init_cash=1000, max_workers=8, save_results=True, 
                 output_file='backtest_results.json', params_mode="mtf", **backtest_kwargs):
    """Enhanced bulk backtest with unified progress tracking and auto-discovery"""
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Validate required parameters first
    if not all([start_date, end_date, interval]):
        raise ValueError("start_date, end_date, and interval are required")

    # Auto-discover coins from database if not provided
    if coins is None:
        console.print("[bold blue]🔍 Auto-discovering coins from database...[/bold blue]")
        try:
            coins = get_available_coins_from_database(collateral=collateral)
            if not coins:
                console.print("[red]❌ No coins found in database[/red]")
                return []
            
            sample_coins = coins[:10]
            remaining = len(coins) - 10
            coins_display = ', '.join(sample_coins)
            if remaining > 0:
                coins_display += f" (and {remaining} more)"
            
            console.print(f"[green]✅ Found {len(coins)} coins: {coins_display}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to get coins from database: {e}[/red]")
            return []

    default_backtest_params = {
        'plot': False,
        'quantstats': False,
        'params_mode': params_mode,
        'add_mtf_resamples': False,
    }
    default_backtest_params.update(backtest_kwargs)
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    results = []
    successful_count = 0
    failed_count = 0

    with progress:
        backtest_task = progress.add_task("🚀 Running backtests...", total=len(coins))
        
        # arguments for parallel execution
        backtest_args = [
            (coin, start_date, end_date, interval, collateral, strategy, init_cash, default_backtest_params)
            for coin in coins
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_coin = {
                executor.submit(run_single_backtest_with_data_fetch, args): args[0] 
                for args in backtest_args
            }
            
            for future in concurrent.futures.as_completed(future_to_coin):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    coin = result.get('coin', 'Unknown')
                    completed = successful_count + failed_count
                    
                    progress.update(
                        backtest_task, 
                        completed=completed,
                        description=f"🚀 Backtests: {completed}/{len(coins)} (✅{successful_count} ❌{failed_count}) - Last: {coin}"
                    )
                    
                except Exception as exc:
                    coin = future_to_coin[future]
                    failed_count += 1
                    completed = successful_count + failed_count
                    
                    progress.console.print(f"[red]Exception for {coin}: {exc}[/red]")
                    results.append({"coin": coin, "asset": f"{coin}/{collateral}", "error": str(exc), "status": "failed"})
                    
                    progress.update(
                        backtest_task, 
                        completed=completed,
                        description=f"🚀 Backtests: {completed}/{len(coins)} (✅{successful_count} ❌{failed_count}) - Last: {coin} ❌"
                    )
        
        progress.update(backtest_task, completed=len(coins), description="🚀 All backtests complete")
    
    if save_results:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\n[green]Results saved to '{output_file}'[/green]")
    
    console.print(f"\n[bold green]🎉 Backtesting completed![/bold green]")
    return print_detailed_results(results)


def run_single_backtest_with_data_fetch(args):
    """Fetch data and run backtest for a single coin - designed for parallel execution"""
    coin, start_date, end_date, interval, collateral, strategy_class, init_cash, backtest_params = args
    
    try:
        # Step 1: Fetch data using existing caching system
        loader = PolarsDataLoader()
        spec = DataSpec(
            symbol=coin,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            collateral=collateral
        )
        
        # This will use cache if available, fetch from DB if not
        df = loader.load_data(spec, use_cache=True)
        
        if df is None or df.is_empty():
            return {"coin": coin, "asset": f"{coin}/{collateral}", "error": "No data available", "status": "skipped"}
        
        # Step 2: Create backtrader feed
        data_feed = loader.make_backtrader_feed(df, spec)
        data_feed._dataname = f"{coin}{collateral}"
        
        # Step 3: Run backtest
        asset = f'{coin}/{collateral}'
        
        result = backtest(
            strategy_class,
            data=data_feed,
            init_cash=init_cash,
            asset_name=asset,
            bulk=True,
            show_progress=False,  # No progress for individual backtests
            **backtest_params
        )
        
        # Clean up
        del df, data_feed
        gc.collect()
        
        return {
            "coin": coin,
            "asset": asset,
            "result": result,
            "final_value": result,
            "pnl": result - init_cash,
            "return_pct": ((result - init_cash) / init_cash) * 100,
            "status": "success"
        }
        
    except Exception as e:
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": str(e), "status": "failed"}


def get_available_coins_from_database(collateral="USDT"):
    """Get all available coins from the database"""
    try:
        from backtrader.feeds.mssql_crypto import MSSQLData
        from backtrader.dontcommit import connection_string
        
        all_pairs = MSSQLData.get_all_pairs(connection_string)
        
        coins = []
        suffix_pattern = f"{collateral}_klines"
        
        for pair in all_pairs:
            if pair.endswith(suffix_pattern):
                # Extract coin name by removing the suffix
                coin = pair.replace(suffix_pattern, "")
                # Remove any trailing underscores
                coin = coin.rstrip("_")
                if coin:
                    coins.append(coin)
        
        # Remove duplicates and sort
        coins = sorted(list(set(coins)))
        
        return coins
        
    except Exception as e:
        console.print(f"[red]❌ Error getting coins from database: {e}[/red]")
        # Fallback to common coins
        return ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOT', 'DOGE', 'AVAX', 'MATIC']

from rich.console import Console
from rich.table import Table

console = Console()

def print_detailed_results(results):
    """Print detailed backtest results in a styled Rich table format"""

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    console.rule("[bold yellow]🚀 BTQuant Results[/]")

    # ✅ Successful backtests
    if successful:
        success_table = Table(title=f"🟢 Successful Backtests ({len(successful)})")
        success_table.add_column("Asset", style="cyan")
        success_table.add_column("Final Value", style="green")
        success_table.add_column("P&L", style="magenta")

        successful_sorted = sorted(successful, key=lambda x: x.get("pnl", 0), reverse=True)
        for r in successful_sorted:
            asset = r.get("asset", r.get("coin", "N/A"))
            final_value = r.get("final_value")
            pnl = r.get("pnl")

            final_value_str = f"${final_value:,.2f}" if isinstance(final_value, (int, float)) else str(final_value)
            pnl_str = f"${pnl:,.2f}" if isinstance(pnl, (int, float)) else str(pnl)

            success_table.add_row(asset, final_value_str, pnl_str)

        console.print(success_table)

    # ❌ Failed backtests
    if failed:
        fail_table = Table(title=f"🔴 Failed Backtests ({len(failed)})")
        fail_table.add_column("Asset", style="red")
        fail_table.add_column("Error", style="yellow")

        for r in failed:
            fail_table.add_row(
                r.get("asset", r.get("coin", "N/A")),
                str(r.get("error", "Unknown error"))
            )

        console.print(fail_table)

    # ⚪ Skipped backtests
    if skipped:
        skip_table = Table(title=f"⚪ Skipped Backtests (No Data) ({len(skipped)})")
        skip_table.add_column("Assets", style="white")

        skipped_assets = [r.get("asset", r.get("coin", "N/A")) for r in skipped]
        skip_table.add_row(", ".join(skipped_assets))

        console.print(skip_table)

    # 📊 Summary
    summary = Table(title="📊 Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Successful", str(len(successful)))
    summary.add_row("Failed", str(len(failed)))
    summary.add_row("Skipped (No Data)", str(len(skipped)))

    console.print(summary)

    return results