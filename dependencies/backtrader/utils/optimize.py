import gc
import math
import time
import urllib.parse
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

import backtrader as bt
import optuna
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from time import sleep

from .backtest import (
    backtest,
    PolarsDataLoader,
    DataSpec,
    CACHE_DIR,
)

from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC

console = Console()

# --- Configuration ---
INIT_CASH = 1000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"

# --------------- Storage helpers ---------------
def mssql_url_from_odbc(odbc_connection_string: str, database_override: Optional[str] = None) -> str:
    parts = {}
    for chunk in odbc_connection_string.split(";"):
        if "=" in chunk and chunk.strip():
            k, v = chunk.split("=", 1)
            parts[k.strip().upper()] = v.strip()

    server = parts.get("SERVER", "localhost")
    database = database_override or parts.get("DATABASE", "OptunaBT")
    uid = parts.get("UID", "SA")
    pwd = parts.get("PWD", "")
    driver_raw = parts.get("DRIVER", "{ODBC Driver 18 for SQL Server}").strip("{}")
    driver = urllib.parse.quote_plus(driver_raw)

    extra = {
        "driver": driver,
        "Encrypt": "yes",
        "TrustServerCertificate": parts.get("TRUSTSERVERCERTIFICATE", "yes"),
    }
    query = "&".join(f"{k}={v}" for k, v in extra.items() if v is not None)
    encoded_pwd = urllib.parse.quote_plus(pwd)

    return f"mssql+pyodbc://{uid}:{encoded_pwd}@{server}/{database}?{query}"

def build_optuna_storage(storage_string: Optional[str]) -> Optional[optuna.storages.RDBStorage]:
    if storage_string is None:
        console.print("[yellow]Using in-memory Optuna storage[/yellow]")
        return None

    if storage_string.lower().startswith("mssql+pyodbc://"):
        storage_url = storage_string
    else:
        storage_url = mssql_url_from_odbc(storage_string)

    console.print(f"Optuna storage URL: {storage_url.split('@')[0]}@***")
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "pool_pre_ping": True,
            "pool_recycle": 300,
            "connect_args": {"timeout": 30},
        },
    )
    console.print("✓ RDBStorage ready")
    return storage

def ensure_mssql_storage(storage_string: str, study_name: str) -> optuna.storages.RDBStorage:
    """
    Create MSSQL storage for Optuna - NO SQLITE FALLBACK FOR MULTI-WORKER!
    """
    if not storage_string:
        raise ValueError("❌ MSSQL connection string is required for multi-worker optimization! No SQLite garbage allowed.")

    if not storage_string.lower().startswith("mssql+pyodbc://"):
        storage_url = mssql_url_from_odbc(storage_string)
    else:
        storage_url = storage_string

    try:
        console.print(f"[cyan]Optuna storage URL: {storage_url.split('@')[0]}@***[/cyan]")
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={
                "pool_pre_ping": True,
                "pool_recycle": 300,
                "pool_size": 20,  # Increased for multi-worker
                "max_overflow": 40,  # Handle concurrent connections
                "connect_args": {
                    "timeout": 30,
                    "connect_timeout": 30,
                }
            },
            skip_compatibility_check=False,
        )
        console.print(f"[green]✅ MSSQL storage initialized successfully[/green]")
        return storage
    except Exception as e:
        console.print(f"[red]❌ MSSQL storage init failed: {e}[/red]")
        raise RuntimeError(f"Failed to connect to MSSQL. Fix your connection string! Error: {e}")

@dataclass
class OptimizationConfig:
    """Configuration for optimization run"""
    strategy_class: type
    coin: str
    interval: str
    start_date: str
    end_date: str
    collateral: str = DEFAULT_COLLATERAL
    init_cash: float = INIT_CASH
    commission: float = COMMISSION_PER_TRANSACTION
    exchange: Optional[str] = None
    
    # Optuna settings
    n_trials: int = 100
    n_jobs: int = 1
    study_name: Optional[str] = None
    storage_string: Optional[str] = None  # MSSQL connection string
    pruner: Optional[str] = "hyperband"
    seed: Optional[int] = 42
    
    # Backtest settings
    plot_best: bool = True
    quantstats_best: bool = False
    min_trades: int = 10
    
    # Parameter space (optional)
    param_space: Optional[Dict[str, Any]] = None

def default_param_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Default parameter search space"""
    return {
        "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.0001, 0.005, log=True),
        "atr_stop_mult": trial.suggest_float("atr_stop_mult", 1.5, 4.0, step=0.25),
        "tp_r_multiple": trial.suggest_float("tp_r_multiple", 1.5, 5.0, step=0.5),
        "confirm_bars": trial.suggest_int("confirm_bars", 1, 3),
        "adxth": trial.suggest_int("adxth", 15, 30, step=5),
        "use_partial_exits": trial.suggest_categorical("use_partial_exits", [True, False]),
        "partial_exit_r": trial.suggest_float("partial_exit_r", 0.8, 2.5, step=0.1),
        "partial_exit_pct": trial.suggest_float("partial_exit_pct", 0.3, 0.7, step=0.1),
        "time_limit_bars": trial.suggest_int("time_limit_bars", 60, 360, step=30),
        "trail_mode": trial.suggest_categorical("trail_mode", ["chandelier", "donchian"]),
        "trail_atr_mult": trial.suggest_float("trail_atr_mult", 2.5, 5.0, step=0.25),
        "move_to_breakeven_R": trial.suggest_float("move_to_breakeven_R", 0.5, 1.5, step=0.25),
    }

def make_objective(
    config: OptimizationConfig,
    param_space_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None
) -> Callable:
    """Create Optuna objective function"""
    
    if param_space_fn is None:
        param_space_fn = default_param_space
    
    # Pre-cache data
    console.print(f"[cyan]Pre-loading data for {config.coin}...[/cyan]")
    loader = PolarsDataLoader()
    spec = DataSpec(
        symbol=config.coin,
        interval=config.interval,
        start_date=config.start_date,
        end_date=config.end_date,
        collateral=config.collateral
    )
    
    try:
        df = loader.load_data(spec, use_cache=True)
        console.print(f"[green]✅ Data cached for {config.coin}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        raise
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for a single trial"""
        
        # Get parameter suggestions
        params = param_space_fn(trial)
        
        # Extract strategy's valid parameters
        try:
            valid_keys = set(config.strategy_class.params._getkeys())
        except:
            valid_keys = set(k for k, _ in config.strategy_class.params)
        
        # Filter to only valid parameters
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}
        
        # Run backtest
        try:
            cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=-1)
            
            # Create fresh data feed for this trial
            feed = loader.make_backtrader_feed(df, spec)
            cerebro.adddata(feed)
            
            # Add strategy with parameters
            strat_kwargs = {
                'backtest': True,
                'optuna': True,  # 🔥 SET OPTUNA FLAG
                'can_short': (str(config.exchange).lower() == "mexc") if config.exchange else False,
            }
            strat_kwargs.update(filtered_params)
            
            cerebro.addstrategy(config.strategy_class, **strat_kwargs)
            
            # Setup broker
            cerebro.broker.setcash(config.init_cash)
            cerebro.broker.setcommission(commission=config.commission)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            
            # Run
            results = cerebro.run(maxcpus=1)
            strat = results[0]
            
            # Extract metrics
            sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
            sharpe = float(sr) if sr is not None and not math.isnan(sr) else 0.0
            
            mdd = float(strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0))
            
            ta = strat.analyzers.trades.get_analysis()
            trades = ta.get("total", {}).get("total", 0)
            wins = ta.get("won", {}).get("total", 0)
            
            # Store metrics
            trial.set_user_attr("sharpe", sharpe)
            trial.set_user_attr("max_dd", mdd)
            trial.set_user_attr("trades", trades)
            trial.set_user_attr("wins", wins)
            trial.set_user_attr("win_rate", (wins/trades*100 if trades > 0 else 0))
            trial.set_user_attr("final_value", cerebro.broker.getvalue())
            
            # Clean up
            del cerebro, feed, strat, results
            gc.collect()
            
            # Score
            if trades < config.min_trades:
                return -999.0
            
            score = sharpe - 0.03 * mdd
            return score
            
        except Exception as e:
            console.print(f"[red]Trial {trial.number} failed: {e}[/red]")
            console.print(traceback.format_exc())
            return -999.0
    
    return objective

def optimize(config: OptimizationConfig, param_space_fn: Optional[Callable] = None) -> optuna.Study:
    """Run Optuna optimization with MSSQL storage"""
    
    # Use MSSQL_ODBC global if not provided
    if not config.storage_string:
        config.storage_string = MSSQL_ODBC
    
    # Generate study name if not provided
    if config.study_name is None:
        config.study_name = f"{config.strategy_class.__name__}_{config.coin}_{config.interval}"
    
    console.print(f"[bold blue]🚀 Starting Optimization: {config.study_name}[/bold blue]")
    console.print(f"Coin: {config.coin}, Interval: {config.interval}")
    console.print(f"Period: {config.start_date} to {config.end_date}")
    console.print(f"Trials: {config.n_trials}, Workers: {config.n_jobs}")
    
    # Setup MSSQL storage (NO SQLITE)
    storage = ensure_mssql_storage(config.storage_string, config.study_name)
    
    # Create sampler
    sampler = optuna.samplers.TPESampler(
        seed=config.seed,
        multivariate=True,
        group=True,
        constant_liar=True if config.n_jobs > 1 else False,
        warn_independent_sampling=False
    )
    
    # Create pruner
    if config.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=config.n_trials,
            reduction_factor=3
        )
    elif config.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    else:
        pruner = None
    
    # Create or load study
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=config.study_name,
        load_if_exists=True
    )
    
    # Create objective
    objective = make_objective(config, param_space_fn)
    
    # Optimize
    t0 = time.time()
    study.optimize(
        objective,
        n_trials=config.n_trials,
        n_jobs=config.n_jobs,
        gc_after_trial=True,
        show_progress_bar=True
    )
    elapsed = time.time() - t0
    
    # Print results
    console.print(f"\n[green]✅ Optimization complete in {elapsed:.1f}s[/green]")
    console.print(f"Best value: {study.best_value:.4f}")
    
    # Print best parameters
    table = Table(title="🏆 Best Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for k, v in study.best_params.items():
        table.add_row(k, str(v))
    
    console.print(table)
    
    # Print best trial metrics
    best_trial = study.best_trial
    metrics_table = Table(title="📊 Best Trial Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    for k, v in best_trial.user_attrs.items():
        if isinstance(v, float):
            metrics_table.add_row(k, f"{v:.2f}")
        else:
            metrics_table.add_row(k, str(v))
    
    console.print(metrics_table)
    
    # Run final backtest with best parameters
    if config.plot_best or config.quantstats_best:
        console.print("\n[magenta]Running final backtest with best parameters...[/magenta]")
        backtest(
            config.strategy_class,
            coin=config.coin,
            start_date=config.start_date,
            end_date=config.end_date,
            interval=config.interval,
            collateral=config.collateral,
            init_cash=config.init_cash,
            commission=config.commission,
            exchange=config.exchange,
            plot=config.plot_best,
            quantstats=config.quantstats_best,
            params=study.best_params,
        )
    
    return study

def ensure_storage_or_sqlite(storage_string: Optional[str], study_name: str) -> optuna.storages.RDBStorage:
    """Ensure a valid Optuna storage connection (MSSQL or SQLite fallback)."""
    if storage_string is None:
        sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
        console.print(f"[yellow]No storage provided → using SQLite at {sqlite_path}[/yellow]")
        return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")

    if not storage_string.lower().startswith("mssql+pyodbc://"):
        storage_url = mssql_url_from_odbc(storage_string)
    else:
        storage_url = storage_string

    try:
        console.print(f"[blue]Connecting Optuna storage →[/blue] {storage_url.split('@')[0]}@***")
        return optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300, "connect_args": {"timeout": 30}},
        )
    except Exception as e:
        console.print(f"[red]MSSQL storage init failed: {e}[/red]")
        sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
        console.print(f"[red]Fallback → SQLite at {sqlite_path} have fun with this fuckery, idiot. NO SUPPORT GIVEN ON THIS SQLITE DEADEND - YOU ARE ON YOUR OWN HERE.[/red]")
        sleep(3)
        console.clear()
        console.print("\n")
        sleep(0.5)
        warning_header = """
[bold red blink]⚠️ WARNING ⚠️[/bold red blink]
[bold red blink]You are now entering unsupported territory![/bold red blink]

⚠️  SQLite storage detected.  ⚠️
⚠️  Parallel execution WILL break, hang, or corrupt your study.  ⚠️
⚠️  This is a fallback for emergencies only — not production use.  ⚠️

[bold red blink]NO SUPPORT GIVEN.[/bold red blink]
[bold]May the RNG gods have mercy with you.[/bold]
"""
        warning_panel = Panel(
            warning_header,
            border_style="bold red",
            title="[bold yellow]⚡ SYSTEM ALERT ⚡[/bold yellow]",
            title_align="center"
        )
        console.print(warning_panel)
        sleep(5)
        return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")

def get_param_names(cls) -> set:
    """Extract parameter names from a Backtrader strategy class."""
    try:
        return set(cls.params._getkeys())  # backtrader 1.9+
    except Exception:
        try:
            return set(k for k, _ in cls.params)  # legacy tuple-based
        except Exception:
            return set()  # fallback
