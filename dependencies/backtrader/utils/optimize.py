import gc
import math
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
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
    """
    Create an Optuna objective function that:
      - runs walk-forward evaluation across multiple OOS slices with an embargo,
      - aggregates performance robustly across OOS slices,
      - gates selection by an approximate Deflated Sharpe Ratio significance check,
      - and records rich trial metrics for later analysis.
    """
    import numpy as np
    import math
    import traceback
    from datetime import timedelta

    if param_space_fn is None:
        param_space_fn = default_param_space

    # Pre-cache data once
    console.print(f"[cyan]Pre-loading data for {config.coin}...[/cyan]")
    loader = PolarsDataLoader()
    spec = DataSpec(
        symbol=config.coin,
        interval=config.interval,
        start_date=config.start_date,
        end_date=config.end_date,
        collateral=config.collateral,
    )
    try:
        df = loader.load_data(spec, use_cache=True)
        console.print(f"[green]✅ Data cached for {config.coin}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        raise

    def build_walkforward_slices(
        df_,
        is_days: int = 90,
        oos_days: int = 30,
        embargo_days: int = 3
    ):
        dts = df_["datetime"]
        if len(dts) == 0:
            return []
        start = dts.min()
        end = dts.max()
        slices = []
        cur_is_start = start
        delta_is = timedelta(days=is_days)
        delta_oos = timedelta(days=oos_days)
        delta_emb = timedelta(days=embargo_days)
        while True:
            is_start = cur_is_start
            is_end = is_start + delta_is
            emb_start = is_end + delta_emb
            oos_start = emb_start
            oos_end = oos_start + delta_oos
            if oos_end > end:
                break
            slices.append((is_start, is_end, oos_start, oos_end))
            cur_is_start = cur_is_start + delta_oos
        return slices

    wf_slices = build_walkforward_slices(df)

    if not wf_slices:
        dts = df["datetime"]
        start = dts.min()
        end = dts.max()
        total_days = max(1, (end - start).days)
        oos_days = max(1, int(0.25 * total_days))
        is_end = start + timedelta(days=(total_days - oos_days))
        oos_start = is_end + timedelta(days=2)
        wf_slices = [(start, is_end, oos_start, end)]

    def approx_deflated_sharpe_prob(sr_hat: float, rets: np.ndarray, n_trials: int) -> float:
        rets = np.asarray(rets, dtype=float)
        T = int(rets.size)
        if T < 10 or np.allclose(rets.std(ddof=1), 0.0):
            return 0.0  # not enough information to claim significance
        mu = rets.mean()
        sd = rets.std(ddof=1)
        # Central moments
        z = (rets - mu) / sd
        gamma3 = float(np.mean(z**3))
        gamma4 = float(np.mean(z**4))  # non-excess kurtosis (>= 1)
        # Selection-bias floor for SR given N trials
        N = max(2, int(n_trials))
        sr0 = math.sqrt(2.0 * math.log(N)) / math.sqrt(max(1, T - 1))
        # Adjusted z of Sharpe
        denom = math.sqrt(max(1e-12, 1.0 - gamma3 * sr_hat + ((gamma4 - 1.0) / 4.0) * (sr_hat ** 2)))
        z_sr = ((sr_hat - sr0) * math.sqrt(max(1, T - 1))) / denom
        # One-sided probability that SR > SR0 under adjusted normal
        # Phi(z)
        prob = 0.5 * (1.0 + math.erf(z_sr / math.sqrt(2.0)))
        return max(0.0, min(1.0, prob))

    def run_one_window(oos_df, filtered_params):
        # Build and run a Cerebro instance for the OOS slice only
        cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=-1)
        feed = loader.make_backtrader_feed(oos_df, spec)
        cerebro.adddata(feed)

        strat_kwargs = {
            'backtest': True,
            'optuna': True,
            'can_short': (str(config.exchange).lower() == "mexc") if config.exchange else False,
        }
        strat_kwargs.update(filtered_params)
        cerebro.addstrategy(config.strategy_class, **strat_kwargs)

        cerebro.broker.setcash(config.init_cash)
        cerebro.broker.setcommission(commission=config.commission)
        # Optional slippage realism (2.5 bps); adjust to venue conditions
        cerebro.broker.set_slippage_perc(0.00025)

        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.NoTimeFrame)

        results = cerebro.run(maxcpus=1)
        strat = results[0]

        sr_val = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr_val) if sr_val is not None and not math.isnan(sr_val) else 0.0
        dd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0) or 0.0

        ta = strat.analyzers.trades.get_analysis()
        trades = int(ta.get("total", {}).get("total", 0) or 0)
        wins = int(ta.get("won", {}).get("total", 0) or 0)

        # Extract per-bar returns for OOS
        tr_map = strat.analyzers.timereturn.get_analysis()
        # Backtrader returns an OrderedDict keyed by datetime -> return
        oos_returns = np.array(list(tr_map.values()), dtype=float) if tr_map else np.array([], dtype=float)

        # Cleanup
        del cerebro, feed, strat, results
        gc.collect()
        return sharpe, float(dd), trades, wins, oos_returns

    def objective(trial: optuna.Trial) -> float:
        # 1) Sample parameters
        params = param_space_fn(trial)

        # 2) Filter to valid strategy params
        try:
            valid_keys = set(config.strategy_class.params._getkeys())
        except Exception:
            valid_keys = set(k for k, _ in getattr(config.strategy_class, "params", []))
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}

        try:
            # 3) Walk-forward across OOS slices
            oos_sharpes = []
            oos_drawdowns = []
            total_trades = 0
            total_wins = 0
            all_oos_returns = []

            for (is_start, is_end, oos_start, oos_end) in wf_slices:
                # Embargo applied in slice construction; run OOS only
                oos_df = df.filter(
                    (df["datetime"] >= oos_start) & (df["datetime"] < oos_end)
                )
                if len(oos_df) == 0:
                    continue

                sr, mdd, tr, wn, rets = run_one_window(oos_df, filtered_params)
                oos_sharpes.append(sr)
                oos_drawdowns.append(mdd)
                total_trades += tr
                total_wins += wn
                if rets.size:
                    all_oos_returns.append(rets)

            # 4) Aggregate robustly across slices
            if total_trades < config.min_trades or len(oos_sharpes) == 0:
                return -999.0

            med_sr = float(np.median(oos_sharpes))
            mean_mdd = float(np.mean(oos_drawdowns)) if oos_drawdowns else 100.0
            concat_rets = np.concatenate(all_oos_returns) if all_oos_returns else np.array([], dtype=float)

            # 5) Complexity regularization: penalize enabling too many blocks
            enabled_blocks = sum([
                1 if filtered_params.get("use_cycle_signals", False) else 0,
                1 if filtered_params.get("use_regime_signals", False) else 0,
                1 if filtered_params.get("use_volatility_signals", False) else 0,
                1 if filtered_params.get("use_momentum_signals", False) else 0,
                1 if filtered_params.get("use_trend_signals", False) else 0,
            ])
            complexity_pen = 0.5 * max(0, enabled_blocks - 3)  # small L1-like bias toward simpler configs

            # 6) Approximate Deflated Sharpe probability (significance against multiple testing)
            n_trials_eff = max(2, trial.number + 1)
            dsr_prob = approx_deflated_sharpe_prob(med_sr, concat_rets, n_trials_eff)

            # 7) Final score: robust SR minus drawdown penalty and complexity;
            # demote if DSR probability is weak.
            score = med_sr * 100.0 - 3.0 * mean_mdd - complexity_pen
            if dsr_prob < 0.8:
                score -= 25.0
            if dsr_prob < 0.6:
                score -= 25.0  # total -50 if very weak

            # 8) Record trial-level metrics
            win_rate = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
            trial.set_user_attr("oos_median_sharpe", med_sr)
            trial.set_user_attr("oos_mean_max_dd", mean_mdd)
            trial.set_user_attr("oos_trades", int(total_trades))
            trial.set_user_attr("oos_wins", int(total_wins))
            trial.set_user_attr("oos_win_rate", float(win_rate))
            trial.set_user_attr("oos_slices", int(len(oos_sharpes)))
            trial.set_user_attr("enabled_blocks", int(enabled_blocks))
            trial.set_user_attr("complexity_pen", float(complexity_pen))
            trial.set_user_attr("dsr_prob", float(dsr_prob))

            return float(score)

        except Exception as e:
            console.print(f"[red]Trial {trial.number} failed: {e}[/red]")
            console.print(traceback.format_exc())
            return -999.0

    return objective


def optimize(config: OptimizationConfig, param_space_fn: Optional[Callable] = None) -> optuna.Study:
    """Run Optuna optimization with MSSQL storage"""
    
    # Validate MSSQL connection is provided for multi-worker
    if config.n_jobs > 1 and not config.storage_string:
        raise ValueError("❌ MSSQL storage_string is REQUIRED for multi-worker optimization (n_jobs > 1)!")
    
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

[red]NO SUPPORT GIVEN.[/red]
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
