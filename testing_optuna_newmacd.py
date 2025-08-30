# optuna_hyperopt_polars.py
import backtrader as bt
import os
import gc
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable

import optuna
import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()
from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData

# Strategy import (adjust if named differently)
from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX3 as StrategyClass

# Defaults
INIT_CASH = 1000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"

# --------------- Data spec ---------------
@dataclass(frozen=True)
class DataSpec:
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL
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

# --------------- Data preload + feed factory ---------------
def preload_polars(
    specs: List[DataSpec],
) -> Dict[str, pl.DataFrame]:
    """
    Load each asset once using your get_database_data (Polars), keep in memory.
    Keyed by symbol. Assumes same interval/date per symbol as in specs.
    """
    df_map: Dict[str, pl.DataFrame] = {}
    seen = set()

    for spec in specs:
        if spec.symbol in seen:
            continue

        if spec.ranges:
            dfs = []
            for s, e in spec.ranges:
                part = get_database_data(
                    ticker=spec.symbol,
                    start_date=s,
                    end_date=e,
                    time_resolution=spec.interval,
                    pair=spec.collateral,
                )
                if part is None or part.is_empty():
                    raise ValueError(f"No data for {spec.symbol} {spec.interval} {s}->{e}")
                dfs.append(part)
            df = pl.concat(dfs).sort("TimestampStart")
        else:
            df = get_database_data(
                ticker=spec.symbol,
                start_date=spec.start_date,
                end_date=spec.end_date,
                time_resolution=spec.interval,
                pair=spec.collateral,
            )
        if df is None or df.is_empty():
            raise ValueError(f"No data for {spec.symbol} {spec.interval} {spec.start_date}->{spec.end_date}")

        # Ensure expected columns exist (your loader provides these)
        required = {"TimestampStart", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"{spec.symbol}: missing required columns. Found: {df.columns}")

        # Sort by time
        df = df.sort("TimestampStart")

        df_map[spec.symbol] = df
        seen.add(spec.symbol)

    return df_map

def make_feed_from_df(df: pl.DataFrame, spec: DataSpec) -> MSSQLData:
    """
    Wrap a cloned Polars DataFrame into your PolarsData feed.
    Always pass explicit mapping, no guessing.
    """
    cloned = df.clone()  # safe, cheap
    feed = MSSQLData(
        dataname=cloned,
        datetime="TimestampStart",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    try:
        feed._name = f"{spec.symbol}-{spec.interval}"
        feed._dataname = f"{spec.symbol}{spec.collateral}"
    except Exception:
        pass
    return feed

# --------------- Search space ---------------
def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    p = dict(
        breakout_period=trial.suggest_int("breakout_period", 15, 80, step=5),
        adxth=trial.suggest_int("adxth", 18, 35),
        vol_mult=trial.suggest_float("vol_mult", 1.1, 2.2, step=0.1),
        stretch_atr_mult=trial.suggest_float("stretch_atr_mult", 0.7, 1.6, step=0.1),

        use_dca=trial.suggest_categorical("use_dca", [True, False]),
        max_adds=trial.suggest_int("max_adds", 1, 99),
        add_cooldown=trial.suggest_int("add_cooldown", 1, 100, step=5),  # bars between adds
        dca_atr_mult=trial.suggest_float("dca_atr_mult", 0.6, 1.4, step=0.1),
        add_on_ema_touch=trial.suggest_categorical("add_on_ema_touch", [True, False]),

        macd_period_me1=trial.suggest_int("macd_period_me1", 8, 14),
        macd_period_me2=trial.suggest_int("macd_period_me2", 18, 30),
        macd_period_signal=trial.suggest_int("macd_period_signal", 5, 10),
        adx_period=trial.suggest_int("adx_period", 10, 20),
        di_period=trial.suggest_int("di_period", 10, 20),
        momentum_period=trial.suggest_int("momentum_period", 10, 20),
        rsi_period=trial.suggest_int("rsi_period", 10, 21),
        stoch_period=trial.suggest_int("stoch_period", 10, 21),
        cci_period=trial.suggest_int("cci_period", 14, 28),
        trix_period=trial.suggest_int("trix_period", 12, 25),

        ema_fast=trial.suggest_int("ema_fast", 18, 30),
        ema_slow=trial.suggest_int("ema_slow", 40, 70),
        ema_trend=trial.suggest_int("ema_trend", 150, 260),
        atr_period=trial.suggest_int("atr_period", 10, 20),

        vol_window=trial.suggest_int("vol_window", 15, 30),
        take_profit=trial.suggest_float("take_profit", 0.5, 20.0, step=0.5),

        # use_trailing_stop=trial.suggest_categorical("use_trailing_stop", [True, False]),

        # ALWAYS SUGGEST THESE (no condition)
        trail_mode=trial.suggest_categorical("trail_mode", ["ema_band", "chandelier", "donchian", "pivot"]),
        trail_atr_mult=trial.suggest_float("trail_atr_mult", 2.5, 6.0, step=0.5),
        ema_band_mult=trial.suggest_float("ema_band_mult", 1.5, 3.0, step=0.25),
        donchian_trail_period=trial.suggest_int("donchian_trail_period", 35, 100, step=5),
        pivot_left=trial.suggest_int("pivot_left", 1, 3),
        pivot_right=trial.suggest_int("pivot_right", 1, 3),
        init_sl_atr_mult=trial.suggest_float("init_sl_atr_mult", 0.9, 1.7, step=0.1),
        trail_arm_R=trial.suggest_float("trail_arm_R", 1.0, 3.0, step=0.25),
        trail_arm_bars=trial.suggest_int("trail_arm_bars", 5, 100, step=1),
        trail_update_every=trial.suggest_int("trail_update_every", 1, 250, step=1), #2, 6),
    )
    return p

# --------------- Scoring ---------------
def score_from_analyzers(
    strat,
    min_trades: int = 8,
    weight_sharpe: float = 0.55,
    weight_calmar: float = 0.45,
) -> Tuple[float, Dict[str, float]]:
    try:
        draw = strat.analyzers.drawdown.get_analysis()
        mdd = float(draw.get("max", {}).get("drawdown", 0.0))
    except Exception:
        mdd = 0.0

    try:
        sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr) if sr is not None else 0.0
    except Exception:
        sharpe = 0.0
    sharpe = max(-2.0, min(5.0, sharpe))

    try:
        ret = strat.analyzers.returns.get_analysis()
        cagr_dec = float(ret.get("rnorm", 0.0))
    except Exception:
        cagr_dec = 0.0

    try:
        ta = strat.analyzers.trades.get_analysis()
        total_trades = ta.get("total", {}).get("total", 0) or 0
        won = ta.get("won", {}).get("total", 0) or 0
        win_rate = (won / total_trades) if total_trades > 0 else 0.0
    except Exception:
        total_trades, win_rate = 0, 0.0

    calmar = cagr_dec / (max(mdd, 1e-9) / 100.0) if mdd > 0 else cagr_dec

    score = weight_sharpe * sharpe + weight_calmar * calmar
    if total_trades < min_trades:
        score -= 0.5
    if mdd > 40:
        score -= (mdd - 40) / 40.0

    metrics = dict(mdd=mdd, sharpe=sharpe, calmar=calmar, cagr=cagr_dec, trades=total_trades, win_rate=win_rate)
    return score, metrics

# --------------- Single backtest ---------------
def run_single_backtest_eval(
    strategy_class,
    df: pl.DataFrame,
    spec: DataSpec,
    init_cash: float,
    commission: float,
    params: Dict[str, Any],
) -> Tuple[float, Dict[str, float], float]:
    cerebro = bt.Cerebro(oldbuysell=True)

    # Wrap DF into your PolarsData feed
    feed = make_feed_from_df(df, spec)
    cerebro.adddata(feed)

    cerebro.addstrategy(strategy_class, backtest=True, **params)

    # Analyzers (match your backtest style without pyfolio overhead)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    # Observers
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)

    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.setcash(init_cash)

    try:
        strats = cerebro.run()
        strat = strats[0]
    except Exception as e:
        console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]")
        return -999.0, {"error": 1.0}, 0.0

    score, metrics = score_from_analyzers(strat)
    final_value = cerebro.broker.getvalue()

    # Cleanup
    del cerebro, feed, strats, strat
    gc.collect()

    return score, metrics, final_value

# --------------- Objective ---------------
def make_objective(
    strategy_class,
    specs: List[DataSpec],
    df_map: Dict[str, pl.DataFrame],
    init_cash: float,
    commission: float,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        # Constraints to keep EMA stack sane
        if params["ema_slow"] - params["ema_fast"] < 10:
            trial.set_user_attr("constraint_fail", "ema_gap")
            return -999
        if params["ema_trend"] - params["ema_slow"] < 60:
            trial.set_user_attr("constraint_fail", "trend_gap")
            return -999

        total_score = 0.0
        wsum = 0.0
        step = 0
        weights = [1.0 + 0.1 * i for i in range(len(specs))]

        for idx, spec in enumerate(specs):
            df = df_map[spec.symbol]
            score, metrics, _ = run_single_backtest_eval(
                strategy_class=strategy_class,
                df=df,
                spec=spec,
                init_cash=init_cash,
                commission=commission,
                params=params,
            )

            w = weights[idx]
            total_score += score * w
            wsum += w
            step += 1

            trial.set_user_attr(f"{spec.symbol}_score", score)
            trial.set_user_attr(f"{spec.symbol}_mdd", metrics.get("mdd", 0.0))
            trial.set_user_attr(f"{spec.symbol}_trades", metrics.get("trades", 0))

            trial.report(total_score / max(wsum, 1e-9), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return total_score / max(wsum, 1e-9)

    return objective

# --------------- Optimize ---------------
def make_pruner(num_steps: int, pruner: Optional[str], n_jobs: int):
    """
    num_steps = len(specs) i.e., how many trial.report steps you produce.
    """
    if pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, num_steps),  # e.g. 5 assets -> 5 rungs
            reduction_factor=3,
            bootstrap_count=max(2 * n_jobs, 8),  # reduce bracket weirdness with parallel starts
        )
    elif pruner == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    elif pruner == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=1)
    else:
        return None

def optimize(
    strategy_class,
    specs: List[DataSpec],
    n_trials: int = 30,
    n_jobs: int = 1,
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION_PER_TRANSACTION,
    pruner: Optional[str] = "hyperband",
    storage_string: Optional[str] = None,
    study_name: str = "strategy_opt",
    seed: int = 42,
):
    storage = build_optuna_storage(storage_string)

    # Preload Polars DataFrames once
    console.print("[bold blue]Preloading data from MSSQL/Polars...[/bold blue]")
    df_map = preload_polars(specs)
    console.print(f"[green]✓ Loaded {len(df_map)} assets[/green]")

    # Pruner
    pruner_obj = make_pruner(num_steps=len(specs), pruner=pruner, n_jobs=n_jobs)

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        constant_liar=True,               # better parallel behavior
        warn_independent_sampling=False,  # silence noisy warning if any edge remains
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner_obj,
        storage=storage,
        study_name=study_name,
        load_if_exists=(storage is not None),
    )

    objective = make_objective(strategy_class, specs, df_map, init_cash, commission)

    console.print(f"Starting Optuna: trials={n_trials}, n_jobs={n_jobs}, pruner={pruner}")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=True)
    console.print(f"[bold green]Done. Best value: {study.best_value:.4f}[/bold green]")

    table = Table(title="Best Parameters", show_lines=True)
    table.add_column("Param")
    table.add_column("Value")
    for k, v in study.best_params.items():
        table.add_row(k, str(v))
    console.print(table)

    return study, study.best_params

# --------------- Script entry ---------------
bull_start = "2020-09-28"
bull_end = "2021-05-31"
bear_start = "2022-05-28"
bear_end = "2023-06-23"
# Optional holdout test period
test_bull_start="2023-06-12"
test_bull_end="2025-05-31"
tf = "1m"


if __name__ == "__main__":
    # Get ODBC string
    try:
        from backtrader.dontcommit import connection_string as MSSQL_ODBC
    except Exception:
        MSSQL_ODBC = os.getenv("MSSQL_ODBC", None)

    # Define training universe
    specs = [
        DataSpec("BTC", interval=tf, ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        DataSpec("ETH", interval=tf, ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        DataSpec("LTC", interval=tf, ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        DataSpec("XRP", interval=tf, ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        DataSpec("BCH", interval=tf, ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
    ]

    study, best_params = optimize(
        strategy_class=StrategyClass,
        specs=specs,
        n_trials=150,
        n_jobs=12,
        init_cash=INIT_CASH,
        commission=COMMISSION_PER_TRANSACTION,
        pruner="hyperband",
        storage_string=MSSQL_ODBC,  # None => in-memory
        study_name="BullBearMarketBTC-ETH-LTC-XRP-BCH_1m_MACD_ADXV3",
        seed=42,
    )

    # Optional holdout using your existing backtest() pipeline:
    from backtrader.utils.backtest import backtest
    backtest(
        StrategyClass,
        coin="BTC",
        start_date=test_bull_start,
        end_date=test_bull_end,
        interval=tf,
        init_cash=1000,
        plot=True,
        quantstats=False,
        params=best_params,
    )