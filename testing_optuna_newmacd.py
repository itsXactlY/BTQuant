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
from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX4 as StrategyClass

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


def compat_params_to_mtf(p: dict) -> dict:
    mtf = dict(
        # MTF/Guards
        tf5m_breakout_period=p.get('breakout_period', 55),
        adxth=p.get('adxth', 22),
        confirm_bars=p.get('confirm_bars', 2),
        max_stretch_atr_mult=p.get('stretch_atr_mult', 1.0),

        # Stops/Trail
        atr_stop_mult=p.get('init_sl_atr_mult', 1.25),
        trail_mode=p.get('trail_mode', 'chandelier'),
        trail_atr_mult=p.get('trail_atr_mult', 4.0),
        ema_band_mult=p.get('ema_band_mult', 2.0),
        donchian_trail_period=p.get('donchian_trail_period', 55),
        move_to_breakeven_R=p.get('trail_arm_R', 1.0),
        trail_update_every=p.get('trail_update_every', 3),

        # Core
        ema_fast=p.get('ema_fast', 20),
        ema_slow=p.get('ema_slow', 50),
        ema_trend=p.get('ema_trend', 200),
        atr_period=p.get('atr_period', 14),

        # TP
        take_profit=p.get('take_profit', 4.0),

        # Volume
        use_volume_filter=(p.get('volume_filter_mult', 1.2) > 1.0),
        volume_filter_mult=p.get('volume_filter_mult', 1.2),

        # Pyramiding (ersetzt DCA)
        use_pyramiding=(p.get('max_adds', 0) > 0 or p.get('use_dca', False)),
        max_adds=p.get('max_adds', 0),
        add_cooldown=p.get('add_cooldown', 20),
        add_atr_mult=p.get('dca_atr_mult', 1.0),
        add_min_R=p.get('add_min_R', 1.0),

        # Defaults
        rsi_overheat=75,
        use_htf=True,
    )
    return mtf

def suggest_params_mtf(trial) -> dict:
    return dict(
        tf5m_breakout_period=trial.suggest_int("tf5m_breakout_period", 35, 75, step=5),
        adxth=trial.suggest_int("adxth", 18, 28),
        confirm_bars=trial.suggest_int("confirm_bars", 1, 3),
        max_stretch_atr_mult=trial.suggest_float("max_stretch_atr_mult", 0.8, 1.3, step=0.1),

        atr_stop_mult=trial.suggest_float("atr_stop_mult", 2.0, 3.0, step=0.25),
        trail_mode=trial.suggest_categorical("trail_mode", ["chandelier","ema_band","donchian"]),
        trail_atr_mult=trial.suggest_float("trail_atr_mult", 3.0, 5.0, step=0.25),
        ema_band_mult=trial.suggest_float("ema_band_mult", 1.6, 2.6, step=0.2),
        donchian_trail_period=trial.suggest_int("donchian_trail_period", 45, 75, step=5),
        move_to_breakeven_R=trial.suggest_float("move_to_breakeven_R", 0.5, 1.5, step=0.25),
        trail_update_every=trial.suggest_int("trail_update_every", 2, 6),

        ema_fast=trial.suggest_int("ema_fast", 18, 26),
        ema_slow=trial.suggest_int("ema_slow", 40, 60),
        ema_trend=trial.suggest_int("ema_trend", 150, 220),
        atr_period=trial.suggest_int("atr_period", 12, 18),

        take_profit=trial.suggest_float("take_profit", 3.0, 8.0, step=0.5),

        use_volume_filter=trial.suggest_categorical("use_volume_filter", [False, True]),
        volume_filter_mult=trial.suggest_float("volume_filter_mult", 1.1, 1.8, step=0.1),

        use_pyramiding=trial.suggest_categorical("use_pyramiding", [False, True]),
        max_adds=trial.suggest_int("max_adds", 0, 2),
        add_cooldown=trial.suggest_int("add_cooldown", 15, 40, step=5),
        add_atr_mult=trial.suggest_float("add_atr_mult", 0.5, 1.5, step=0.25),
        add_min_R=trial.suggest_float("add_min_R", 0.5, 1.5, step=0.25),

        risk_per_trade_pct=trial.suggest_float("risk_per_trade_pct", 0.001, 0.005, step=0.0005),
        close_based_stop=True,
    )



def score_sharpe_dd(strat, lam_dd=0.03):
    """
    Score = Sharpe - lam_dd * (MaxDD/100). Maximiert Sharpe, bestraft hohen DD.
    """
    import math
    try:
        sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr) if sr is not None and not math.isnan(sr) else 0.0
    except Exception:
        sharpe = 0.0

    try:
        draw = strat.analyzers.drawdown.get_analysis()
        mdd = float(draw.get("max", {}).get("drawdown", 0.0))
    except Exception:
        mdd = 0.0

    try:
        ta = strat.analyzers.trades.get_analysis()
        trades = ta.get("total", {}).get("total", 0) or 0
        won = ta.get("won", {}).get("total", 0) or 0
        lost = ta.get("lost", {}).get("total", 0) or 0
        win_rate = (won / trades) if trades > 0 else 0.0
    except Exception:
        trades, won, lost, win_rate = 0, 0, 0, 0.0

    score = sharpe - lam_dd * (mdd / 100.0)
    metrics = dict(mdd=mdd, sharpe=sharpe, trades=trades, win_rate=win_rate)
    return score, metrics





# --------------- Search space - Optimized for 1m timeframe ---------------
def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Parameters tuned for 1m timeframe profitability"""
    p = dict(
        # Entry parameters - balanced for 1m
        breakout_period=trial.suggest_int("breakout_period", 15, 35, step=5),
        adxth=trial.suggest_int("adxth", 20, 32),  # Lowered for more signals
        vol_mult=trial.suggest_float("vol_mult", 1.2, 2.5, step=0.1),
        stretch_atr_mult=trial.suggest_float("stretch_atr_mult", 0.8, 1.3, step=0.1),

        # Conservative DCA
        use_dca=trial.suggest_categorical("use_dca", [True, False]),
        max_adds=trial.suggest_int("max_adds", 1, 3),
        add_cooldown=trial.suggest_int("add_cooldown", 15, 40, step=5),
        dca_atr_mult=trial.suggest_float("dca_atr_mult", 1.2, 2.0, step=0.1),
        add_on_ema_touch=trial.suggest_categorical("add_on_ema_touch", [True, False]),

        # Faster indicators for 1m
        macd_period_me1=trial.suggest_int("macd_period_me1", 8, 14),
        macd_period_me2=trial.suggest_int("macd_period_me2", 18, 28),
        macd_period_signal=trial.suggest_int("macd_period_signal", 6, 10),
        adx_period=trial.suggest_int("adx_period", 10, 18),
        di_period=trial.suggest_int("di_period", 10, 18),
        momentum_period=trial.suggest_int("momentum_period", 10, 18),
        rsi_period=trial.suggest_int("rsi_period", 12, 20),
        stoch_period=trial.suggest_int("stoch_period", 12, 20),
        cci_period=trial.suggest_int("cci_period", 14, 24),
        trix_period=trial.suggest_int("trix_period", 12, 22),

        # EMA settings optimized for 1m
        ema_fast=trial.suggest_int("ema_fast", 18, 25),
        ema_slow=trial.suggest_int("ema_slow", 38, 55),
        ema_trend=trial.suggest_int("ema_trend", 100, 180),  # Shorter for 1m
        atr_period=trial.suggest_int("atr_period", 12, 18),
        vol_window=trial.suggest_int("vol_window", 15, 25),
        
        # Realistic take profit for 1m scalping
        take_profit=trial.suggest_float("take_profit", 3.0, 8.0, step=0.5),
        
        # Partial exits
        use_partial_exits=trial.suggest_categorical("use_partial_exits", [True, False]),
        partial_exit_1_pct=trial.suggest_float("partial_exit_1_pct", 0.3, 0.6, step=0.1),
        partial_exit_1_target=trial.suggest_float("partial_exit_1_target", 1.5, 3.5, step=0.5),
        
        # More reasonable entry filters
        min_signal_strength=trial.suggest_float("min_signal_strength", 0.5, 0.75, step=0.05),
        volume_filter_mult=trial.suggest_float("volume_filter_mult", 1.2, 2.0, step=0.1),
        trend_filter_strength=trial.suggest_float("trend_filter_strength", 0.4, 0.7, step=0.05),

        # Trailing stops
        trail_mode=trial.suggest_categorical("trail_mode", ["chandelier", "ema_band", "donchian"]),
        trail_atr_mult=trial.suggest_float("trail_atr_mult", 2.0, 4.5, step=0.25),
        ema_band_mult=trial.suggest_float("ema_band_mult", 1.5, 2.8, step=0.2),
        donchian_trail_period=trial.suggest_int("donchian_trail_period", 30, 70, step=5),
        pivot_left=trial.suggest_int("pivot_left", 1, 3),
        pivot_right=trial.suggest_int("pivot_right", 1, 3),
        init_sl_atr_mult=trial.suggest_float("init_sl_atr_mult", 1.0, 1.8, step=0.1),
        trail_arm_R=trial.suggest_float("trail_arm_R", 1.2, 3.0, step=0.2),
        trail_arm_bars=trial.suggest_int("trail_arm_bars", 15, 60, step=5),
        trail_update_every=trial.suggest_int("trail_update_every", 3, 15, step=2),
    )
    return p

# --------------- Improved Scoring for 1m timeframe ---------------
def score_from_analyzers(
    strat,
    min_trades: int = 20,     # Adjusted for 1m timeframe
    max_trades: int = 300,    # Increased for high-frequency trading
    weight_roi: float = 0.6,      # Slightly reduced ROI weight
    weight_drawdown: float = 0.25,  
    weight_quality: float = 0.15,   # Increased quality weight
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
    sharpe = max(-3.0, min(6.0, sharpe))

    try:
        ret = strat.analyzers.returns.get_analysis()
        total_return = float(ret.get("rtot", 0.0))
        cagr_dec = float(ret.get("rnorm", 0.0))
    except Exception:
        total_return, cagr_dec = 0.0, 0.0

    try:
        ta = strat.analyzers.trades.get_analysis()
        total_trades = ta.get("total", {}).get("total", 0) or 0
        won = ta.get("won", {}).get("total", 0) or 0
        lost = ta.get("lost", {}).get("total", 0) or 0
        win_rate = (won / total_trades) if total_trades > 0 else 0.0
        
        won_total = ta.get("won", {}).get("pnl", {}).get("total", 0) or 0
        lost_total = ta.get("lost", {}).get("pnl", {}).get("total", 0) or 0
        avg_win = (won_total / won) if won > 0 else 0.0
        avg_loss = abs(lost_total / lost) if lost > 0 else 0.0
        profit_factor = (won_total / abs(lost_total)) if lost_total != 0 else float('inf')
        
    except Exception:
        total_trades, win_rate, avg_win, avg_loss, profit_factor = 0, 0.0, 0.0, 0.0, 0.0

    # More forgiving base score calculation
    roi_score = total_return * 3.0  # Reduced multiplier
    drawdown_penalty = -(mdd / 100.0) * 6.0  # Reduced penalty
    
    # Quality scoring - more realistic thresholds for 1m trading
    quality_score = 0.0
    if total_trades > 0:
        # Lower win rate expectations for 1m scalping
        if win_rate > 0.35:  # Lowered from 0.4
            quality_score += (win_rate - 0.35) * 4.0  # Reduced multiplier
        elif win_rate < 0.25:  # More forgiving threshold
            quality_score -= (0.25 - win_rate) * 6.0  # Reduced penalty
            
        # More realistic profit factor for high-frequency trading
        if profit_factor > 1.1:  # Lowered threshold
            quality_score += min((profit_factor - 1.1) * 2.0, 2.5)
        elif profit_factor < 0.9:
            quality_score -= (0.9 - profit_factor) * 4.0  # Reduced penalty
            
        # Risk/reward ratio - more realistic for 1m
        if avg_win > 0 and avg_loss > 0:
            rr_ratio = avg_win / avg_loss
            if rr_ratio > 1.0:  # Lowered expectation
                quality_score += min((rr_ratio - 1.0) * 1.2, 1.5)
            elif rr_ratio < 0.7:  # More realistic threshold
                quality_score -= (0.7 - rr_ratio) * 2.5

    # Adjusted overtrading penalty for 1m timeframe
    overtrade_penalty = 0.0
    if total_trades > max_trades:
        overtrade_penalty = -((total_trades - max_trades) * 0.02)  # Reduced penalty
    elif total_trades > max_trades * 0.7:
        overtrade_penalty = -((total_trades - max_trades * 0.7) * 0.01)

    # Under-trading penalty
    undertrade_penalty = 0.0
    if total_trades == 0:
        console.print(f"[red]WARNING: Zero trades detected! Strategy too restrictive.[/red]")
        undertrade_penalty = -8.0  # Fixed penalty instead of scaling
    elif total_trades < min_trades:
        undertrade_penalty = -((min_trades - total_trades) * 0.2)  # Reduced penalty

    # Less harsh drawdown penalty
    extreme_dd_penalty = 0.0
    if mdd > 30:  # Higher threshold
        extreme_dd_penalty = -(mdd - 30) * 0.2  # Reduced multiplier

    # Final score with base offset to avoid negative scores during exploration
    base_score = 2.0  # Give everything a base score
    score = (
        base_score +
        weight_roi * roi_score +
        weight_drawdown * drawdown_penalty +
        weight_quality * quality_score +
        overtrade_penalty +
        undertrade_penalty +
        extreme_dd_penalty
    )

    # Smaller bonus for excellent performance
    if total_return > 5 and mdd < 15 and win_rate > 0.35:  # More realistic thresholds
        score += 2.0

    metrics = dict(
        mdd=mdd, 
        sharpe=sharpe, 
        total_return=total_return,
        cagr=cagr_dec, 
        trades=total_trades, 
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss
    )
    
    return score, metrics

# --------------- Single backtest ---------------
# def run_single_backtest_eval(
#     strategy_class,
#     df: pl.DataFrame,
#     spec: DataSpec,
#     init_cash: float,
#     commission: float,
#     params: Dict[str, Any],
# ) -> Tuple[float, Dict[str, float], float]:
#     cerebro = bt.Cerebro(oldbuysell=True)

#     # Wrap DF into your PolarsData feed
#     feed = make_feed_from_df(df, spec)
#     cerebro.adddata(feed)

#     cerebro.addstrategy(strategy_class, backtest=True, **params)

#     # Analyzers (match your backtest style without pyfolio overhead)
#     cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
#     cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
#     cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
#     cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

#     # Observers
#     cerebro.addobserver(bt.observers.Value)
#     cerebro.addobserver(bt.observers.DrawDown)
#     cerebro.addobserver(bt.observers.Cash)

#     cerebro.broker.setcommission(commission=commission)
#     cerebro.broker.setcash(init_cash)

#     try:
#         strats = cerebro.run()
#         strat = strats[0]
#     except Exception as e:
#         console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]")
#         return -999.0, {"error": 1.0}, 0.0

#     score, metrics = score_from_analyzers(strat)
#     final_value = cerebro.broker.getvalue()

#     # Cleanup
#     del cerebro, feed, strats, strat
#     gc.collect()

#     return score, metrics, final_value

def run_single_backtest_eval(
    strategy_class,
    df: pl.DataFrame,
    spec: DataSpec,
    init_cash: float,
    commission: float,
    params: Dict[str, Any],
    exchange: Optional[str] = None,    # NEW
    slippage_bps: float = 5.0,         # NEW
    min_qty: float = 0.0,              # NEW
    qty_step: float = 1.0,             # NEW
    price_tick: Optional[float] = None,# NEW
    params_mode: str = "compat",       # "compat" (map altes Set) oder "mtf"
    score_fn: Callable = score_sharpe_dd,  # NEW
) -> Tuple[float, Dict[str, float], float]:
    cerebro = bt.Cerebro(oldbuysell=True)

    # Feed
    feed = make_feed_from_df(df, spec)
    cerebro.adddata(feed)
    # MTF Resamples
    cerebro.resampledata(feed, timeframe=bt.TimeFrame.Minutes, compression=5)
    cerebro.resampledata(feed, timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.resampledata(feed, timeframe=bt.TimeFrame.Minutes, compression=60)

    # Params vorbereiten
    if params_mode == "compat":
        strat_params = compat_params_to_mtf(params)
    else:
        strat_params = dict(params)

    # Short nur auf MEXC
    can_short = (str(exchange).lower() == "mexc") if exchange else False
    strat_params.update(dict(
        can_short=can_short,
        min_qty=min_qty,
        qty_step=qty_step,
        price_tick=price_tick
    ))

    cerebro.addstrategy(strategy_class, backtest=True, **strat_params)

    # Analyzers (gleich wie vorher, Namen wichtig)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)

    cerebro.broker.setcommission(commission=commission)
    try:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception:
        pass
    cerebro.broker.setcash(init_cash)

    try:
        strats = cerebro.run()
        strat = strats[0]
    except Exception as e:
        console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]")
        return -999.0, {"error": 1.0}, 0.0

    score, metrics = score_fn(strat)
    final_value = cerebro.broker.getvalue()

    # Cleanup
    del cerebro, feed, strats, strat
    gc.collect()

    return score, metrics, final_value

# --------------- Objective with relaxed constraints ---------------
# def make_objective(
#     strategy_class,
#     specs: List[DataSpec],
#     df_map: Dict[str, pl.DataFrame],
#     init_cash: float,
#     commission: float,
# ) -> Callable[[optuna.Trial], float]:
#     def objective(trial: optuna.Trial) -> float:
#         params = suggest_params(trial)

#         # Much more relaxed constraints
#         if params["ema_slow"] - params["ema_fast"] < 6:
#             trial.set_user_attr("constraint_fail", "ema_gap")
#             return -999
#         if params["ema_trend"] - params["ema_slow"] < 30:
#             trial.set_user_attr("constraint_fail", "trend_gap") 
#             return -999

#         # Only check severe DCA misconfigurations
#         if params["max_adds"] > 2 and params["take_profit"] < 3.0:
#             trial.set_user_attr("constraint_fail", "risky_dca_tp_combo")
#             return -999

#         # Very lenient risk/reward check
#         if params["init_sl_atr_mult"] > params["take_profit"] / 2.0:
#             trial.set_user_attr("constraint_fail", "poor_risk_reward")
#             return -999

#         total_score = 0.0
#         wsum = 0.0
#         step = 0
#         weights = [1.0 + 0.1 * i for i in range(len(specs))]

#         total_trades_all = 0
#         total_returns_all = []
#         valid_results = 0

#         for idx, spec in enumerate(specs):
#             df = df_map[spec.symbol]
            
#             try:
#                 score, metrics, _ = run_single_backtest_eval(
#                     strategy_class=strategy_class,
#                     df=df,
#                     spec=spec,
#                     init_cash=init_cash,
#                     commission=commission,
#                     params=params,
#                 )
                
#                 # Only count valid results (not errors)
#                 if score > -999:
#                     w = weights[idx]
#                     total_score += score * w
#                     wsum += w
#                     valid_results += 1
                    
#                     total_trades_all += metrics.get("trades", 0)
#                     total_returns_all.append(metrics.get("total_return", 0.0))

#                     trial.set_user_attr(f"{spec.symbol}_score", score)
#                     trial.set_user_attr(f"{spec.symbol}_return", metrics.get("total_return", 0.0))
#                     trial.set_user_attr(f"{spec.symbol}_mdd", metrics.get("mdd", 0.0))
#                     trial.set_user_attr(f"{spec.symbol}_trades", metrics.get("trades", 0))
#                     trial.set_user_attr(f"{spec.symbol}_win_rate", metrics.get("win_rate", 0.0))
#                 else:
#                     console.print(f"[yellow]Warning: {spec.symbol} backtest failed (score: {score})[/yellow]")
                    
#             except Exception as e:
#                 console.print(f"[red]Exception in {spec.symbol}: {e}[/red]")
#                 continue

#             step += 1
#             trial.report(total_score / max(wsum, 1e-9), step=step)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()

#         # If no valid results, return penalty
#         if valid_results == 0:
#             console.print(f"[red]Trial {trial.number}: No valid backtest results[/red]")
#             return -999.0

#         # Aggregate scoring adjustments
#         if valid_results > 0:
#             avg_trades = total_trades_all / valid_results
#             avg_return = sum(total_returns_all) / len(total_returns_all) if total_returns_all else 0.0

#             # Less harsh overtrading penalty
#             if avg_trades > 200:  # Increased threshold for 1m
#                 total_score -= (avg_trades - 200) * 0.05

#             # Bonus for consistent positive performance
#             positive_assets = sum(1 for r in total_returns_all if r > 1.0)  # Lowered threshold
#             if positive_assets >= valid_results * 0.6:  # 60% instead of 80%
#                 total_score += 1.5

#         final_score = total_score / max(wsum, 1e-9)
#         return final_score

#     return objective

def make_objective(
    strategy_class,
    specs: List[DataSpec],
    df_map: Dict[str, pl.DataFrame],
    init_cash: float,
    commission: float,
    exchange: Optional[str] = None,          # NEW
    slippage_bps: float = 5.0,               # NEW
    min_qty: float = 0.0, qty_step: float = 1.0, price_tick: Optional[float] = None,  # NEW
    param_mode: str = "compat",              # "compat" oder "mtf"
    scoring: str = "sharpe_dd",              # "sharpe_dd" oder "legacy"
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        # Parameter
        if param_mode == "mtf":
            params = suggest_params_mtf(trial)
        else:
            params = suggest_params(trial)  # dein alter Suchraum

        total_score = 0.0
        wsum = 0.0
        weights = [1.0 + 0.1 * i for i in range(len(specs))]
        valid_results = 0

        for idx, spec in enumerate(specs):
            df = df_map[spec.symbol]
            try:
                score, metrics, _ = run_single_backtest_eval(
                    strategy_class=strategy_class,
                    df=df, spec=spec,
                    init_cash=init_cash, commission=commission,
                    params=params,
                    exchange=exchange,
                    slippage_bps=slippage_bps,
                    min_qty=min_qty, qty_step=qty_step, price_tick=price_tick,
                    params_mode=("mtf" if param_mode=="mtf" else "compat"),
                    score_fn=(score_sharpe_dd if scoring=="sharpe_dd" else score_from_analyzers)
                )
                if score > -999:
                    w = weights[idx]
                    total_score += score * w
                    wsum += w
                    valid_results += 1
                    trial.set_user_attr(f"{spec.symbol}_score", score)
                    trial.set_user_attr(f"{spec.symbol}_mdd", metrics.get("mdd", 0.0))
                    trial.set_user_attr(f"{spec.symbol}_sharpe", metrics.get("sharpe", 0.0))
                    trial.set_user_attr(f"{spec.symbol}_trades", metrics.get("trades", 0))
            except Exception as e:
                console.print(f"[red]Exception in {spec.symbol}: {e}[/red]")
                continue

            trial.report(total_score / max(wsum, 1e-9), step=idx+1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if valid_results == 0:
            console.print(f"[red]Trial {trial.number}: No valid backtest results[/red]")
            return -999.0

        return total_score / max(wsum, 1e-9)
    return objective


# --------------- Basic strategy test function ---------------
def test_strategy_basic():
    """Quick test to see if strategy can generate trades with basic parameters"""
    console.print("[bold yellow]Testing basic strategy functionality...[/bold yellow]")
    
    # Very basic, aggressive parameters
    test_params = {
        'breakout_period': 20,
        'adxth': 20,  # Very low
        'vol_mult': 1.0,  # Very low
        'stretch_atr_mult': 1.0,
        'use_dca': False,
        'max_adds': 1,
        'add_cooldown': 20,
        'dca_atr_mult': 1.5,
        'add_on_ema_touch': False,
        'macd_period_me1': 12,
        'macd_period_me2': 26,
        'macd_period_signal': 9,
        'adx_period': 14,
        'di_period': 14,
        'momentum_period': 14,
        'rsi_period': 14,
        'stoch_period': 14,
        'cci_period': 20,
        'trix_period': 18,
        'ema_fast': 20,
        'ema_slow': 40,
        'ema_trend': 100,  # Much shorter
        'atr_period': 14,
        'vol_window': 20,
        'take_profit': 5.0,
        'use_partial_exits': False,
        'partial_exit_1_pct': 0.5,
        'partial_exit_1_target': 3.0,
        'min_signal_strength': 0.3,  # Very low
        'volume_filter_mult': 1.0,  # No volume filter
        'trend_filter_strength': 0.3,  # Very low
        'trail_mode': 'chandelier',
        'trail_atr_mult': 2.5,
        'ema_band_mult': 2.0,
        'donchian_trail_period': 50,
        'pivot_left': 2,
        'pivot_right': 2,
        'init_sl_atr_mult': 1.2,
        'trail_arm_R': 1.0,  # Very low
        'trail_arm_bars': 10,  # Very fast
        'trail_update_every': 5,
    }
    
    # Test on BTC data
    specs = [DataSpec("BTC", interval="1m", ranges=[("2020-09-28", "2021-05-31")])]
    df_map = preload_polars(specs)
    
    score, metrics, _ = run_single_backtest_eval(
        strategy_class=StrategyClass,
        df=df_map["BTC"],
        spec=specs[0],
        init_cash=1000,
        commission=0.00075,
        params=test_params,
    )
    
    console.print(f"[cyan]Basic test results:[/cyan]")
    console.print(f"Score: {score}")
    console.print(f"Trades: {metrics.get('trades', 0)}")
    console.print(f"Return: {metrics.get('total_return', 0)}%")
    console.print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    
    if metrics.get('trades', 0) > 0:
        console.print("[green]✓ Strategy can generate trades![/green]")
        return True
    else:
        console.print("[red]✗ Strategy still generates no trades even with aggressive parameters[/red]")
        return False

# --------------- Optimize ---------------
def make_pruner(num_steps: int, pruner: Optional[str], n_jobs: int):
    """
    num_steps = len(specs) i.e., how many trial.report steps you produce.
    """
    if pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max(1, num_steps),
            reduction_factor=3,
            bootstrap_count=max(2 * n_jobs, 8),
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
        constant_liar=True,
        warn_independent_sampling=False,
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

    # Enhanced results table
    table = Table(title="Best Parameters", show_lines=True)
    table.add_column("Parameter")
    table.add_column("Value")
    table.add_column("Description")
    
    param_descriptions = {
        "adxth": "ADX threshold (optimized for 1m)",
        "take_profit": "Take profit % (realistic for scalping)",
        "min_signal_strength": "Entry filter strength",
        "volume_filter_mult": "Volume requirement",
        "use_partial_exits": "Partial profit taking",
        "trail_mode": "Trailing stop method",
        "ema_trend": "Trend filter period (shortened)"
    }
    
    for k, v in study.best_params.items():
        desc = param_descriptions.get(k, "")
        table.add_row(k, str(v), desc)
    console.print(table)

    # Show some key stats from best trial
    if hasattr(study.best_trial, 'user_attrs'):
        console.print("\n[bold cyan]Best Trial Performance:[/bold cyan]")
        for attr, value in study.best_trial.user_attrs.items():
            if "_return" in attr or "_trades" in attr or "_win_rate" in attr:
                console.print(f"{attr}: {value}")

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
        from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC
    except Exception:
        MSSQL_ODBC = os.getenv("MSSQL_ODBC", None)

    # Test basic functionality first
    if test_strategy_basic():
        console.print("[green]Strategy works, proceeding with optimization...[/green]")
        
        # Define training universe
        specs = [
            DataSpec("BTC", interval="1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
            DataSpec("ETH", interval="1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        ]
        df_map = preload_polars(specs)

        objective = make_objective(
            strategy_class=StrategyClass,
            specs=specs,
            df_map=df_map,
            init_cash=INIT_CASH,
            commission=COMMISSION_PER_TRANSACTION,
            exchange="MEXC",         # Short only on MEXC
            slippage_bps=5,
            min_qty=0.001, qty_step=0.001, price_tick=0.1,
            param_mode="mtf",         # oder "compat" für alten Suchraum
            scoring="sharpe_dd"       # Sharpe-first, DD-penalized
        )

        study, best_params = optimize(
            strategy_class=StrategyClass,
            specs=specs,
            n_trials=200,
            n_jobs=12,
            init_cash=INIT_CASH,
            commission=COMMISSION_PER_TRANSACTION,
            pruner="hyperband",
            storage_string=MSSQL_ODBC,  # None => in-memory
            study_name="Optimized_1m_MACD_ADX_BullBear_V3",
            seed=42,
        )

        # Optuna setup wie gehabt...
        study.optimize(objective, n_trials=200, n_jobs=12, gc_after_trial=True, show_progress_bar=True)
    
        console.print(f"[bold green]Optimization complete. Best value: {study.best_value:.4f}[/bold green]")