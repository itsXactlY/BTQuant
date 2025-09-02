''' After days and days of working on this, here it is: a full parallel Optuna + Polars + Cache + SQLite/MSSQL + Hyperopt example, on somewhat more quantitative strategy.'''

# optuna_hyperopt_polars.py
import os
import gc
import math
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable

import backtrader as bt
import optuna
import polars as pl
from rich.console import Console
from rich.table import Table
from pathlib import Path
import multiprocessing as mp
import hashlib

console = Console()

from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData
from backtrader.dontcommit import connection_string as MSSQL_ODBC

INIT_CASH = 100_000_000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"

# Parquet cache dir (override with env BTQ_CACHE_DIR)
CACHE_DIR = Path(os.getenv("BTQ_CACHE_DIR", ".btq_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- Data spec ---------------------
@dataclass(frozen=True)
class DataSpec:
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL

# --------------------- Optuna storage helpers ---------------------
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

    extra = {"driver": driver, "Encrypt": "yes", "TrustServerCertificate": parts.get("TRUSTSERVERCERTIFICATE", "yes")}
    query = "&".join(f"{k}={v}" for k, v in extra.items() if v is not None)
    encoded_pwd = urllib.parse.quote_plus(pwd)
    return f"mssql+pyodbc://{uid}:{encoded_pwd}@{server}/{database}?{query}"

def ensure_storage_or_sqlite(storage_string: Optional[str], study_name: str) -> optuna.storages.RDBStorage:
    """
    Try to use provided MSSQL/ODBC storage; if it fails or is None, fall back to SQLite file in CACHE_DIR.
    """
    if storage_string is None:
        sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
        console.print(f"[yellow]No storage provided → using SQLite at {sqlite_path}[/yellow]")
        return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")

    if not storage_string.lower().startswith("mssql+pyodbc://"):
        storage_url = mssql_url_from_odbc(storage_string)
    else:
        storage_url = storage_string

    try:
        console.print(f"Optuna storage URL: {storage_url.split('@')[0]}@***")
        return optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300, "connect_args": {"timeout": 30}},
        )
    except Exception as e:
        console.print(f"[red]RDB storage init failed: {e}[/red]")
        sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
        console.print(f"[yellow]Falling back to SQLite at {sqlite_path}[/yellow]")
        return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")

# --------------------- Parquet cache ---------------------
def _cache_key(spec: DataSpec) -> str:
    ranges = spec.ranges or [(spec.start_date or "", spec.end_date or "")]
    ranges_str = ",".join([f"{s}:{e}" for s, e in ranges])
    raw = f"{spec.symbol}|{spec.interval}|{spec.collateral}|{ranges_str}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{spec.symbol}_{spec.interval}_{spec.collateral}_{h}.parquet"

def _cache_path(spec: DataSpec) -> Path:
    return CACHE_DIR / _cache_key(spec)

def _try_load_cache(spec: DataSpec) -> Optional[pl.DataFrame]:
    p = _cache_path(spec)
    if not p.exists():
        return None
    try:
        return pl.scan_parquet(str(p)).collect()
    except Exception as e:
        console.print(f"[yellow]Cache read failed for {spec.symbol}: {e}[/yellow]")
        return None

def _save_cache(spec: DataSpec, df: pl.DataFrame) -> None:
    p = _cache_path(spec)
    try:
        df.write_parquet(str(p), compression="zstd", statistics=True)
        console.print(f"[green]Cached {spec.symbol} -> {p}[/green]")
    except Exception as e:
        console.print(f"[yellow]Cache write failed for {spec.symbol}: {e}[/yellow]")

def preload_polars(specs: List[DataSpec], force_refresh: bool = False) -> Dict[str, pl.DataFrame]:
    """
    Load once from cache or DB into Polars, sorted by TimestampStart.
    """
    df_map: Dict[str, pl.DataFrame] = {}
    seen = set()
    for spec in specs:
        if spec.symbol in seen:
            continue

        df = None if force_refresh else _try_load_cache(spec)
        if df is None:
            if spec.ranges:
                parts = []
                for s, e in spec.ranges:
                    part = get_database_data(
                        ticker=spec.symbol, start_date=s, end_date=e,
                        time_resolution=spec.interval, pair=spec.collateral
                    )
                    if part is None or part.is_empty():
                        raise ValueError(f"No data for {spec.symbol} {spec.interval} {s}->{e}")
                    parts.append(part)
                df = pl.concat(parts).sort("TimestampStart")
            else:
                df = get_database_data(
                    ticker=spec.symbol, start_date=spec.start_date, end_date=spec.end_date,
                    time_resolution=spec.interval, pair=spec.collateral
                )
            if df is None or df.is_empty():
                raise ValueError(f"No data for {spec.symbol} {spec.interval} {spec.start_date}->{spec.end_date}")
            df = df.sort("TimestampStart")
            _save_cache(spec, df)
        else:
            console.print(f"[cyan]Loaded from cache: {spec.symbol}[/cyan]")

        required = {"TimestampStart", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"{spec.symbol}: missing required columns. Found: {df.columns}")

        df_map[spec.symbol] = df
        seen.add(spec.symbol)
    return df_map

def build_cache(specs: List[DataSpec], force_refresh=True):
    console.print(f"[bold blue]Building cache for {len(specs)} assets...[/bold blue]")
    t0 = time.time()
    df_map = preload_polars(specs, force_refresh=force_refresh)
    total_rows = sum(int(df.height) for df in df_map.values())
    console.print(f"[green]✓ Cache build complete. {len(df_map)} assets, {total_rows:,} rows. Took {time.time()-t0:.2f}s[/green]")

# --------------------- Feed factory ---------------------
def make_feed_from_df(df: pl.DataFrame, spec: DataSpec) -> MSSQLData:
    # No clone → lower memory; DataFrame not mutated by feed
    feed = MSSQLData(
        dataname=df,
        datetime="TimestampStart", open="Open", high="High", low="Low", close="Close", volume="Volume",
        timeframe=bt.TimeFrame.Minutes, compression=1
    )
    try:
        feed.p.timeframe = bt.TimeFrame.Minutes
        feed.p.compression = 1
    except Exception:
        setattr(feed, "_timeframe", bt.TimeFrame.Minutes)
        setattr(feed, "_compression", 1)
    try:
        feed._name = f"{spec.symbol}-{spec.interval}"
        feed._dataname = f"{spec.symbol}{spec.collateral}"
    except Exception:
        pass
    return feed

# --------------------- Param helpers (compat + MTF) ---------------------
def compat_params_to_mtf(p: dict) -> dict:
    return dict(
        tf5m_breakout_period=p.get('breakout_period', 55),
        adxth=p.get('adxth', 22),
        confirm_bars=p.get('confirm_bars', 2),
        max_stretch_atr_mult=p.get('stretch_atr_mult', 1.0),
        atr_stop_mult=p.get('init_sl_atr_mult', 1.25),
        trail_mode=p.get('trail_mode', 'chandelier'),
        trail_atr_mult=p.get('trail_atr_mult', 4.0),
        ema_band_mult=p.get('ema_band_mult', 2.0),
        donchian_trail_period=p.get('donchian_trail_period', 55),
        move_to_breakeven_R=p.get('trail_arm_R', 1.0),
        trail_update_every=p.get('trail_update_every', 3),
        ema_fast=p.get('ema_fast', 20),
        ema_slow=p.get('ema_slow', 50),
        ema_trend=p.get('ema_trend', 200),
        atr_period=p.get('atr_period', 14),
        take_profit=p.get('take_profit', 4.0),
        use_volume_filter=(p.get('volume_filter_mult', 1.2) > 1.0),
        volume_filter_mult=p.get('volume_filter_mult', 1.2),
        use_pyramiding=(p.get('max_adds', 0) > 0 or p.get('use_dca', False)),
        max_adds=p.get('max_adds', 0),
        add_cooldown=p.get('add_cooldown', 20),
        add_atr_mult=p.get('dca_atr_mult', 1.0),
        add_min_R=p.get('add_min_R', 1.0),
        rsi_overheat=75,
        use_htf=True,
    )

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

# Legacy suggestor (kept for compat mode)
def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    p = dict(
        breakout_period=trial.suggest_int("breakout_period", 15, 35, step=5),
        adxth=trial.suggest_int("adxth", 20, 32),
        vol_mult=trial.suggest_float("vol_mult", 1.2, 2.5, step=0.1),
        stretch_atr_mult=trial.suggest_float("stretch_atr_mult", 0.8, 1.3, step=0.1),
        use_dca=trial.suggest_categorical("use_dca", [True, False]),
        max_adds=trial.suggest_int("max_adds", 1, 3),
        add_cooldown=trial.suggest_int("add_cooldown", 15, 40, step=5),
        dca_atr_mult=trial.suggest_float("dca_atr_mult", 1.2, 2.0, step=0.1),
        add_on_ema_touch=trial.suggest_categorical("add_on_ema_touch", [True, False]),
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
        ema_fast=trial.suggest_int("ema_fast", 18, 25),
        ema_slow=trial.suggest_int("ema_slow", 38, 55),
        ema_trend=trial.suggest_int("ema_trend", 100, 180),
        atr_period=trial.suggest_int("atr_period", 12, 18),
        vol_window=trial.suggest_int("vol_window", 15, 25),
        take_profit=trial.suggest_float("take_profit", 3.0, 8.0, step=0.5),
        use_partial_exits=trial.suggest_categorical("use_partial_exits", [True, False]),
        partial_exit_1_pct=trial.suggest_float("partial_exit_1_pct", 0.3, 0.6, step=0.1),
        partial_exit_1_target=trial.suggest_float("partial_exit_1_target", 1.5, 3.5, step=0.5),
        min_signal_strength=trial.suggest_float("min_signal_strength", 0.5, 0.75, step=0.05),
        volume_filter_mult=trial.suggest_float("volume_filter_mult", 1.2, 2.0, step=0.1),
        trend_filter_strength=trial.suggest_float("trend_filter_strength", 0.4, 0.7, step=0.05),
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

# --------------------- Scoring ---------------------
def score_sharpe_dd(strat, lam_dd=0.03):
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
        trades, win_rate = 0, 0.0
    score = sharpe - lam_dd * (mdd / 100.0)
    metrics = dict(mdd=mdd, sharpe=sharpe, trades=trades, win_rate=win_rate)
    return score, metrics

# --------------------- Strategy (MTF + Risk + optional Short) ---------------------
class Enhanced_MACD_ADX3(bt.Strategy):
    params = (
        ('risk_per_trade_pct', 0.0025),
        ('max_leverage', 2.0),
        ('min_qty', 0.0),
        ('qty_step', 1.0),
        ('price_tick', None),
        ('round_prices', True),
        ('can_short', False),
        ('short_regime_mode', 'neutral'),
        ('rsi_oversold', 25),
        ('use_htf', True),
        ('tf5m_breakout_period', 55),
        ('tf15m_adx_period', 14),
        ('tf15m_ema_fast', 50),
        ('tf15m_ema_slow', 200),
        ('tf60m_ema_fast', 50),
        ('tf60m_ema_slow', 200),
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('ema_trend', 200),
        ('atr_period', 14),
        ('rsi_overheat', 75),
        ('adxth', 20),
        ('confirm_bars', 2),
        ('max_stretch_atr_mult', 1.0),
        ('atr_stop_mult', 2.5),
        ('use_trailing_stop', True),
        ('trail_mode', 'chandelier'),
        ('trail_atr_mult', 4.0),
        ('ema_band_mult', 2.0),
        ('donchian_trail_period', 55),
        ('close_based_stop', True),
        ('move_to_breakeven_R', 1.0),
        ('trail_update_every', 2),
        ('max_bars_in_trade', 6*60),
        ('reentry_cooldown_bars', 5),
        ('use_pyramiding', True),
        ('max_adds', 2),
        ('add_cooldown', 20),
        ('add_atr_mult', 1.0),
        ('add_min_R', 1.0),
        ('take_profit', 4.0),
        ('use_volume_filter', False),
        ('volume_filter_mult', 1.2),
        ('use_regime_long', True),
        ('use_trend_long', True),
        ('use_regime_short', True),
        ('use_trend_short', True),
        ('regime_mode_long', 'ema'),      # 'ema' | 'price_vs_slow' | 'off'
        ('regime_mode_short', 'neutral'), # 'ema' | 'neutral' | 'off'
        ('backtest', True),
        ('debug', False),
    )

    def __init__(self):
        self.d1  = self.datas[0]
        self.d5  = self.datas[1] if len(self.datas) > 1 else self.d1
        self.d15 = self.datas[2] if len(self.datas) > 2 else self.d5
        self.d60 = self.datas[3] if len(self.datas) > 3 else self.d15

        self.atr1 = bt.ind.ATR(self.d1, period=self.p.atr_period)
        self.ema1_fast = bt.ind.EMA(self.d1.close, period=self.p.ema_fast)
        self.ema1_slow = bt.ind.EMA(self.d1.close, period=self.p.ema_slow)
        self.ema1_trend = bt.ind.EMA(self.d1.close, period=self.p.ema_trend)
        self.rsi1 = bt.ind.RSI(self.d1, period=14)
        self.vsma1 = bt.ind.SMA(self.d1.volume, period=20) if self.p.use_volume_filter else None

        self.atr5 = bt.ind.ATR(self.d5, period=self.p.atr_period)
        self.dc_high5 = bt.ind.Highest(self.d5.high, period=self.p.tf5m_breakout_period)
        self.dc_low5  = bt.ind.Lowest(self.d5.low,  period=self.p.tf5m_breakout_period)
        self.dc_exit5_low  = bt.ind.Lowest(self.d5.low,  period=self.p.donchian_trail_period)
        self.dc_exit5_high = bt.ind.Highest(self.d5.high, period=self.p.donchian_trail_period)

        self.adx15 = bt.ind.ADX(self.d15, period=self.p.tf15m_adx_period)
        self.plusDI15 = bt.ind.PlusDI(self.d15, period=self.p.tf15m_adx_period)
        self.minusDI15 = bt.ind.MinusDI(self.d15, period=self.p.tf15m_adx_period)
        self.ema15_fast = bt.ind.EMA(self.d15.close, period=self.p.tf15m_ema_fast)
        self.ema15_slow = bt.ind.EMA(self.d15.close, period=self.p.tf15m_ema_slow)

        self.ema60_fast = bt.ind.EMA(self.d60.close, period=self.p.tf60m_ema_fast)
        self.ema60_slow = bt.ind.EMA(self.d60.close, period=self.p.tf60m_ema_slow)

        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.last_trail_update = -10**9
        self.n_adds = 0
        self.last_add_bar = -10**9
        self.last_exit_bar = -10**9

        self.active_orders, self.entry_prices, self.sizes = [], [], []
        self.block_counts = dict(regime=0, trend=0, breakout=0, s_regime=0, s_trend=0, breakdown=0)

    def start(self):
        if self.p.debug:
            names = ['Ticks', 'MicroSec', 'Seconds', 'Minutes', 'Days', 'Weeks', 'Months', 'Years']
            for i, d in enumerate(self.datas):
                name = getattr(d, '_name', f'data{i}')
                tf = getattr(d.p, 'timeframe', getattr(d, '_timeframe', None))
                comp_p = getattr(d.p, 'compression', None)
                comp_attr = getattr(d, '_compression', None)
                tfstr = names[int(tf)] if isinstance(tf, int) and 0 <= tf < len(names) else str(tf)
                print(f"Data{i} {name} -> TF={tfstr} p.comp={comp_p} attr._comp={comp_attr}")

    def stop(self):
        if self.p.debug:
            print("Blocks:", self.block_counts)

    def _equity(self): return self.broker.getvalue()
    def _round_qty(self, size):
        step = float(self.p.qty_step) if self.p.qty_step else 1.0
        size = math.floor(size / step) * step
        if self.p.min_qty and size < self.p.min_qty: return 0.0
        return size
    def _round_price(self, price):
        if not (self.p.round_prices and self.p.price_tick): return float(price)
        tick = float(self.p.price_tick);  return round(price / tick) * tick
    def _risk_based_size(self, entry, stop):
        eq = self._equity(); risk = eq * self.p.risk_per_trade_pct
        dist = max(1e-8, abs(entry - stop))
        size = min(risk / dist, (eq * self.p.max_leverage) / entry)
        return self._round_qty(size)
    def _avg_entry(self):
        if self.entry_prices and self.sizes:
            tot = sum(self.sizes);  return sum(p*s for p,s in zip(self.entry_prices, self.sizes))/tot if tot else None
        return None
    def _R(self):
        ae = self._avg_entry()
        if not (ae and self.initial_risk and self.initial_risk > 0): return 0.0
        if self.position.size > 0: return (self.d1.close[0] - ae) / self.initial_risk
        if self.position.size < 0: return (ae - self.d1.close[0]) / self.initial_risk
        return 0.0

    def _enough_history(self):
        if len(self.d5)  <= max(self.p.tf5m_breakout_period, self.p.donchian_trail_period) + 2: return False
        if len(self.d15) <= max(self.p.tf15m_adx_period, self.p.tf15m_ema_slow) + 2: return False
        if len(self.d60) <= self.p.tf60m_ema_slow + 2: return False
        if len(self.d1)  <= max(self.p.ema_trend, self.p.atr_period) + 2: return False
        return True

    def regime_ok_long(self):
        if not self.p.use_regime_long or self.p.regime_mode_long == 'off':
            return True
        if self.p.regime_mode_long == 'price_vs_slow':
            return self.d60.close[0] > self.ema60_slow[0]
        return self.ema60_fast[0] > self.ema60_slow[0]

    def regime_ok_short(self):
        if not self.p.can_short or not self.p.use_regime_short or self.p.regime_mode_short == 'off':
            return True
        if self.p.regime_mode_short == 'neutral':
            return True
        return self.ema60_fast[0] < self.ema60_slow[0]

    def trend_ok_long(self):
        if not self.p.use_trend_long:
            return True
        return (self.adx15[0] >= self.p.adxth and self.plusDI15[0] > self.minusDI15[0]
                and self.ema15_fast[0] > self.ema15_slow[0] and self.ema1_fast[0] > self.ema1_slow[0])

    def trend_ok_short(self):
        if not self.p.use_trend_short:
            return True
        return (self.adx15[0] >= self.p.adxth and self.minusDI15[0] > self.plusDI15[0]
                and self.ema15_fast[0] < self.ema15_slow[0] and self.ema1_fast[0] < self.ema1_slow[0])

    def breakout_up(self):
        if len(self.d5) < 2 or len(self.d1) < self.p.confirm_bars + 2: return False
        level = float(self.dc_high5[-1])
        confirmed = all(self.d1.close[-i] > level for i in range(self.p.confirm_bars, 0, -1))
        stretched = (self.d1.close[0] - level) > self.p.max_stretch_atr_mult * float(self.atr5[0])
        if (not confirmed) or stretched or (self.rsi1[0] >= self.p.rsi_overheat): return False
        if self.p.use_volume_filter and self.vsma1 is not None:
            if self.d1.volume[0] <= self.p.volume_filter_mult * max(self.vsma1[0], 1e-8): return False
        return True

    def breakdown_down(self):
        if len(self.d5) < 2 or len(self.d1) < self.p.confirm_bars + 2: return False
        level = float(self.dc_low5[-1])
        confirmed = all(self.d1.close[-i] < level for i in range(self.p.confirm_bars, 0, -1))
        stretched = (level - self.d1.close[0]) > self.p.max_stretch_atr_mult * float(self.atr5[0])
        if (not confirmed) or stretched or (self.rsi1[0] <= self.p.rsi_oversold): return False
        if self.p.use_volume_filter and self.vsma1 is not None:
            if self.d1.volume[0] <= self.p.volume_filter_mult * max(self.vsma1[0], 1e-8): return False
        return True

    def _update_trailing_stop(self):
        if not self.position:
            self.trail_stop=None; return
        if self.position.size > 0:
            self.run_high = max(self.run_high or self.d1.high[0], self.d1.high[0])
        else:
            self.run_low  = min(self.run_low  or self.d1.low[0],  self.d1.low[0])
        if (len(self) - self.last_trail_update) < self.p.trail_update_every:
            return

        candidate = None
        if self.p.trail_mode == "chandelier":
            candidate = float((self.run_high - self.p.trail_atr_mult * self.atr5[0]) if self.position.size>0
                              else (self.run_low + self.p.trail_atr_mult * self.atr5[0]))
        elif self.p.trail_mode == "ema_band":
            candidate = float((self.ema1_fast[0] - self.p.ema_band_mult * self.atr1[0]) if self.position.size>0
                              else (self.ema1_fast[0] + self.p.ema_band_mult * self.atr1[0]))
        elif self.p.trail_mode == "donchian":
            candidate = float(self.dc_exit5_low[0] if self.position.size>0 else self.dc_exit5_high[0])

        if candidate is not None:
            candidate = max(candidate, self.init_stop or -1e18) if self.position.size>0 else min(candidate, self.init_stop or 1e18)
            ae = self._avg_entry()
            if ae and self._R() >= self.p.move_to_breakeven_R:
                candidate = max(candidate, ae) if self.position.size>0 else min(candidate, ae)
            self.trail_stop = candidate if self.trail_stop is None else (
                max(self.trail_stop, candidate) if self.position.size>0 else min(self.trail_stop, candidate)
            )
            self.last_trail_update = len(self)

    def _stop_hit(self):
        if self.trail_stop is None: return False
        if self.p.close_based_stop:
            return (self.d1.close[0] <= self.trail_stop) if self.position.size>0 else (self.d1.close[0] >= self.trail_stop)
        else:
            return (self.d1.low[0]   <= self.trail_stop) if self.position.size>0 else (self.d1.high[0]  >= self.trail_stop)

    def _enter_long(self):
        entry = float(self._round_price(self.d1.close[0]))
        init_stop = self._round_price(entry - self.p.atr_stop_mult * float(self.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0: return
        tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
        self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=+1)))
        self.entry_prices.append(entry); self.sizes.append(size)
        self.buy(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop; self.trail_stop = init_stop
        self.initial_risk = max(1e-8, entry - init_stop)
        self.run_high = self.d1.high[0]; self.run_low=None
        self.entry_bar = len(self); self.n_adds=0; self.last_add_bar = len(self)

    def _enter_short(self):
        entry = float(self._round_price(self.d1.close[0]))
        init_stop = self._round_price(entry + self.p.atr_stop_mult * float(self.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0: return
        tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
        self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=-1)))
        self.entry_prices.append(entry); self.sizes.append(size)
        self.sell(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop; self.trail_stop = init_stop
        self.initial_risk = max(1e-8, init_stop - entry)
        self.run_low = self.d1.low[0]; self.run_high=None
        self.entry_bar = len(self); self.n_adds=0; self.last_add_bar = len(self)

    def _can_pyramid(self):
        if not (self.p.use_pyramiding and self.position): return False
        if self.n_adds >= self.p.max_adds: return False
        if (len(self) - self.last_add_bar) < self.p.add_cooldown: return False
        if self._R() < self.p.add_min_R: return False
        if self.position.size > 0:
            return self.d1.close[0] >= ((self.run_high or self.d1.high[0]) + self.p.add_atr_mult * float(self.atr5[0]))
        else:
            return self.d1.close[0] <= ((self.run_low  or self.d1.low[0])  - self.p.add_atr_mult * float(self.atr5[0]))

    def _do_pyramid(self):
        entry = float(self._round_price(self.d1.close[0]))
        stop = float(self.trail_stop or (entry - self.p.atr_stop_mult * float(self.atr5[0])) if self.position.size>0
                     else self.trail_stop or (entry + self.p.atr_stop_mult * float(self.atr5[0])))
        size = self._round_qty(self._risk_based_size(entry, stop) / 2.0)
        if size <= 0: return
        if self.position.size > 0:
            tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
            self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=+1)))
            self.entry_prices.append(entry); self.sizes.append(size)
            self.buy(size=size, exectype=bt.Order.Market)
        else:
            tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
            self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=-1)))
            self.entry_prices.append(entry); self.sizes.append(size)
            self.sell(size=size, exectype=bt.Order.Market)
        self.n_adds += 1; self.last_add_bar = len(self)

    def _exit_all(self, reason):
        qty = sum(l.size for l in self.active_orders) if self.active_orders else abs(self.position.size)
        if self.position.size > 0: self.sell(size=qty, exectype=bt.Order.Market)
        elif self.position.size < 0: self.buy(size=qty, exectype=bt.Order.Market)
        self.active_orders.clear(); self.entry_prices.clear(); self.sizes.clear()
        self.trail_stop=None; self.init_stop=None; self.initial_risk=None
        self.run_high=None; self.run_low=None; self.entry_bar=None
        self.n_adds=0; self.last_trail_update=-10**9; self.last_add_bar=-10**9
        self.last_exit_bar = len(self)

    def prenext(self):
        if not self._enough_history():
            return

    def next(self):
        if not self._enough_history():
            return
        if (len(self) - self.last_exit_bar) < self.p.reentry_cooldown_bars:
            pass

        if self.position:
            if self.p.max_bars_in_trade and self.entry_bar and (len(self) - self.entry_bar) >= self.p.max_bars_in_trade:
                self._exit_all(f"TimeStop {self.p.max_bars_in_trade}"); return
            self._update_trailing_stop()
            if self._stop_hit(): self._exit_all("TrailStop"); return

            price = self.d1.close[0]
            to_remove = []
            for idx, leg in enumerate(self.active_orders):
                if leg.dir > 0 and price >= leg.take_profit_price:
                    self.sell(size=leg.size, exectype=bt.Order.Market); to_remove.append(idx)
                elif leg.dir < 0 and price <= leg.take_profit_price:
                    self.buy(size=leg.size, exectype=bt.Order.Market);  to_remove.append(idx)
            for idx in reversed(to_remove):
                self.active_orders.pop(idx); self.entry_prices.pop(idx); self.sizes.pop(idx)
            if not self.active_orders: self._exit_all("All TPs"); return

            if self._can_pyramid(): self._do_pyramid()
            return

        # Flat → entries
        if self.p.use_htf:
            if not self.regime_ok_long(): self.block_counts['regime'] += 1
            elif not self.trend_ok_long(): self.block_counts['trend'] += 1
            elif not self.breakout_up(): self.block_counts['breakout'] += 1
            else:
                self._enter_long(); return

            if self.p.can_short:
                if not self.regime_ok_short(): self.block_counts['s_regime'] += 1
                elif not self.trend_ok_short(): self.block_counts['s_trend'] += 1
                elif not self.breakdown_down(): self.block_counts['breakdown'] += 1
                else:
                    self._enter_short(); return
        else:
            if self.ema1_fast[0] > self.ema1_slow[0] and self.breakout_up(): self._enter_long(); return
            if self.p.can_short and self.ema1_fast[0] < self.ema1_slow[0] and self.breakdown_down(): self._enter_short(); return

# --------------------- Feeds helper ---------------------
def add_mtf_feeds(cerebro: bt.Cerebro, base_feed, add_5m=True, add_15m=True, add_60m=True):
    if add_5m:
        cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=5,  name='5m',  boundoff=1)
    if add_15m:
        cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=15, name='15m', boundoff=1)
    if add_60m:
        cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=60, name='60m', boundoff=1)

# --------------------- Quick backtest ---------------------
class BuySellArrows(bt.observers.BuySell):
    def next(self):
        super().next()
        if self.lines.buy[0]:
            self.lines.buy[0] -= self.data.low[0] * 0.02
        if self.lines.sell[0]:
            self.lines.sell[0] += self.data.high[0] * 0.02
    plotlines = dict(
        buy=dict(marker='$\u21E7$', markersize=8.0),
        sell=dict(marker='$\u21E9$', markersize=8.0)
    )

def quick_backtest(
    df: pl.DataFrame,
    spec: DataSpec,
    params: Dict[str, Any],
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION_PER_TRANSACTION,
    exchange: Optional[str] = None,
    slippage_bps: float = 5.0,
    params_mode: str = "mtf",
    min_qty: float = 0.0, qty_step: float = 1.0, price_tick: Optional[float] = None,
    plot: bool = False,
    debug: bool = False,
):
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
    feed = make_feed_from_df(df, spec)
    cerebro.adddata(feed)
    add_mtf_feeds(cerebro, feed)

    sp = compat_params_to_mtf(params) if params_mode == "compat" else dict(params)
    can_short = (str(exchange).lower() == "mexc") if exchange else False
    sp.update(dict(can_short=can_short, min_qty=min_qty, qty_step=qty_step, price_tick=price_tick, debug=debug))

    cerebro.addstrategy(Enhanced_MACD_ADX3, backtest=True, **sp)
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    try: cerebro.broker.set_slippage_perc(perc=slippage_bps/10000.0)
    except Exception: pass

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    if debug: cerebro.addobserver(bt.observers.Trades)

    res = cerebro.run(maxcpus=1)
    strat = res[0]

    try: maxdd = float(strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0))
    except Exception: maxdd = 0.0
    try:
        sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr) if sr is not None else 0.0
    except Exception: sharpe = 0.0
    ta = strat.analyzers.trades.get_analysis()
    trades = ta.get('total', {}).get('total', 0) if ta else 0

    console.print(f"Trades={trades}, Sharpe={sharpe:.3f}, MaxDD={maxdd:.2f}%, Value={cerebro.broker.getvalue():.2f}")
    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')
    return dict(trades=trades, sharpe=sharpe, maxdd=maxdd, value=cerebro.broker.getvalue())

# --------------------- Single backtest eval (Optuna) ---------------------
def run_single_backtest_eval(
    strategy_class,
    df: pl.DataFrame,
    spec: DataSpec,
    init_cash: float,
    commission: float,
    params: Dict[str, Any],
    exchange: Optional[str] = None,
    slippage_bps: float = 5.0,
    min_qty: float = 0.0,
    qty_step: float = 1.0,
    price_tick: Optional[float] = None,
    params_mode: str = "compat",
    score_fn: Callable = score_sharpe_dd,
) -> Tuple[float, Dict[str, float], float]:
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
    feed = make_feed_from_df(df, spec)
    cerebro.adddata(feed)
    add_mtf_feeds(cerebro, feed)

    sp = compat_params_to_mtf(params) if params_mode == "compat" else dict(params)
    can_short = (str(exchange).lower() == "mexc") if exchange else False
    sp.update(dict(can_short=can_short, min_qty=min_qty, qty_step=qty_step, price_tick=price_tick))
    cerebro.addstrategy(strategy_class, backtest=True, **sp)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.broker.setcommission(commission=commission)
    try: cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception: pass
    cerebro.broker.setcash(init_cash)

    try:
        strats = cerebro.run(maxcpus=1)
        strat = strats[0]
    except Exception as e:
        console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]")
        return -999.0, {"error": 1.0}, 0.0

    score, metrics = score_fn(strat)
    final_value = cerebro.broker.getvalue()
    del cerebro, feed, strats, strat
    gc.collect()
    return score, metrics, final_value

# --------------------- Objective ---------------------
def make_objective(
    strategy_class,
    specs: List[DataSpec],
    df_map: Dict[str, pl.DataFrame],
    init_cash: float,
    commission: float,
    exchange: Optional[str] = None,
    slippage_bps: float = 5.0,
    min_qty: float = 0.0, qty_step: float = 1.0, price_tick: Optional[float] = None,
    param_mode: str = "mtf",
    scoring: str = "sharpe_dd",
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params_mtf(trial) if param_mode == "mtf" else suggest_params(trial)
        total_score, wsum, valid = 0.0, 0.0, 0
        weights = [1.0 + 0.1 * i for i in range(len(specs))]

        for idx, spec in enumerate(specs):
            df = df_map[spec.symbol]
            try:
                score, metrics, _ = run_single_backtest_eval(
                    strategy_class=strategy_class, df=df, spec=spec,
                    init_cash=init_cash, commission=commission, params=params,
                    exchange=exchange, slippage_bps=slippage_bps,
                    min_qty=min_qty, qty_step=qty_step, price_tick=price_tick,
                    params_mode=("mtf" if param_mode=="mtf" else "compat"),
                    score_fn=score_sharpe_dd
                )
                if score > -999:
                    w = weights[idx]
                    total_score += score * w
                    wsum += w
                    valid += 1
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

        if valid == 0:
            console.print(f"[red]Trial {trial.number}: No valid backtest results[/red]")
            return -999.0
        return total_score / max(wsum, 1e-9)
    return objective

# --------------------- Optimize (single-process) ---------------------
def make_pruner(num_steps: int, pruner: Optional[str], n_jobs: int):
    if pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max(1, num_steps), reduction_factor=3, bootstrap_count=max(2 * n_jobs, 8))
    elif pruner == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
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
    exchange: Optional[str] = None,
    param_mode: str = "mtf",
):
    storage = ensure_storage_or_sqlite(storage_string, study_name)
    console.print("[bold blue]Preloading data (cache-aware)...[/bold blue]")
    df_map = preload_polars(specs)
    console.print(f"[green]✓ Loaded {len(df_map)} assets[/green]")

    pruner_obj = make_pruner(num_steps=len(specs), pruner=pruner, n_jobs=n_jobs)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True, warn_independent_sampling=False)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner_obj, storage=storage, study_name=study_name, load_if_exists=True)

    objective = make_objective(
        strategy_class, specs, df_map, init_cash, commission,
        exchange=exchange, slippage_bps=5, min_qty=0.001, qty_step=0.001, price_tick=0.1,
        param_mode=param_mode, scoring="sharpe_dd"
    )

    console.print(f"Starting Optuna: trials={n_trials}, n_jobs={n_jobs}, pruner={pruner}, param_mode={param_mode}")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=True)
    console.print(f"[bold green]Done. Best value: {study.best_value:.4f}[/bold green]")

    table = Table(title="Best Parameters", show_lines=True)
    table.add_column("Parameter"); table.add_column("Value")
    for k, v in study.best_params.items():
        table.add_row(k, str(v))
    console.print(table)

    if hasattr(study.best_trial, 'user_attrs'):
        console.print("\n[bold cyan]Best Trial Performance:[/bold cyan]")
        for attr, value in study.best_trial.user_attrs.items():
            if "_score" in attr or "_trades" in attr or "_sharpe" in attr or "_mdd" in attr:
                console.print(f"{attr}: {value}")

    return study, study.best_params

# --------------------- Multi-process launcher ---------------------
def _worker_optimize(
    worker_id: int,
    strategy_class,
    specs: List[DataSpec],
    storage_string: Optional[str],
    study_name: str,
    trials_for_worker: int,
    init_cash: float,
    commission: float,
    exchange: Optional[str],
    param_mode: str,
    seed_base: int,
):
    try:
        optimize(
            strategy_class=strategy_class,
            specs=specs,
            n_trials=trials_for_worker,
            n_jobs=1,  # critical for true parallel CPU via processes
            init_cash=init_cash,
            commission=commission,
            pruner="hyperband",
            storage_string=storage_string,
            study_name=study_name,
            seed=seed_base + worker_id,
            exchange=exchange,
            param_mode=param_mode,
        )
    except Exception as e:
        console.print(f"[red]Worker {worker_id} crashed: {e}[/red]")

def launch_multiprocess_optimize(
    strategy_class,
    specs: List[DataSpec],
    storage_string: Optional[str],
    study_name: str,
    total_trials: int,
    workers: int,
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION_PER_TRANSACTION,
    exchange: Optional[str] = None,
    param_mode: str = "mtf",
    seed_base: int = 42,
):
    if workers < 1:
        workers = 1
    trials_per_worker = math.ceil(total_trials / workers)

    console.print(f"[bold magenta]Launching {workers} workers @ {trials_per_worker} trials each (total≈{trials_per_worker*workers})[/bold magenta]")
    preload_polars(specs, force_refresh=False)  # ensure cache is built before forking

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    procs: List[mp.Process] = []
    for wid in range(workers):
        p = mp.Process(
            target=_worker_optimize,
            args=(wid, strategy_class, specs, storage_string, study_name,
                  trials_per_worker, init_cash, commission, exchange, param_mode, seed_base)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    storage = ensure_storage_or_sqlite(storage_string, study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    console.print(f"[green]Multiprocess optimize done. Best value: {study.best_value:.4f}[/green]")
    console.print(study.best_params)

# --------------------- Example usage (programmatic) ---------------------
if __name__ == "__main__":
    # Build cache, quick sanity backtest, then multi-process optimize, then holdout
    bull_start = "2020-09-28"; bull_end = "2021-05-31"
    bear_start = "2022-05-28"; bear_end = "2023-06-23"

    # 1) Training universe
    specs = [
        DataSpec("BTC", interval="1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        DataSpec("ETH", interval="1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
    ]

    # 2) Build cache once
    build_cache(specs, force_refresh=False)

    # 3) Quick sanity backtest on a short window
    sanity_spec = DataSpec("BTC", interval="1m", ranges=[("2024-01-01","2024-01-15")])
    df_map = preload_polars([sanity_spec])
    sanity_params = dict(
        use_regime_long=True, regime_mode_long='price_vs_slow',
        tf60m_ema_fast=20, tf60m_ema_slow=60,
        tf5m_breakout_period=20, confirm_bars=1, adxth=18,
        atr_stop_mult=2.5, trail_mode='chandelier', trail_atr_mult=4.0,
        move_to_breakeven_R=0.5, risk_per_trade_pct=0.002,
        ema_fast=20, ema_slow=50, ema_trend=200, atr_period=14,
        rsi_overheat=80, use_pyramiding=False, use_volume_filter=False,
    )
    console.print("[yellow]Quick sanity backtest (BTC 2024-01-01..2024-01-15)...[/yellow]")
    quick_backtest(df_map["BTC"], sanity_spec, sanity_params, exchange="MEXC", params_mode="mtf", plot=False, debug=False)

    # 4) Multi-process optimization (uses SQLite by default; pass ODBC if you want/have MSSQL)
    study_name = "Optimized_1m_MTF_MACD_ADX"
    launch_multiprocess_optimize(
        strategy_class=Enhanced_MACD_ADX3,
        specs=specs,
        storage_string=MSSQL_ODBC,   # None => SQLite fallback; pass your ODBC if you prefer MSSQL
        study_name=study_name,
        total_trials=200,
        workers=8,
        init_cash=INIT_CASH,
        commission=COMMISSION_PER_TRANSACTION,
        exchange="MEXC",
        param_mode="mtf",
        seed_base=42,
    )

    # 5) Holdout with best params
    storage = ensure_storage_or_sqlite(None, study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_params = study.best_params
    holdout_spec = DataSpec("BTC", interval="1m", ranges=[("2023-06-12", "2025-05-31")])
    df_hold = preload_polars([holdout_spec])["BTC"]
    console.print("[magenta]Holdout backtest with best params...[/magenta]")
    quick_backtest(df_hold, holdout_spec, best_params, exchange="MEXC", params_mode="mtf", plot=False, debug=False)