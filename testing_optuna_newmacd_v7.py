# optuna_hyperopt_polars_vec_fixed.py  (patched)
import os
import gc
import math
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable

import backtrader as bt
import optuna
import polars as pl
import pandas as pd
from rich.console import Console
from rich.table import Table
from pathlib import Path
import multiprocessing as mp
import hashlib
import numpy as np

console = Console()

# Your DB accessors (expected in your environment)
from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData  # keep as-is if present
from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC

# --- Globals & Configuration ---
INIT_CASH = 1000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"
STUDYNAME = "Optimized_1m_MTF_VEC_V7"

# Parquet cache dir (override with env BTQ_CACHE_DIR)
CACHE_DIR = Path(os.getenv("BTQ_CACHE_DIR", ".btq_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Exchange metadata for realistic sizing
SYMBOL_META = {
    "BTC": dict(qty_step=0.001, price_tick=0.1, min_qty=0.0),
    "ETH": dict(qty_step=0.001, price_tick=0.01, min_qty=0.0),
    "BNB": dict(qty_step=0.01, price_tick=0.01, min_qty=0.0),
    "PENDLE": dict(qty_step=0.1, price_tick=0.001, min_qty=0.0),
}
def meta_for(symbol: str):
    return SYMBOL_META.get(symbol.upper(), dict(qty_step=0.001, price_tick=0.001, min_qty=0.0))

# --- Data Specification ---
@dataclass(frozen=True)
class DataSpec:
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL

def expand_specs_by_ranges(specs: List[DataSpec]) -> List[DataSpec]:
    expanded: List[DataSpec] = []
    for s in specs:
        if s.ranges:
            for rs, re in s.ranges:
                expanded.append(DataSpec(
                    symbol=s.symbol, interval=s.interval,
                    start_date=rs, end_date=re, ranges=None, collateral=s.collateral
                ))
        else:
            expanded.append(s)
    return expanded
    
# --- Optuna Storage (with SQLite fallback) ---
def mssql_url_from_odbc(odbc: str, db_override: Optional[str] = None) -> str:
    parts = {k.strip().upper(): v.strip() for chunk in odbc.split(";") if "=" in chunk and chunk.strip() for k, v in [chunk.split("=", 1)]}
    server = parts.get("SERVER", "localhost")
    db = db_override or parts.get("DATABASE", "OptunaBT")
    uid = parts.get("UID", "SA")
    pwd = parts.get("PWD", "")
    driver = urllib.parse.quote_plus(parts.get("DRIVER", "{ODBC Driver 18 for SQL Server}").strip("{}"))
    query = f"driver={driver}&Encrypt=yes&TrustServerCertificate={parts.get('TRUSTSERVERCERTIFICATE', 'yes')}"
    return f"mssql+pyodbc://{uid}:{urllib.parse.quote_plus(pwd)}@{server}/{db}?{query}"

def ensure_storage_or_sqlite(storage_string: Optional[str], study_name: str) -> optuna.storages.RDBStorage:
    if storage_string is None:
        sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
        console.print(f"[yellow]No storage provided → using SQLite at {sqlite_path}[/yellow]")
        return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")

    if not storage_string.startswith("mssql+pyodbc://"):
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

# --- Data Loading & Caching ---
class PolarsDataLoader:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, symbol: str, interval: str, collateral: str, start: str, end: str) -> str:
        raw = f"{symbol}|{interval}|{collateral}|{start}:{end}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{symbol}_{interval}_{collateral}_{h}.parquet"

    def load_union_by_symbol(self, symbol: str, interval: str, collateral: str, pairs: List[Tuple[str,str]], use_cache=True) -> pl.DataFrame:
        # compute overall start/end for cache granularity
        start, end = min(s for s,_ in pairs), max(e for _,e in pairs)
        path = self.cache_dir / self._cache_key(symbol, interval, collateral, start, end)
        
        if use_cache and path.exists():
            try:
                console.print(f"[cyan]Loaded from cache: {symbol} ({start}..{end})[/cyan]")
                return pl.scan_parquet(str(path)).collect()
            except Exception as e:
                console.print(f"[yellow]Cache read failed {path.name}: {e}[/yellow]")
        
        df = get_database_data(ticker=symbol, start_date=start, end_date=end, time_resolution=interval, pair=collateral)
        if df is None:
            raise ValueError(f"No data returned from DB for {symbol} {interval} {start}->{end}")
        # robust check for emptiness across polars versions
        try:
            empty = (hasattr(df, "is_empty") and df.is_empty()) or (hasattr(df, "height") and df.height == 0)
        except Exception:
            empty = False
        if empty:
            raise ValueError(f"No data for {symbol} {interval} {start}->{end}")
        
        df = df.sort("TimestampStart")
        if use_cache:
            try:
                df.write_parquet(str(path), compression="zstd", statistics=True)
                console.print(f"[green]Cached -> {path}[/green]")
            except Exception as e:
                console.print(f"[yellow]Cache write failed: {e}[/yellow]")
        return df

def preload_polars(specs: List[DataSpec], force_refresh=False) -> Dict[str, pl.DataFrame]:
    loader = PolarsDataLoader()
    by_symbol = {}
    for s in specs:
        by_symbol.setdefault(s.symbol, []).extend(s.ranges or [(s.start_date or "", s.end_date or "")])

    df_map = {}
    for symbol, pairs in by_symbol.items():
        one = next(x for x in specs if x.symbol == symbol)
        df = loader.load_union_by_symbol(symbol, one.interval, one.collateral, pairs, use_cache=not force_refresh)
        df_map[symbol] = df
    return df_map

def slice_df_to_spec(df: pl.DataFrame, spec: DataSpec) -> pl.DataFrame:
    """
    Safely slice a Polars DataFrame to the spec's start/end window.
    Ensures TimestampStart is a Datetime type (tries Polars parsing; falls back to pandas).
    Returns an empty DataFrame on failure.
    """
    if df is None:
        return pl.DataFrame()

    # If TimestampStart isn't present, return empty
    if "TimestampStart" not in df.columns:
        return pl.DataFrame()

    # Try to ensure TimestampStart is a proper Datetime dtype in Polars
    try:
        # If it's already datetime type this is a no-op; otherwise try string->datetime
        # Use .str.to_datetime() which is widely supported across polars versions
        df = df.with_columns([
            pl.when(pl.col("TimestampStart").dtype == pl.Utf8)
              .then(pl.col("TimestampStart").str.to_datetime())
              .otherwise(pl.col("TimestampStart"))
              .alias("TimestampStart")
        ])
    except Exception:
        # Best-effort: try a simple cast, then fall back to pandas parsing below
        try:
            df = df.with_columns([pl.col("TimestampStart").cast(pl.Datetime).alias("TimestampStart")])
        except Exception:
            pass

    # Build lazy frame and filter using lit(...).str.to_datetime() for comparisons
    lf = df.lazy()
    try:
        if spec.start_date:
            start_expr = pl.lit(spec.start_date).str.to_datetime()
            lf = lf.filter(pl.col("TimestampStart") >= start_expr)
        if spec.end_date:
            end_expr = pl.lit(spec.end_date).str.to_datetime()
            lf = lf.filter(pl.col("TimestampStart") <= end_expr)
        res = lf.collect()
        return res
    except Exception:
        # Polars parse/filter failed for some reason — fall back to pandas parsing
        try:
            pdf = df.to_pandas()
            pdf["TimestampStart"] = pd.to_datetime(pdf["TimestampStart"], utc=True, errors="coerce")
            if spec.start_date:
                s = pd.to_datetime(spec.start_date, utc=True, errors="coerce")
                pdf = pdf[pdf["TimestampStart"] >= s]
            if spec.end_date:
                e = pd.to_datetime(spec.end_date, utc=True, errors="coerce")
                pdf = pdf[pdf["TimestampStart"] <= e]
            return pl.from_pandas(pdf)
        except Exception:
            return pl.DataFrame()

def build_cache(specs: List[DataSpec], force_refresh=True):
    console.print(f"[bold blue]Building cache for {len(set(s.symbol for s in specs))} symbols...[/bold blue]")
    t0 = time.time()
    preload_polars(specs, force_refresh=force_refresh)
    console.print(f"[green]Cache build complete in {time.time()-t0:.1f}s[/green]")

# --- Polars Feature Engineering ---
def ema(expr: pl.Expr, n: int) -> pl.Expr: return expr.ewm_mean(alpha=2.0/(n+1.0), adjust=False)
def wilder(expr: pl.Expr, n: int) -> pl.Expr: return expr.ewm_mean(alpha=1.0/max(1,n), adjust=False)

def add_atr(lf: pl.LazyFrame, n: int, prefix: str) -> pl.LazyFrame:
    tr = pl.max_horizontal((pl.col("High")-pl.col("Low")), (pl.col("High")-pl.col("Close").shift(1)).abs(), (pl.col("Low")-pl.col("Close").shift(1)).abs())
    return lf.with_columns(wilder(tr, n).alias(f"{prefix}_atr"))

def add_rsi(lf: pl.LazyFrame, n: int = 14, col: str = "Close") -> pl.LazyFrame:
    diff = pl.col("Close") - pl.col("Close").shift(1)
    up = pl.when(diff > 0).then(diff).otherwise(0.0)
    down = pl.when(diff < 0).then(-diff).otherwise(0.0)
    lf = lf.with_columns(wilder(up, n).alias("_up"), wilder(down, n).alias("_down"))
    lf = lf.with_columns((pl.col("_up") / (pl.col("_down") + 1e-12)).alias("_rs"))
    lf = lf.with_columns((100.0 - (100.0 / (1.0 + pl.col("_rs")))).alias("rsi"))
    return lf.drop(["_up", "_down", "_rs"])

def add_adx(lf: pl.LazyFrame, n: int) -> pl.LazyFrame:
    up = pl.col("High") - pl.col("High").shift(1)
    down = pl.col("Low").shift(1) - pl.col("Low")
    pdm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    mdm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    tr = pl.max_horizontal(
        pl.col("High") - pl.col("Low"),
        (pl.col("High") - pl.col("Close").shift(1)).abs(),
        (pl.col("Low") - pl.col("Close").shift(1)).abs()
    )
    lf = lf.with_columns(
        wilder(tr, n).alias("_trn"),
        wilder(pdm, n).alias("_pdmn"),
        wilder(mdm, n).alias("_mdmn")
    )
    lf = lf.with_columns(
        (100 * pl.col("_pdmn") / (pl.col("_trn") + 1e-12)).alias("plus_di"),
        (100 * pl.col("_mdmn") / (pl.col("_trn") + 1e-12)).alias("minus_di")
    )
    lf = lf.with_columns(
        (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() / (pl.col("plus_di") + pl.col("minus_di") + 1e-12)).alias("_dx")
    )
    lf = lf.with_columns(wilder(pl.col("_dx"), n).alias("adx"))
    return lf.drop(["_trn", "_pdmn", "_mdmn", "_dx"])

def resample(lf: pl.LazyFrame, every: str) -> pl.LazyFrame:
    # group_by_dynamic expects the timestamp column to be a Datetime type
    return lf.group_by_dynamic("TimestampStart", every=every, period=every, closed="right") \
             .agg(pl.first("Open").alias("Open"), pl.max("High").alias("High"), pl.min("Low").alias("Low"), pl.last("Close").alias("Close"), pl.sum("Volume").alias("Volume")) \
             .sort("TimestampStart")

def build_feature_df(df_base: pl.DataFrame, params: Dict[str, Any], base_tf: str) -> pl.DataFrame:
    defaults = dict(
        ema_fast=20, ema_slow=50, ema_trend=200,
        atr_period=14, rsi_overheat=75, rsi_oversold=25,
        adxth=20, confirm_bars=2,
        max_stretch_atr_mult=1.0, ema_band_mult=2.0,
        donchian_trail_period=55, close_based_stop=True,
        trail_update_every=2, max_bars_in_trade=360,
        use_pyramiding=False, max_adds=5, add_cooldown=20,
        add_atr_mult=1.0, add_min_R=1.0,
        use_volume_filter=False, volume_filter_mult=1.2,
        regime_mode_long="ema", regime_mode_short="neutral",
        use_htf=True,
        breakout_period=40, htf1_adx_period=14,
        htf1_ema_fast=40, htf1_ema_slow=120,
        htf2_ema_fast=80, htf2_ema_slow=200
    )
    p = {**defaults, **params}

    lf = df_base.lazy().sort("TimestampStart")

    # base features
    lf_b = lf.with_columns(
        ema(pl.col("Close"), p["ema_fast"]).alias("ema_base_fast"),
        ema(pl.col("Close"), p["ema_slow"]).alias("ema_base_slow"),
        pl.col("Volume").rolling_mean(20, min_samples=1).alias("vsma_base"),
    )
    lf_b = add_atr(lf_b, p["atr_period"], "atr_base")
    lf_b = add_rsi(lf_b, 14)

    # timeframe mapping
    if base_tf == "15m":
        brk_tf, htf1, htf2 = "60m", "60m", "240m"
    elif base_tf == "1m":
        brk_tf, htf1, htf2 = "5m", "15m", "60m"
    else:
        brk_tf, htf1, htf2 = "60m", "60m", "240m"

    # breakout features
    lf_brk = resample(lf, brk_tf)
    lf_brk = add_atr(lf_brk, p["atr_period"], "atr_brk")
    lf_brk = lf_brk.with_columns(
        pl.col("High").rolling_max(p["breakout_period"]).shift(1).alias("dc_high_prev"),
        pl.col("Low").rolling_min(p["breakout_period"]).shift(1).alias("dc_low_prev"),
        pl.col("Low").rolling_min(p["donchian_trail_period"]).alias("dc_exit_low"),
        pl.col("High").rolling_max(p["donchian_trail_period"]).alias("dc_exit_high"),
    )

    # HTFs (optional)
    if p.get("use_htf", True):
        lf_htf1 = resample(lf, htf1).with_columns(
            ema(pl.col("Close"), p["htf1_ema_fast"]).alias("ema_htf1_fast"),
            ema(pl.col("Close"), p["htf1_ema_slow"]).alias("ema_htf1_slow"),
        )
        lf_htf1 = add_adx(lf_htf1, p["htf1_adx_period"])

        lf_htf2 = resample(lf, htf2).with_columns(
            ema(pl.col("Close"), p["htf2_ema_fast"]).alias("ema_htf2_fast"),
            ema(pl.col("Close"), p["htf2_ema_slow"]).alias("ema_htf2_slow"),
        )
    else:
        lf_htf1 = lf.select("TimestampStart")
        lf_htf2 = lf.select("TimestampStart")

    # Always compute base DMI/ADX as a fallback
    lf_dmi_base = add_adx(lf, p["htf1_adx_period"]).select([
        "TimestampStart",
        pl.col("plus_di").alias("plus_di_base"),
        pl.col("minus_di").alias("minus_di_base"),
        pl.col("adx").alias("adx_base"),
    ])

    # join features
    sel_brk = lf_brk.select([c for c in ["TimestampStart", "atr_brk_atr", "dc_high_prev", "dc_low_prev", "dc_exit_low", "dc_exit_high"] if c in lf_brk.collect_schema().names()])
    sel_htf1 = lf_htf1.select([c for c in ["TimestampStart", "plus_di", "minus_di", "adx", "ema_htf1_fast", "ema_htf1_slow"] if c in lf_htf1.collect_schema().names()])
    sel_htf2 = lf_htf2.select([c for c in ["TimestampStart", "ema_htf2_fast", "ema_htf2_slow"] if c in lf_htf2.collect_schema().names()])

    lf_all = (
        lf_b.join_asof(sel_brk, on="TimestampStart", strategy="backward")
            .join_asof(sel_htf1, on="TimestampStart", strategy="backward")
            .join_asof(sel_htf2, on="TimestampStart", strategy="backward")
            .join_asof(lf_dmi_base, on="TimestampStart", strategy="backward")
    )

    # Ensure required columns exist for downstream expressions:
    names = set(lf_all.collect_schema().names())
    add_cols = []
    if "ema_htf1_fast" not in names:
        add_cols.append(pl.col("ema_base_fast").alias("ema_htf1_fast"))
    if "ema_htf1_slow" not in names:
        add_cols.append(pl.col("ema_base_slow").alias("ema_htf1_slow"))
    if "ema_htf2_fast" not in names:
        add_cols.append(pl.col("ema_base_fast").alias("ema_htf2_fast"))
    if "ema_htf2_slow" not in names:
        add_cols.append(pl.col("ema_base_slow").alias("ema_htf2_slow"))
    if "plus_di" not in names:
        add_cols.append(pl.col("plus_di_base").alias("plus_di"))
    if "minus_di" not in names:
        add_cols.append(pl.col("minus_di_base").alias("minus_di"))
    if "adx" not in names:
        add_cols.append(pl.col("adx_base").alias("adx"))
    if add_cols:
        lf_all = lf_all.with_columns(add_cols)

    # basic breakout booleans
    cb = int(p["confirm_bars"])
    lf_all = lf_all.with_columns(
        (pl.col("Close").rolling_min(cb, min_samples=1) > pl.col("dc_high_prev")).alias("breakout_up"),
        (pl.col("Close").rolling_max(cb, min_samples=1) < pl.col("dc_low_prev")).alias("breakdown_down"),
    )

    # regimes
    if p["regime_mode_long"] == "price_vs_slow":
        long_regime = pl.col("Close") > pl.col("ema_htf2_slow")
    elif p["regime_mode_long"] == "off":
        long_regime = pl.lit(True)
    else:
        long_regime = pl.col("ema_htf2_fast") > pl.col("ema_htf2_slow")

    if p["regime_mode_short"] == "off":
        short_regime = pl.lit(True)
    elif p["regime_mode_short"] == "ema":
        short_regime = pl.col("ema_htf2_fast") < pl.col("ema_htf2_slow")
    else:
        short_regime = pl.lit(True)

    # volume filter
    vol_ok = pl.when(pl.lit(p["use_volume_filter"])).then(
        pl.col("Volume") > p["volume_filter_mult"] * pl.col("vsma_base")
    ).otherwise(True)

    # entry signals (safe now that columns exist)
    lf_all = lf_all.with_columns(
        (
            long_regime & (pl.col("adx") >= p["adxth"]) & (pl.col("plus_di") > pl.col("minus_di")) &
            (pl.col("ema_htf1_fast") > pl.col("ema_htf1_slow")) & (pl.col("ema_base_fast") > pl.col("ema_base_slow")) &
            pl.col("breakout_up") &
            ((pl.col("Close") - pl.col("dc_high_prev")) <= p["max_stretch_atr_mult"] * pl.col("atr_brk_atr")) &
            vol_ok & (pl.col("rsi") < p["rsi_overheat"])
        ).alias("long_entry_signal"),
        (
            short_regime & (pl.col("adx") >= p["adxth"]) & (pl.col("minus_di") > pl.col("plus_di")) &
            (pl.col("ema_htf1_fast") < pl.col("ema_htf1_slow")) & (pl.col("ema_base_fast") < pl.col("ema_base_slow")) &
            pl.col("breakdown_down") &
            ((pl.col("dc_low_prev") - pl.col("Close")) <= p["max_stretch_atr_mult"] * pl.col("atr_brk_atr")) &
            vol_ok & (pl.col("rsi") > p["rsi_oversold"])
        ).alias("short_entry_signal"),
    )

    df = lf_all.collect()

    # normalize atr names
    df = df.rename({c: c.replace("_atr", "") for c in df.columns if c.endswith("_atr")})

    required = ["atr_brk", "dc_high_prev", "adx", "ema_htf2_fast", "ema_base_fast", "atr_base", "rsi", "vsma_base"]
    req_present = [c for c in required if c in df.columns]
    if req_present:
        df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in req_present]))

    keep = [
        "TimestampStart", "Open", "High", "Low", "Close", "Volume",
        "ema_base_fast", "ema_base_slow", "atr_base", "rsi", "vsma_base",
        "atr_brk", "dc_high_prev", "dc_low_prev", "dc_exit_low", "dc_exit_high",
        "ema_htf1_fast", "ema_htf1_slow", "plus_di", "minus_di", "adx",
        "ema_htf2_fast", "ema_htf2_slow", "long_entry_signal", "short_entry_signal",
    ]
    return df.select([c for c in keep if c in df.columns])

# --- Backtrader Components ---
# FeatureData: map polars->feed columns, assume you have PolarsData feed in your Backtrader
class FeatureData(bt.feeds.PolarsData):
    # Note: using PandasData because not all Backtrader installs have PolarsData helper
    # lines (extra indicators) will be inferred from params map
    lines = (
        'ema_base_fast','ema_base_slow','atr_base','rsi','vsma_base',
        'atr_brk','dc_high_prev','dc_low_prev','dc_exit_low','dc_exit_high',
        'ema_htf1_fast','ema_htf1_slow','plus_di','minus_di','adx',
        'ema_htf2_fast','ema_htf2_slow','long_entry_signal','short_entry_signal'
    )
    params = dict(datetime='TimestampStart',open='Open',high='High',low='Low',close='Close',volume='Volume',openinterest=-1, **{k:k for k in lines})

def make_feature_feed(df_feat: pl.DataFrame, name: str) -> FeatureData:
    # convert to pandas; ensure datetime is datetime64[ns]
    pdf = df_feat.to_pandas()
    if "TimestampStart" in pdf.columns:
        try:
            pdf["TimestampStart"] = pd.to_datetime(pdf["TimestampStart"], utc=True)
        except Exception:
            pdf["TimestampStart"] = pd.to_datetime(pdf["TimestampStart"], errors='coerce')
    feed = FeatureData(dataname=pdf)
    feed._name = f"{name}-features"
    return feed

class VectorMACD_ADX(bt.Strategy):
    params = (
        # position sizing / risk
        ('risk_per_trade_pct', 0.005),
        ('exposure_pct_cap', 0.95),
        ('reserve_cash_pct', 0.02),
        ('slip_fee_buffer_bps', 10.0),
        ('max_leverage', 1.0),

        # exits / trailing
        ('use_dynamic_exits', True),
        ('tp_r_multiple', 2.0),
        ('use_partial_exits', True),
        ('partial_exit_r', 1.5),
        ('partial_exit_pct', 0.5),
        ('time_limit_bars', 120),
        ('atr_stop_mult', 2.5),
        ('use_trailing_stop', True),
        ('trail_mode', 'chandelier'),
        ('trail_atr_mult', 4.0),
        ('move_to_breakeven_R', 1.0),

        # execution / position management
        ('reentry_cooldown_bars', 5),
        ('min_qty', 0.0),
        ('qty_step', 0.001),
        ('price_tick', 0.1),
        ('round_prices', True),
        ('can_short', False),

        # indicators / signals (base)
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('ema_trend', 200),
        ('atr_period', 14),
        ('rsi_overheat', 75),
        ('rsi_oversold', 25),
        ('adxth', 20),
        ('confirm_bars', 2),
        ('max_stretch_atr_mult', 1.0),
        ('ema_band_mult', 2.0),
        ('donchian_trail_period', 55),
        ('close_based_stop', True),
        ('trail_update_every', 2),
        ('max_bars_in_trade', 360),
        ('use_pyramiding', False),
        ('max_adds', 5),
        ('add_cooldown', 20),
        ('add_atr_mult', 1.0),
        ('add_min_R', 1.0),

        # volume filter
        ('use_volume_filter', False),
        ('volume_filter_mult', 1.2),

        # legacy / flags
        ('backtest', True),
        ('debug', False),

        # regime
        ('regime_mode_long', 'ema'),
        ('regime_mode_short', 'neutral'),

        # enable HTF usage
        ('use_htf', True),

        # fallback TP if dynamic exits disabled
        ('take_profit', 4.0),

        # --- New params that Optuna may pass ---
        ('breakout_period', 40),
        ('htf1_adx_period', 14),
        ('htf1_ema_fast', 40),
        ('htf1_ema_slow', 120),
        ('htf2_ema_fast', 80),
        ('htf2_ema_slow', 200),

        # compatibility names
        ('tf5m_breakout_period', 55),
        ('tf15m_adx_period', 14),
        ('tf15m_ema_fast', 50),
        ('tf15m_ema_slow', 200),
        ('tf60m_ema_fast', 50),
        ('tf60m_ema_slow', 200),
    )

    def __init__(self):
        self.d = self.datas[0]
        self._reset_position_state()
        self.active_orders = []

    def _reset_position_state(self):
        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.partial_exit_done = False
        self.trail_stop_active = False

    def _round_qty(self, s):
        step = self.p.qty_step if self.p.qty_step > 0 else 0.001
        q = math.floor(max(0.0, s) / step) * step
        if q == 0.0 and s > 0:
            q = step
        return 0.0 if (self.p.min_qty > 0 and q < self.p.min_qty) else q

    def _round_price(self, p):
        return round(p / self.p.price_tick) * self.p.price_tick if (self.p.round_prices and self.p.price_tick > 0) else p

    def _risk_based_size(self, entry_price, stop_price):
        eq = float(self.broker.getvalue())
        cash = float(self.broker.getcash())

        dist = abs(entry_price - stop_price)
        if dist <= 0 or math.isclose(dist, 0.0):
            return 0.0

        risk_dollars = max(0.0, eq * self.p.risk_per_trade_pct)
        raw_units = risk_dollars / dist

        buffer_mult = 1.0 + (self.p.slip_fee_buffer_bps / 10000.0)
        max_cash_for_position = max(
            0.0,
            (cash * self.p.exposure_pct_cap) - (eq * self.p.reserve_cash_pct)
        )
        cash_cap_units = (max_cash_for_position / (entry_price * buffer_mult)) if entry_price > 0 else 0.0

        size = min(raw_units, cash_cap_units)
        return self._round_qty(size)

    def _enter(self, direction: int):
        entry = self._round_price(self.d.close[0])

        # defensive ATR read (prefer atr_brk then atr_base)
        atr_value = None
        try:
            atr_value = float(getattr(self.d, 'atr_brk', [None])[0])
        except Exception:
            atr_value = None
        if atr_value is None:
            try:
                atr_value = float(getattr(self.d, 'atr_base', [None])[0])
            except Exception:
                atr_value = None
        if atr_value is None:
            return

        stop_dist = self.p.atr_stop_mult * atr_value
        init_stop = self._round_price(entry - stop_dist * direction)

        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            return

        self.initial_risk = abs(entry - init_stop)

        if self.p.use_dynamic_exits:
            self.take_profit_price = self._round_price(entry + self.p.tp_r_multiple * self.initial_risk * direction)
            self.partial_tp_price = self._round_price(entry + self.p.partial_exit_r * self.initial_risk * direction)
        else:
            self.take_profit_price = self._round_price(entry * (1 + (getattr(self.p, "take_profit", 4.0) / 100.0) * direction))
            self.partial_tp_price = None

        if direction > 0:
            o = self.buy(size=size)
            self.run_high = self.d.high[0]
        else:
            o = self.sell(size=size)
            self.run_low = self.d.low[0]

        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.entry_bar = len(self)
        self.active_orders.append(o)

    def notify_order(self, o):
        if o.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return

        if o.status == bt.Order.Completed:
            # Mark entry bar and run_high/run_low on execution confirmation
            # so entry_bar is always defined when we actually have a position.
            try:
                # len(self) returns current bar index
                self.entry_bar = len(self)
            except Exception:
                self.entry_bar = None

            if o.isbuy():
                # initialize run_high on confirmed buy
                try:
                    self.run_high = float(self.d.high[0])
                except Exception:
                    self.run_high = None
                if self.p.debug:
                    console.log(f'BUY EXECUTED, Price: {o.executed.price:.2f}, Cost: {o.executed.value:.2f}, Comm: {o.executed.comm:.2f}')
            elif o.issell():
                # initialize run_low on confirmed sell
                try:
                    self.run_low = float(self.d.low[0])
                except Exception:
                    self.run_low = None
                if self.p.debug:
                    console.log(f'SELL EXECUTED, Price: {o.executed.price:.2f}, Cost: {o.executed.value:.2f}, Comm: {o.executed.comm:.2f}')

        elif o.status in [bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected]:
            try:
                status_name = bt.Order.Status[o.status]
            except Exception:
                status_name = str(o.status)
            console.print(f'Order {status_name}')

        if o in self.active_orders:
            self.active_orders.remove(o)


    def _R(self):
        if not self.position:
            return 0.0
        current_price = self.d.close[0]
        direction = 1 if self.position.size > 0 else -1
        return (current_price - self.init_stop) * direction / (self.initial_risk if self.initial_risk else 1e-9)

    def _update_trailing_stop(self):
        if not self.position:
            self.trail_stop = None
            return
        if self.position.size > 0:
            self.run_high = max(self.run_high or self.d.high[0], self.d.high[0])
        else:
            self.run_low = min(self.run_low or self.d.low[0], self.d.low[0])

        # defensive ATR read
        atr_value = None
        try:
            atr_value = float(getattr(self.d, 'atr_brk', [None])[0])
        except Exception:
            atr_value = None
        if atr_value is None:
            try:
                atr_value = float(getattr(self.d, 'atr_base', [None])[0])
            except Exception:
                atr_value = None
        if atr_value is None:
            return

        cand = None
        if self.p.trail_mode == "chandelier":
            cand = (self.run_high - self.p.trail_atr_mult * atr_value) if self.position.size > 0 else (self.run_low + self.p.trail_atr_mult * atr_value)
        elif self.p.trail_mode == "donchian":
            cand = getattr(self.d, 'dc_exit_low', [None])[0] if self.position.size > 0 else getattr(self.d, 'dc_exit_high', [None])[0]

        if cand is not None:
            if self.position.size > 0:
                cand = max(cand, self.init_stop or -1e18)
            else:
                cand = min(cand, self.init_stop or 1e18)
            self.trail_stop = cand if self.trail_stop is None else (max(self.trail_stop, cand) if self.position.size > 0 else min(self.trail_stop, cand))

    def _stop_hit(self):
        if self.trail_stop is None:
            return False
        if self.position.size > 0:
            return self.d.close[0] <= self.trail_stop if self.p.close_based_stop else self.d.low[0] <= self.trail_stop
        else:
            return self.d.close[0] >= self.trail_stop if self.p.close_based_stop else self.d.high[0] >= self.trail_stop

    def next(self):
        if not self.position:
            if (len(self) - getattr(self, 'last_exit_bar', -1e9)) < self.p.reentry_cooldown_bars:
                return
            long_sig = getattr(self.d, 'long_entry_signal', None)
            short_sig = getattr(self.d, 'short_entry_signal', None)
            if long_sig is not None and self.d.long_entry_signal[0]:
                self._enter(1)
            elif self.p.can_short and short_sig is not None and self.d.short_entry_signal[0]:
                self._enter(-1)
        else:
            current_price = self.d.close[0]
            direction = 1 if self.position.size > 0 else -1

            if self.p.use_partial_exits and not self.partial_exit_done and getattr(self, 'partial_tp_price', None) and (
                (direction > 0 and current_price >= self.partial_tp_price) or (direction < 0 and current_price <= self.partial_tp_price)
            ):
                partial_size = self._round_qty(abs(self.position.size) * self.p.partial_exit_pct)
                if partial_size > 0:
                    if direction > 0:
                        self.sell(size=partial_size)
                    else:
                        self.buy(size=partial_size)
                    self.partial_exit_done = True
                    self.trail_stop_active = True

            if (direction > 0 and current_price >= getattr(self, 'take_profit_price', 1e18)) or (direction < 0 and current_price <= getattr(self, 'take_profit_price', -1e18)):
                self.close()
                return

            if self.p.time_limit_bars > 0 and (self.entry_bar is not None) and ((len(self) - self.entry_bar) > self.p.time_limit_bars) and not self.partial_exit_done:
                self.close()
                return

            if self.p.use_trailing_stop and (not self.p.use_partial_exits or self.partial_exit_done):
                self._update_trailing_stop()
                if self._stop_hit():
                    self.close()
                    return

    def notify_trade(self, trade):
        if trade.isclosed:
            setattr(self, 'last_exit_bar', len(self))
            self._reset_position_state()


# --- Scoring utilities ---

def score_metrics_from_strat(strat, df_slice: pl.DataFrame, init_value: float, final_value: float):
    """
    Robust extraction of performance metrics from the Backtrader strategy/analyzers.
    df_slice is used only for computing the evaluation period (CAGR). This code
    tolerates empty/NaT/odd timestamp types and falls back safely.
    """
    # default values
    sharpe = 0.0
    mdd = 0.0
    trades = 0
    wins = 0
    losses = 0
    profit_factor = 0.0
    cagr = 0.0

    # analyzers (defensive)
    try:
        sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr) if sr is not None and not math.isnan(sr) else 0.0
    except Exception:
        sharpe = 0.0
    try:
        mdd = float(strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0))
    except Exception:
        mdd = 0.0
    try:
        ta = strat.analyzers.trades.get_analysis()
        trades = int(ta.get("total", {}).get("total", 0))
        wins = int(ta.get("won", {}).get("total", 0))
        losses = int(ta.get("lost", {}).get("total", 0))
        gross_won = float(ta.get("won", {}).get("pnl", {}).get("gross", 0.0) or 0.0)
        gross_lost = float(ta.get("lost", {}).get("pnl", {}).get("gross", 0.0) or 0.0)
        profit_factor = (gross_won / abs(gross_lost)) if (gross_lost and gross_lost != 0.0) else (float('inf') if gross_won > 0 else 0.0)
    except Exception:
        trades, wins, losses, profit_factor = 0, 0, 0, 0.0

    # robustly compute CAGR using pandas.to_datetime; fallback to zero on failure
    try:
        if (df_slice is not None) and (not (hasattr(df_slice, "height") and df_slice.height == 0)):
            pdf = df_slice.to_pandas() if not isinstance(df_slice, pd.DataFrame) else df_slice
            if "TimestampStart" in pdf.columns:
                s0 = pd.to_datetime(pdf["TimestampStart"].min(), utc=True, errors='coerce')
                s1 = pd.to_datetime(pdf["TimestampStart"].max(), utc=True, errors='coerce')
                if pd.isna(s0) or pd.isna(s1) or (s1 <= s0):
                    cagr = 0.0
                else:
                    total_days = (s1 - s0).total_seconds() / 86400.0
                    years = max(1e-6, total_days / 365.25)
                    cagr = (final_value / max(1e-9, init_value)) ** (1.0 / years) - 1.0
            else:
                cagr = 0.0
        else:
            cagr = 0.0
    except Exception:
        cagr = 0.0

    return dict(sharpe=sharpe, mdd=mdd, trades=trades, wins=wins, losses=losses, pf=profit_factor, cagr=cagr)


# --- Backtest Runners ---
def run_backtest(strategy_class, df, spec: DataSpec, params: Dict[str, Any], init_cash=INIT_CASH, commission=COMMISSION_PER_TRANSACTION, exchange=None, slippage_bps=5.0, plot=True, debug=False):
    m = meta_for(spec.symbol)
    min_qty, qty_step, price_tick = m["min_qty"], m["qty_step"], m["price_tick"]
    
    cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
    df_slice = slice_df_to_spec(df, spec)
    
    feat_df = build_feature_df(df_slice, dict(params), base_tf=spec.interval)
    if feat_df is None or (hasattr(feat_df, "height") and feat_df.height == 0):
        console.print(f"[yellow]No feature rows for {spec.symbol} {spec.start_date}->{spec.end_date} (base {spec.interval})[/yellow]")
        return dict(trades=0, sharpe=0.0, maxdd=0.0, value=init_cash)
    feed = make_feature_feed(feat_df, name=spec.symbol)
    cerebro.adddata(feed)

    # Only pass params the strategy accepts — avoid unexpected kw errors
    strat_kwargs = {}
    try:
        valid_keys = set(strategy_class.params._getkeys())
    except Exception:
        try:
            valid_keys = set(k for k, _ in strategy_class.params)
        except Exception:
            valid_keys = set()

    for k, v in params.items():
        if k in valid_keys:
            strat_kwargs[k] = v

    # guaranteed kwargs
    strat_kwargs.update(dict(
        backtest=True,
        can_short=(str(exchange).lower() == "mexc") if exchange else False,
        min_qty=min_qty, qty_step=qty_step, price_tick=price_tick, debug=debug
    ))

    cerebro.addstrategy(strategy_class, **strat_kwargs)
    
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    try:
        # correct closing slippage: slippage_bps is basis points (e.g. 5 -> 0.0005)
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception:
        pass
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    console.print(f"Running backtest for {spec.symbol} ({spec.start_date} to {spec.end_date}) on base {spec.interval} ...")
    # exactbars=1 ensures runonce-style performance
    results = cerebro.run(maxcpus=1, exactbars=1)
    strat = results[0]
    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')
    
    final_value = cerebro.broker.getvalue()
    metrics = score_metrics_from_strat(strat, df_slice, init_cash, final_value)
    trades = int(metrics.get("trades", 0) or 0)
    losses = int(metrics.get("losses", 0) or 0)
    wins = int(metrics.get('wins', 0) or 0)
    sharpe = float(metrics.get("sharpe", 0) or 0.0)
    mdd = float(metrics.get('mdd', 0.0) or 0.0)

    console.print(f"\n--- Backtest Results for {spec.symbol} ---")
    console.print(f"Trades: {trades}, Wins: {wins}, Win Rate: {(wins/trades*100 if trades>0 else 0):.1f}%")
    console.print(f"Sharpe: {sharpe:.3f}, MaxDD: {mdd:.2f}%, Final Value: ${final_value:,.2f}")
    # cleanup
    try:
        del cerebro, feed, results, strat
        gc.collect()
    except Exception:
        pass
    return dict(trades=trades, sharpe=sharpe, maxdd=mdd, value=final_value)

def run_single_backtest_eval(strategy_class, df, spec: DataSpec, init_cash, commission, params, exchange=None, slippage_bps=5.0, min_qty=None, qty_step=None, price_tick=None):
    m = meta_for(spec.symbol); min_qty=min_qty or m["min_qty"]; qty_step=qty_step or m["qty_step"]; price_tick=price_tick or m["price_tick"]
    cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)

    # slice data early and guard for empty slices
    df_slice = slice_df_to_spec(df, spec)
    if df_slice is None or (hasattr(df_slice, "height") and df_slice.height == 0):
        # No data for this spec — return a clear error/metrics payload and the unchanged cash
        return {"error": 1.0, "trades": 0, "sharpe": 0.0, "mdd": 0.0, "cagr": 0.0}, init_cash

    # Build features and feed
    feat_df = build_feature_df(df_slice, dict(params), base_tf=spec.interval)
    if feat_df is None or (hasattr(feat_df, "height") and feat_df.height == 0):
        return {"error": 1.0, "trades": 0, "sharpe": 0.0, "mdd": 0.0, "cagr": 0.0}, init_cash

    feed = make_feature_feed(feat_df, name=f"{spec.symbol}")
    cerebro.adddata(feed)
    
    # filter params against strategy params to avoid unexpected kwargs
    sp = {}
    try:
        valid_keys = set(strategy_class.params._getkeys())
    except Exception:
        try:
            valid_keys = set(k for k, _ in strategy_class.params)
        except Exception:
            valid_keys = set()
    for k, v in params.items():
        if k in valid_keys:
            sp[k] = v

    sp.update(dict(can_short=(str(exchange).lower()=="mexc") if exchange else False, min_qty=min_qty, qty_step=qty_step, price_tick=price_tick))
    cerebro.addstrategy(strategy_class, **sp)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    cerebro.broker.setcommission(commission=commission); cerebro.broker.setcash(init_cash)
    try: cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except: pass

    try:
        strats = cerebro.run(maxcpus=1, exactbars=1)
        strat = strats[0]
    except Exception as e:
        console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]")
        traceback.print_exc()
        # cleanup
        try:
            del cerebro, feed, strats, strat
        except Exception:
            pass
        gc.collect()
        return {"error": 1.0, "trades": 0, "sharpe": 0.0, "mdd": 0.0, "cagr": 0.0}, init_cash

    final_value = cerebro.broker.getvalue()
    metrics = score_metrics_from_strat(strat, df_slice, init_cash, final_value)

    # cleanup
    try:
        del cerebro, feed, strats, strat
        gc.collect()
    except Exception:
        pass
    return metrics, final_value


# --- Optuna ---
def make_objective(strategy_class, specs, df_map, init_cash, commission,
                   exchange=None, min_trades_per_spec=30, use_median=True,
                   asset_weights=None) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        params = {
            # --- position sizing / risk ---
            "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.001, 0.02),
            "exposure_pct_cap": trial.suggest_float("exposure_pct_cap", 0.5, 1.0),
            "reserve_cash_pct": trial.suggest_float("reserve_cash_pct", 0.0, 0.1),
            "slip_fee_buffer_bps": trial.suggest_float("slip_fee_buffer_bps", 0.0, 25.0),
            "max_leverage": trial.suggest_float("max_leverage", 1.0, 10.0),

            # --- exits / trailing ---
            "use_dynamic_exits": trial.suggest_categorical("use_dynamic_exits", [True, False]),
            "tp_r_multiple": trial.suggest_float("tp_r_multiple", 1.0, 5.0),
            "use_partial_exits": trial.suggest_categorical("use_partial_exits", [True, False]),
            "partial_exit_r": trial.suggest_float("partial_exit_r", 0.5, 3.0),
            "partial_exit_pct": trial.suggest_float("partial_exit_pct", 0.1, 0.9),
            "time_limit_bars": trial.suggest_int("time_limit_bars", 30, 500),
            "atr_stop_mult": trial.suggest_float("atr_stop_mult", 1.0, 5.0),
            "use_trailing_stop": trial.suggest_categorical("use_trailing_stop", [True, False]),
            "trail_mode": trial.suggest_categorical("trail_mode", ["chandelier", "atr", "donchian"]),
            "trail_atr_mult": trial.suggest_float("trail_atr_mult", 1.0, 6.0),
            "move_to_breakeven_R": trial.suggest_float("move_to_breakeven_R", 0.5, 2.0),

            # --- execution / position management ---
            "reentry_cooldown_bars": trial.suggest_int("reentry_cooldown_bars", 0, 20),
            "min_qty": 0.0,
            "qty_step": 0.001,
            "price_tick": 0.1,
            "round_prices": True,
            "can_short": trial.suggest_categorical("can_short", [True, False]),

            # --- indicators / signals ---
            "ema_fast": trial.suggest_int("ema_fast", 5, 50),
            "ema_slow": trial.suggest_int("ema_slow", 20, 200),
            "ema_trend": trial.suggest_int("ema_trend", 100, 500),
            "atr_period": trial.suggest_int("atr_period", 5, 50),
            "rsi_overheat": trial.suggest_int("rsi_overheat", 60, 90),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 10, 40),
            "adxth": trial.suggest_int("adxth", 10, 40),
            "confirm_bars": trial.suggest_int("confirm_bars", 1, 5),
            "max_stretch_atr_mult": trial.suggest_float("max_stretch_atr_mult", 0.5, 3.0),
            "ema_band_mult": trial.suggest_float("ema_band_mult", 1.0, 4.0),
            "donchian_trail_period": trial.suggest_int("donchian_trail_period", 20, 100),
            "close_based_stop": trial.suggest_categorical("close_based_stop", [True, False]),
            "trail_update_every": trial.suggest_int("trail_update_every", 1, 10),
            "max_bars_in_trade": trial.suggest_int("max_bars_in_trade", 60, 1000),
            "use_pyramiding": trial.suggest_categorical("use_pyramiding", [True, False]),
            "max_adds": trial.suggest_int("max_adds", 0, 10),
            "add_cooldown": trial.suggest_int("add_cooldown", 5, 50),
            "add_atr_mult": trial.suggest_float("add_atr_mult", 0.5, 3.0),
            "add_min_R": trial.suggest_float("add_min_R", 0.5, 2.0),

            # --- volume filter ---
            "use_volume_filter": trial.suggest_categorical("use_volume_filter", [True, False]),
            "volume_filter_mult": trial.suggest_float("volume_filter_mult", 1.0, 2.0),

            # --- regime ---
            "regime_mode_long": trial.suggest_categorical("regime_mode_long", ["ema", "neutral", "off"]),
            "regime_mode_short": trial.suggest_categorical("regime_mode_short", ["neutral", "ema", "off"]),

            # --- HTF usage ---
            "use_htf": [True],

            # --- fallback TP ---
            "take_profit": trial.suggest_float("take_profit", 1.0, 10.0),

            # --- new params ---
            "breakout_period": trial.suggest_int("breakout_period", 20, 80),
            "htf1_adx_period": trial.suggest_int("htf1_adx_period", 10, 30),
            "htf1_ema_fast": trial.suggest_int("htf1_ema_fast", 10, 80),
            "htf1_ema_slow": trial.suggest_int("htf1_ema_slow", 50, 200),
            "htf2_ema_fast": trial.suggest_int("htf2_ema_fast", 50, 200),
            "htf2_ema_slow": trial.suggest_int("htf2_ema_slow", 100, 400),

            # --- compatibility ---
            "tf5m_breakout_period": trial.suggest_int("tf5m_breakout_period", 20, 80),
            "tf15m_adx_period": trial.suggest_int("tf15m_adx_period", 10, 30),
            "tf15m_ema_fast": trial.suggest_int("tf15m_ema_fast", 10, 80),
            "tf15m_ema_slow": trial.suggest_int("tf15m_ema_slow", 50, 200),
            "tf60m_ema_fast": trial.suggest_int("tf60m_ema_fast", 10, 100),
            "tf60m_ema_slow": trial.suggest_int("tf60m_ema_slow", 50, 300),
        }

        # --- merge with defaults from strategy class ---
        try:
            default_params = dict(strategy_class.params._getitems())
        except Exception:
            try:
                default_params = {k: v for k, v in strategy_class.params}
            except Exception:
                default_params = {}
        final_params = {**default_params, **params}

        # --- run all specs ---
        scores, weights = [], []
        for spec in specs:
            base_symbol = spec.symbol
            df = df_map.get(base_symbol)
            if df is None:
                console.print(f"[yellow]No data for {base_symbol} — skipping[/yellow]")
                continue

            metrics, final_value = run_single_backtest_eval(
                strategy_class, df, spec, init_cash, commission, final_params, exchange
            )

            trades = int(metrics.get("trades", 0))
            mdd = float(metrics.get("mdd", 100.0))
            cagr = float(metrics.get("cagr", 0.0))
            sharpe = float(metrics.get("sharpe", 0.0))

            # Calmar-like score + Sharpe bonus
            calmar = (cagr / (mdd / 100.0)) if mdd > 0 else (-10.0 if cagr <= 0 else 10.0)
            score = calmar + 0.25 * sharpe

            if trades < min_trades_per_spec:
                score -= 5.0

            scores.append(score)
            weights.append((asset_weights or {}).get(base_symbol, 1.0))

            for k, v in metrics.items():
                trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_{k}", v)

        if not scores:
            return -999.0
        if use_median:
            return float(np.median(scores))
        else:
            wsum = sum(weights) or 1.0
            return sum(s * w for s, w in zip(scores, weights)) / wsum
    return objective


def make_pruner(num_steps: int, pruner: Optional[str], n_jobs: int):
    if pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max(1, num_steps), reduction_factor=3, bootstrap_count=max(2 * n_jobs, 8))
    elif pruner == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
    elif pruner == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=1)
    else:
        return None

def optimize(strategy_class,specs,n_trials,n_jobs,init_cash,commission,pruner,storage_string,study_name,seed,exchange):
    storage=ensure_storage_or_sqlite(storage_string,study_name)
    df_map=preload_polars(specs)
    
    pruner_obj=make_pruner(len(specs),pruner,n_jobs)
    sampler=optuna.samplers.TPESampler(seed=seed,multivariate=True,group=True,constant_liar=True,warn_independent_sampling=False)
    study=optuna.create_study(direction="maximize",sampler=sampler,pruner=pruner_obj,storage=storage,study_name=study_name,load_if_exists=True)
    
    objective=make_objective(strategy_class,specs,df_map,init_cash,commission,exchange=exchange)
    
    console.print(f"Starting Optuna: trials={n_trials}, n_jobs={n_jobs}, pruner={pruner}")
    study.optimize(objective,n_trials=n_trials,n_jobs=n_jobs,gc_after_trial=True,show_progress_bar=True)
    
    table=Table(title="Best Parameters"); table.add_column("Parameter"); table.add_column("Value")
    for k,v in study.best_params.items(): table.add_row(k,str(v))
    console.print(table)
    return study,study.best_params

def _worker_optimize(
    worker_id: int,
    strategy_class,
    specs: List[DataSpec],
    storage_string: str,
    study_name: str,
    trials_for_worker: int,
    init_cash: float,
    comm: float,
    exch: str,
    seed: int,
):
    try:
        # ensure each worker can connect
        storage = ensure_storage_or_sqlite(storage_string, study_name)
        optimize(
            strategy_class=strategy_class,
            specs=specs,
            n_trials=trials_for_worker,
            n_jobs=1,
            init_cash=init_cash,
            commission=comm,
            pruner="hyperband",
            storage_string=storage,
            study_name=study_name,
            seed=seed,
            exchange=exch,
        )
    except Exception as e:
        console.print(f"[red]Worker {worker_id} crashed: {e}[/red]")
        traceback.print_exc()

def launch_multiprocess_optimize(
    strategy_class,
    specs,
    storage_string,
    study_name,
    total_trials,
    workers,
    init_cash,
    comm,
    exch,
    seed_base,
):
    trials_per = math.ceil(total_trials / workers)
    console.print(f"[bold magenta]Launching {workers} workers @ {trials_per} trials each[/bold magenta]")
    preload_polars(specs, force_refresh=False)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    procs = []
    for wid in range(workers):
        p = mp.Process(
            target=_worker_optimize,
            args=(
                wid,
                strategy_class,
                specs,
                storage_string,
                study_name,
                trials_per,
                init_cash,
                comm,
                exch,
                seed_base + wid,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    storage = ensure_storage_or_sqlite(storage_string, study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    console.print(f"[green]Multiprocess optimize done. Best value: {study.best_value:.4f}[/green]")
    console.print(study.best_params)

# --- Main execution block ---

storage = ensure_storage_or_sqlite(MSSQL_ODBC, STUDYNAME)
study = optuna.load_study(study_name=STUDYNAME, storage=storage)
console.print(f"[green]Multiprocess optimize done. Best value: {study.best_value:.4f}[/green]")
console.print(study.best_params)

if __name__ == "__main__":
    bull_start, bull_end = "2020-09-28", "2021-05-31"
    bear_start, bear_end = "2022-05-28", "2023-06-23"
    holdout_start, holdout_end = "2023-06-12", "2025-05-31"

    train_specs = expand_specs_by_ranges([
        DataSpec("BTC", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        DataSpec("ETH", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
    ])
    build_cache(train_specs, force_refresh=False)

    # Use the same dataset as optimization (or swap for holdout / sanity check)
    df_map_full = preload_polars(train_specs)

    best_params = study.best_params  # <-- directly from Optuna

    console.print("[yellow]Final backtest on full dataset with best Optuna params...[/yellow]")
    for spec in train_specs:   # backtest BTC and ETH separately
        run_backtest(
            VectorMACD_ADX,
            df_map_full[spec.symbol],
            spec,                  # single DataSpec
            best_params,           # <-- best params here
            init_cash=INIT_CASH,
            exchange="MEXC",
            plot=False,
            debug=False
        )


# storage = ensure_storage_or_sqlite(MSSQL_ODBC, STUDYNAME)
# study = optuna.load_study(study_name=STUDYNAME, storage=storage)
# console.print(f"[green]Multiprocess optimize done. Best value: {study.best_value:.4f}[/green]")
# console.print(study.best_params)

# if __name__ == "__main__":
#     try:
#         bull_start, bull_end = "2020-09-28", "2021-05-31"
#         bear_start, bear_end = "2022-05-28", "2023-06-23"
#         holdout_start, holdout_end = "2023-06-12", "2025-05-31"
#         train_specs = expand_specs_by_ranges([
#             DataSpec("BTC", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
#             DataSpec("ETH", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
#         ])
#         build_cache(train_specs, force_refresh=False)

#         sanity_spec = DataSpec("BTC", "15m", start_date="2024-01-01", end_date="2024-01-31")
#         # df_map_sanity = preload_polars([train_specs])
#         df_map_sanity = preload_polars(train_specs)
        
        
#         sanity_params = {
#             'breakout_period': 40,
#             'adxth': 20,
#             'confirm_bars': 1,
#             'atr_stop_mult': 2.5,
#             'trail_mode': 'chandelier',
#             'trail_atr_mult': 4.0,
#             'move_to_breakeven_R': 0.75,
#             'risk_per_trade_pct': 0.003,
#             'rsi_overheat': 80,
#             'rsi_oversold': 25,
#             'use_pyramiding': False,
#             'use_volume_filter': False,
#             'volume_filter_mult': 1.2,
#             'ema_fast': 20,
#             'ema_slow': 60,
#             'atr_period': 14,
#             'donchian_trail_period': 55,
#             'trail_update_every': 2,
#             'max_adds': 0,
#             'add_cooldown': 20,
#             'add_atr_mult': 1.0,
#             'add_min_R': 1.0,
#             'close_based_stop': True,
#             'use_htf': True,
#             'regime_mode_long': 'price_vs_slow',
#             'regime_mode_short': 'neutral',
#             'use_dynamic_exits': True,
#             'tp_r_multiple': 2.0,
#             'use_partial_exits': True,
#             'partial_exit_r': 1.5,
#             'partial_exit_pct': 0.5,
#             'time_limit_bars': 120,
#             'htf1_ema_fast': 40, 'htf1_ema_slow': 120, 'htf1_adx_period': 14,
#             'htf2_ema_fast': 80, 'htf2_ema_slow': 200,
#             'max_stretch_atr_mult': 1.0,
#             'exposure_pct_cap': 0.9,
#             'reserve_cash_pct': 0.02,
#             'slip_fee_buffer_bps': 10.0,
#         }

#         console.print("[yellow]Sanity backtest (BTC 15m Jan 2024) with vector features...[/yellow]")
#         run_backtest(
#             VectorMACD_ADX,
#             df_map_sanity["BTC"],
#             sanity_spec,              # pass the single spec, not the whole train_specs list
#             sanity_params,
#             init_cash=INIT_CASH,
#             exchange="MEXC",
#             plot=True,
#             debug=True
#         )

#         study_name = STUDYNAME
#         launch_multiprocess_optimize(
#             strategy_class=VectorMACD_ADX,
#             specs=train_specs,
#             storage_string=MSSQL_ODBC,
#             study_name=study_name,
#             total_trials=200,
#             workers=min(8, max(1, mp.cpu_count()-1)),
#             init_cash=INIT_CASH,
#             comm=COMMISSION_PER_TRANSACTION,
#             exch="MEXC",
#             seed_base=42,
#         )

#         storage = ensure_storage_or_sqlite(MSSQL_ODBC, study_name)
#         study = optuna.load_study(study_name=study_name, storage=storage)
#         best_params = study.best_params
#         console.print("[green]Optuna best params:[/green]")
#         console.print(best_params)

#         holdout_spec = DataSpec("BNB", "15m", start_date=holdout_start, end_date=holdout_end)
#         df_hold = preload_polars([holdout_spec])["BNB"]
#         console.print("[magenta]Holdout backtest with best params on BNB (15m)...[/magenta]")
#         run_backtest(VectorMACD_ADX, df_hold, holdout_spec, best_params, init_cash=INIT_CASH, exchange="MEXC", plot=False, debug=False)
#     except KeyboardInterrupt:
#         console.print("Process interrupted by user.")
#     except Exception as e:
#         console.print(f"An error occurred: {e}")
#         traceback.print_exc()





# # from backtrader.utils.backtest import backtest
# # from testing_optuna_newmacd import build_optuna_storage
# # storage = build_optuna_storage(MSSQL_ODBC)
# # study = optuna.load_study(study_name=STUDYNAME, storage=storage)
# # strategy = VectorMACD_ADX

# # trial_num = None # or None for best
# # trial = (study.best_trial if trial_num is None
# #          else next(t for t in study.get_trials(deepcopy=False) if t.number == trial_num))

# # raw_params = trial.params

# # def get_param_names(cls) -> set:
# #     names = set()
# #     try:
# #         names = set(cls.params._getkeys())  # type: ignore[attr-defined]
# #     except Exception:
# #         try:
# #             # legacy tuple-of-tuples style
# #             names = set(k for k, _ in cls.params)  # type: ignore[assignment]
# #         except Exception:
# #             # fallback: just trust trial params
# #             names = set(raw_params.keys())
# #     return names

# # param_names = get_param_names(strategy)
# # params = {k: v for k, v in raw_params.items() if k in param_names}


# # # --------------- Data spec ---------------
# # bull_start = "2020-09-28"
# # bull_end = "2021-05-31"
# # bear_start = "2022-05-28"
# # bear_end = "2023-06-23"
# # # Optional holdout test period
# # test_bull_start="2023-06-12"
# # test_bull_end="2025-05-31"
# # tf = "1m"

# # if __name__ == '__main__':
# #     console.print(f"Using params: {params}")
# #     # print(f"All raw params: {raw_params}")
# #     console.print(f"Trial number: {trial.number}")
# #     # print(f"Trial value: {trial.value}")
# #     # print(f"Trial state: {trial.state}")


# #     bull_start, bull_end = "2020-09-28", "2021-05-31"
# #     bear_start, bear_end = "2022-05-28", "2023-06-23"
# #     holdout_start, holdout_end = "2023-06-12", "2025-05-31"
# #     train_specs = expand_specs_by_ranges([
# #         DataSpec("BTC", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
# #         # DataSpec("ETH", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
# #     ])
# #     build_cache(train_specs, force_refresh=True)

# #     sanity_spec = DataSpec("BTC", "1m", start_date="2024-01-01", end_date="2024-01-31")
# #     df_map_sanity = preload_polars([sanity_spec])

# #     try:
# #         run_backtest(VectorMACD_ADX, df_map_sanity["BTC"], sanity_spec, params, init_cash=INIT_CASH, exchange="MEXC", plot=True, debug=True)

# #     except Exception as e:
# #         console.print(f"An error occurred: {e}")
# #         import traceback
# #         traceback.print_exc()
# #     except KeyboardInterrupt:
# #         console.print("Process interrupted by user.")