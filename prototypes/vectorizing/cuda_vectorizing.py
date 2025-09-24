# ultra_fast_cuda_backtest.py - Complete CUDA-accelerated backtesting system
import os
import gc
import math
import time
import traceback
import urllib.parse
import cProfile
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
from numba import jit

# CUDA imports (with fallback)
try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    CUDA_AVAILABLE = False

import backtrader as bt
import optuna
import polars as pl
from rich.console import Console
from rich.table import Table
from pathlib import Path

# Your imports (adjust paths as needed)
from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData
from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC

console = Console()

# --- CUDA Kernels ---
if CUDA_AVAILABLE:
    cuda_signal_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void compute_signals_kernel(
        const float* close, const float* high, const float* low, const float* volume,
        const float* ema1_fast, const float* ema1_slow, 
        const float* ema15_fast, const float* ema15_slow,
        const float* ema60_fast, const float* ema60_slow,
        const float* atr5, const float* dc_high5_prev, const float* dc_low5_prev,
        const float* adx, const float* plus_di, const float* minus_di,
        const float* rsi, const float* vsma1,
        bool* long_signals, bool* short_signals,
        int n, int confirm_bars,
        float adxth, float rsi_overheat, float rsi_oversold,
        float max_stretch_atr_mult, float volume_filter_mult,
        bool use_volume_filter
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (i >= n || i < confirm_bars) return;
        
        // Check for NaN values
        if (isnan(close[i]) || isnan(ema60_fast[i]) || isnan(ema60_slow[i]) ||
            isnan(adx[i]) || isnan(plus_di[i]) || isnan(minus_di[i]) ||
            isnan(ema15_fast[i]) || isnan(ema15_slow[i]) ||
            isnan(ema1_fast[i]) || isnan(ema1_slow[i]) ||
            isnan(dc_high5_prev[i]) || isnan(atr5[i]) ||
            isnan(rsi[i]) || isnan(vsma1[i])) {
            return;
        }
        
        // Breakout logic
        bool breakout_up = true;
        bool breakdown_down = true;
        
        for (int j = 0; j < confirm_bars; j++) {
            int idx = i - j;
            if (idx >= 0 && !isnan(close[idx]) && !isnan(dc_high5_prev[idx]) && !isnan(dc_low5_prev[idx])) {
                if (close[idx] <= dc_high5_prev[idx]) breakout_up = false;
                if (close[idx] >= dc_low5_prev[idx]) breakdown_down = false;
            }
        }
        
        // Volume filter
        bool vol_ok = true;
        if (use_volume_filter && !isnan(volume[i]) && !isnan(vsma1[i])) {
            if (volume[i] <= volume_filter_mult * vsma1[i]) vol_ok = false;
        }
        
        // Long signal
        if (ema60_fast[i] > ema60_slow[i] &&
            adx[i] >= adxth &&
            plus_di[i] > minus_di[i] &&
            ema15_fast[i] > ema15_slow[i] &&
            ema1_fast[i] > ema1_slow[i] &&
            breakout_up &&
            (close[i] - dc_high5_prev[i]) <= max_stretch_atr_mult * atr5[i] &&
            vol_ok &&
            rsi[i] < rsi_overheat) {
            long_signals[i] = true;
        }
        
        // Short signal
        if (ema60_fast[i] < ema60_slow[i] &&
            adx[i] >= adxth &&
            minus_di[i] > plus_di[i] &&
            ema15_fast[i] < ema15_slow[i] &&
            ema1_fast[i] < ema1_slow[i] &&
            breakdown_down &&
            (dc_low5_prev[i] - close[i]) <= max_stretch_atr_mult * atr5[i] &&
            vol_ok &&
            rsi[i] > rsi_oversold) {
            short_signals[i] = true;
        }
    }
    ''', 'compute_signals_kernel')

# --- JIT-Compiled CPU Functions (Fallback) ---
@jit(nopython=True, fastmath=True, cache=True)
def compute_all_signals_vectorized(
    close, high, low, volume, 
    ema1_fast, ema1_slow, ema15_fast, ema15_slow, 
    ema60_fast, ema60_slow, atr5, dc_high5_prev, dc_low5_prev,
    adx, plus_di, minus_di, rsi, vsma1,
    adxth, rsi_overheat, rsi_oversold, max_stretch_atr_mult,
    volume_filter_mult, use_volume_filter, confirm_bars
):
    n = len(close)
    long_signals = np.zeros(n, dtype=np.bool_)
    short_signals = np.zeros(n, dtype=np.bool_)
    
    for i in range(confirm_bars, n):
        if (np.isnan(close[i]) or np.isnan(ema60_fast[i]) or np.isnan(ema60_slow[i]) or
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema15_fast[i]) or np.isnan(ema15_slow[i]) or np.isnan(ema1_fast[i]) or
            np.isnan(ema1_slow[i]) or np.isnan(dc_high5_prev[i]) or np.isnan(atr5[i]) or
            np.isnan(rsi[i]) or np.isnan(vsma1[i])):
            continue
            
        breakout_up = True
        breakdown_down = True
        for j in range(confirm_bars):
            if i-j >= 0 and not np.isnan(close[i-j]) and not np.isnan(dc_high5_prev[i-j]) and not np.isnan(dc_low5_prev[i-j]):
                if close[i-j] <= dc_high5_prev[i-j]:
                    breakout_up = False
                if close[i-j] >= dc_low5_prev[i-j]:
                    breakdown_down = False
        
        vol_ok = True
        if use_volume_filter and not np.isnan(volume[i]) and not np.isnan(vsma1[i]):
            if volume[i] <= volume_filter_mult * vsma1[i]:
                vol_ok = False
            
        if (ema60_fast[i] > ema60_slow[i] and adx[i] >= adxth and
            plus_di[i] > minus_di[i] and ema15_fast[i] > ema15_slow[i] and
            ema1_fast[i] > ema1_slow[i] and breakout_up and
            (close[i] - dc_high5_prev[i]) <= max_stretch_atr_mult * atr5[i] and
            vol_ok and rsi[i] < rsi_overheat):
            long_signals[i] = True
            
        if (ema60_fast[i] < ema60_slow[i] and adx[i] >= adxth and
            minus_di[i] > plus_di[i] and ema15_fast[i] < ema15_slow[i] and
            ema1_fast[i] < ema1_slow[i] and breakdown_down and
            (dc_low5_prev[i] - close[i]) <= max_stretch_atr_mult * atr5[i] and
            vol_ok and rsi[i] > rsi_oversold):
            short_signals[i] = True
    
    return long_signals, short_signals

# --- CUDA Acceleration Functions ---
def manage_gpu_memory():
    """Clean up GPU memory"""
    if CUDA_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except:
            pass

def compute_signals_cuda(
    close, high, low, volume, 
    ema1_fast, ema1_slow, ema15_fast, ema15_slow, 
    ema60_fast, ema60_slow, atr5, dc_high5_prev, dc_low5_prev,
    adx, plus_di, minus_di, rsi, vsma1,
    adxth, rsi_overheat, rsi_oversold, max_stretch_atr_mult,
    volume_filter_mult, use_volume_filter, confirm_bars
):
    n = len(close)
    
    # Transfer data to GPU
    arrays = [close, high, low, volume, ema1_fast, ema1_slow, ema15_fast, ema15_slow,
              ema60_fast, ema60_slow, atr5, dc_high5_prev, dc_low5_prev,
              adx, plus_di, minus_di, rsi, vsma1]
    
    gpu_arrays = [cp.asarray(arr, dtype=cp.float32) for arr in arrays]
    
    # Allocate output arrays on GPU
    long_signals_gpu = cp.zeros(n, dtype=cp.bool_)
    short_signals_gpu = cp.zeros(n, dtype=cp.bool_)
    
    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    cuda_signal_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (*gpu_arrays, long_signals_gpu, short_signals_gpu,
         n, confirm_bars, float(adxth), float(rsi_overheat), float(rsi_oversold),
         float(max_stretch_atr_mult), float(volume_filter_mult), bool(use_volume_filter))
    )
    
    # Transfer results back to CPU
    return cp.asnumpy(long_signals_gpu), cp.asnumpy(short_signals_gpu)

def compute_ema_cuda(prices, period):
    """GPU-accelerated EMA computation"""
    prices_gpu = cp.asarray(prices, dtype=cp.float32)
    alpha = 2.0 / (period + 1.0)
    
    ema_gpu = cp.zeros_like(prices_gpu)
    ema_gpu[0] = prices_gpu[0]
    
    # Sequential computation - could be optimized further
    for i in range(1, len(prices_gpu)):
        ema_gpu[i] = alpha * prices_gpu[i] + (1 - alpha) * ema_gpu[i-1]
    
    return cp.asnumpy(ema_gpu)

def compute_atr_cuda(high, low, close, period):
    """GPU-accelerated ATR computation"""
    high_gpu = cp.asarray(high, dtype=cp.float32)
    low_gpu = cp.asarray(low, dtype=cp.float32)
    close_gpu = cp.asarray(close, dtype=cp.float32)
    
    # True Range calculation
    tr1 = high_gpu - low_gpu
    tr2 = cp.abs(high_gpu - cp.roll(close_gpu, 1))
    tr3 = cp.abs(low_gpu - cp.roll(close_gpu, 1))
    
    tr = cp.maximum(tr1, cp.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Wilder's smoothing
    alpha = 1.0 / period
    atr_gpu = cp.zeros_like(tr)
    atr_gpu[0] = tr[0]
    
    for i in range(1, len(tr)):
        atr_gpu[i] = alpha * tr[i] + (1 - alpha) * atr_gpu[i-1]
    
    return cp.asnumpy(atr_gpu)

# --- Configuration & Globals ---
INIT_CASH = 100_000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"
FEATURE_CACHE = {}
MIN_CUDA_SIZE = 1000  # Minimum data size to use CUDA

CACHE_DIR = Path(os.getenv("BTQ_CACHE_DIR", ".btq_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

# --- Storage & Caching ---
def ensure_storage_or_sqlite(storage_string: Optional[str], study_name: str):
    if storage_string is None:
        sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
        console.print(f"[yellow]Using SQLite at {sqlite_path}[/yellow]")
        return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")
    
    if not storage_string.startswith("mssql+pyodbc://"):
        parts = {k.strip().upper(): v.strip() for chunk in storage_string.split(";") 
                if "=" in chunk and chunk.strip() for k, v in [chunk.split("=", 1)]}
        server = parts.get("SERVER", "localhost")
        db = parts.get("DATABASE", "OptunaBT")
        uid = parts.get("UID", "SA")
        pwd = parts.get("PWD", "")
        driver = urllib.parse.quote_plus(parts.get("DRIVER", "{ODBC Driver 18 for SQL Server}").strip("{}"))
        query = f"driver={driver}&Encrypt=yes&TrustServerCertificate={parts.get('TRUSTSERVERCERTIFICATE', 'yes')}"
        storage_string = f"mssql+pyodbc://{uid}:{urllib.parse.quote_plus(pwd)}@{server}/{db}?{query}"
    
    try:
        return optuna.storages.RDBStorage(
            url=storage_string,
            engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300, "connect_args": {"timeout": 30}},
        )
    except Exception as e:
        console.print(f"[red]RDB storage failed: {e}, falling back to SQLite[/red]")
        sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
        return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")

class PolarsDataLoader:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir

    def _cache_key(self, symbol: str, interval: str, collateral: str, start: str, end: str) -> str:
        raw = f"{symbol}|{interval}|{collateral}|{start}:{end}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{symbol}_{interval}_{collateral}_{h}.parquet"

    def load_union_by_symbol(self, symbol: str, interval: str, collateral: str, pairs: List[Tuple[str,str]], use_cache=True) -> pl.DataFrame:
        start, end = min(s for s,_ in pairs), max(e for _,e in pairs)
        path = self.cache_dir / self._cache_key(symbol, interval, collateral, start, end)
        
        if use_cache and path.exists():
            try:
                console.print(f"[cyan]Loaded from cache: {symbol} ({start}..{end})[/cyan]")
                return pl.scan_parquet(str(path)).collect()
            except Exception as e:
                console.print(f"[yellow]Cache read failed {path.name}: {e}[/yellow]")
        
        df = get_database_data(ticker=symbol, start_date=start, end_date=end, time_resolution=interval, pair=collateral)
        if df is None or df.is_empty():
            raise ValueError(f"No data for {symbol} {interval} {start}->{end}")
        
        df = df.sort("TimestampStart")
        if use_cache:
            df.write_parquet(str(path), compression="zstd", statistics=True)
            console.print(f"[green]Cached -> {path}[/green]")
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
    if df.is_empty(): 
        return df
    start, end = spec.start_date, spec.end_date
    if start: 
        df = df.filter(pl.col("TimestampStart") >= pl.lit(start).str.to_datetime())
    if end:   
        df = df.filter(pl.col("TimestampStart") <= pl.lit(end).str.to_datetime())
    return df

# --- Feature Engineering ---
def ema(expr: pl.Expr, n: int) -> pl.Expr: 
    return expr.ewm_mean(alpha=2.0/(n+1.0), adjust=False)

def wilder(expr: pl.Expr, n: int) -> pl.Expr: 
    return expr.ewm_mean(alpha=1.0/max(1,n), adjust=False)

def add_atr(lf: pl.LazyFrame, n: int, prefix: str) -> pl.LazyFrame:
    tr = pl.max_horizontal(
        (pl.col("High")-pl.col("Low")), 
        (pl.col("High")-pl.col("Close").shift(1)).abs(), 
        (pl.col("Low")-pl.col("Close").shift(1)).abs()
    )
    return lf.with_columns(wilder(tr, n).alias(f"{prefix}_atr"))

def add_rsi(lf: pl.LazyFrame, n: int = 14) -> pl.LazyFrame:
    diff = pl.col("Close") - pl.col("Close").shift(1)
    up = pl.when(diff > 0).then(diff).otherwise(0.0)
    down = pl.when(diff < 0).then(-diff).otherwise(0.0)
    return lf.with_columns(wilder(up, n).alias("_up"), wilder(down, n).alias("_down")) \
             .with_columns((pl.col("_up") / (pl.col("_down") + 1e-12)).alias("_rs")) \
             .with_columns((100.0 - (100.0 / (1.0 + pl.col("_rs")))).alias("rsi")) \
             .drop(["_up", "_down", "_rs"])

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
    return lf.with_columns(
        wilder(tr, n).alias("_trn"),
        wilder(pdm, n).alias("_pdmn"),
        wilder(mdm, n).alias("_mdmn")
    ).with_columns(
        (100 * pl.col("_pdmn") / (pl.col("_trn") + 1e-12)).alias("plus_di"),
        (100 * pl.col("_mdmn") / (pl.col("_trn") + 1e-12)).alias("minus_di")
    ).with_columns(
        (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() / (pl.col("plus_di") + pl.col("minus_di") + 1e-12)).alias("_dx")
    ).with_columns(
        wilder(pl.col("_dx"), n).alias("adx")
    ).drop(["_trn", "_pdmn", "_mdmn", "_dx"])

def resample(lf: pl.LazyFrame, every: str) -> pl.LazyFrame:
    return lf.group_by_dynamic("TimestampStart", every=every, period=every, closed="right") \
             .agg(pl.first("Open"), pl.max("High"), pl.min("Low"), pl.last("Close"), pl.sum("Volume")) \
             .sort("TimestampStart")

def build_feature_df(df_1m: pl.DataFrame, params: Dict[str, Any], use_cuda: bool = True) -> pl.DataFrame:
    """Build features with optional CUDA acceleration"""
    
    # Check if CUDA is beneficial
    use_cuda = use_cuda and CUDA_AVAILABLE and len(df_1m) >= MIN_CUDA_SIZE
    
    if use_cuda:
        console.print(f"[green]Using CUDA for feature computation ({len(df_1m)} rows)[/green]")
    
    p = params
    lf_1m = df_1m.lazy().sort("TimestampStart")
    
    if use_cuda:
        # GPU-accelerated feature computation
        close_np = df_1m.select("Close").to_numpy().flatten()
        high_np = df_1m.select("High").to_numpy().flatten()
        low_np = df_1m.select("Low").to_numpy().flatten()
        
        try:
            ema_fast = compute_ema_cuda(close_np, p["ema_fast"])
            ema_slow = compute_ema_cuda(close_np, p["ema_slow"])
            atr = compute_atr_cuda(high_np, low_np, close_np, p["atr_period"])
            
            lf_1m_feat = lf_1m.with_columns([
                pl.Series("ema1_fast", ema_fast),
                pl.Series("ema1_slow", ema_slow),
                pl.Series("atr1_computed", atr),
                pl.col("Volume").rolling_mean(20, min_samples=20).alias("vsma1")
            ])
            
            manage_gpu_memory()  # Clean up GPU memory
        except Exception as e:
            console.print(f"[yellow]CUDA computation failed: {e}, using CPU[/yellow]")
            use_cuda = False
    
    if not use_cuda:
        # CPU computation
        lf_1m_feat = lf_1m.with_columns(
            ema(pl.col("Close"), p["ema_fast"]).alias("ema1_fast"),
            ema(pl.col("Close"), p["ema_slow"]).alias("ema1_slow"),
            pl.col("Volume").rolling_mean(20, min_samples=20).alias("vsma1")
        )
    
    lf_1m_feat = add_atr(lf_1m_feat, p["atr_period"], "atr1")
    lf_1m_feat = add_rsi(lf_1m_feat, 14)

    # Continue with multi-timeframe features (CPU)
    lf_5m = resample(lf_1m, "5m")
    lf_5m = add_atr(lf_5m, p["atr_period"], "atr5")
    lf_5m = lf_5m.with_columns(
        pl.col("High").rolling_max(p["tf5m_breakout_period"]).shift(1).alias("dc_high5_prev"),
        pl.col("Low").rolling_min(p["tf5m_breakout_period"]).shift(1).alias("dc_low5_prev"),
        pl.col("Low").rolling_min(p["donchian_trail_period"]).alias("dc_exit5_low"),
        pl.col("High").rolling_max(p["donchian_trail_period"]).alias("dc_exit5_high"),
    )

    lf_15m = resample(lf_1m, "15m").with_columns(
        ema(pl.col("Close"), p["tf15m_ema_fast"]).alias("ema15_fast"),
        ema(pl.col("Close"), p["tf15m_ema_slow"]).alias("ema15_slow")
    )
    lf_15m = add_adx(lf_15m, p["tf15m_adx_period"])

    lf_60m = resample(lf_1m, "60m").with_columns(
        ema(pl.col("Close"), p["tf60m_ema_fast"]).alias("ema60_fast"),
        ema(pl.col("Close"), p["tf60m_ema_slow"]).alias("ema60_slow")
    )

    # Join all features
    lf_all = lf_1m_feat.join_asof(
        lf_5m.select([
            "TimestampStart", "atr5_atr", "dc_high5_prev", "dc_low5_prev",
            "dc_exit5_low", "dc_exit5_high"
        ]),
        on="TimestampStart", strategy="backward", suffix="_5m"
    ).join_asof(
        lf_15m.select([
            "TimestampStart", "plus_di", "minus_di", "adx",
            "ema15_fast", "ema15_slow"
        ]),
        on="TimestampStart", strategy="backward", suffix="_15m"
    ).join_asof(
        lf_60m.select([
            "TimestampStart", "ema60_fast", "ema60_slow"
        ]),
        on="TimestampStart", strategy="backward", suffix="_60m"
    )

    # Add entry signals (CPU computation)
    cb = p["confirm_bars"]
    lf_all = lf_all.with_columns(
        (pl.col("Close").rolling_min(cb, min_samples=cb) > pl.col("dc_high5_prev")).alias("breakout_up"),
        (pl.col("Close").rolling_max(cb, min_samples=cb) < pl.col("dc_low5_prev")).alias("breakdown_down")
    )

    # Regime logic
    long_regime = (pl.col("Close") > pl.col("ema60_slow")) if p.get("regime_mode_long") == "price_vs_slow" else (pl.col("ema60_fast") > pl.col("ema60_slow"))
    if p.get("regime_mode_long") == 'off':
        long_regime = pl.lit(True)
    short_regime = pl.lit(True) if p.get("regime_mode_short") == "neutral" else (pl.col("ema60_fast") < pl.col("ema60_slow"))
    if p.get("regime_mode_short") == 'off':
        short_regime = pl.lit(True)
    vol_ok = pl.when(pl.lit(p.get("use_volume_filter"))).then(
        pl.col("Volume") > p["volume_filter_mult"] * pl.col("vsma1")
    ).otherwise(True)

    # Entry signals
    lf_all = lf_all.with_columns(
        (long_regime & (pl.col("adx") >= p["adxth"]) & (pl.col("plus_di") > pl.col("minus_di")) &
         (pl.col("ema15_fast") > pl.col("ema15_slow")) & (pl.col("ema1_fast") > pl.col("ema1_slow")) &
         pl.col("breakout_up") &
         ((pl.col("Close") - pl.col("dc_high5_prev")) <= p["max_stretch_atr_mult"] * pl.col("atr5_atr")) &
         vol_ok & (pl.col("rsi")< p["rsi_overheat"])
        ).alias("long_entry_signal"),
        (short_regime & (pl.col("adx") >= p["adxth"]) & (pl.col("minus_di") > pl.col("plus_di")) &
         (pl.col("ema15_fast") < pl.col("ema15_slow")) & (pl.col("ema1_fast") < pl.col("ema1_slow")) &
         pl.col("breakdown_down") &
         ((pl.col("dc_low5_prev") - pl.col("Close")) <= p["max_stretch_atr_mult"] * pl.col("atr5_atr")) &
         vol_ok & (pl.col("rsi") > p["rsi_oversold"])
        ).alias("short_entry_signal")
    )

    # Collect and clean up
    df = lf_all.collect()
    df = df.rename({"atr1_atr": "atr1", "atr5_atr": "atr5"})
    required = ["atr5", "dc_high5_prev", "adx", "ema60_fast", "ema1_fast", "atr1", "rsi", "vsma1"]
    df = df.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in required)) \
           .with_columns(pl.col(pl.Boolean).fill_null(False).cast(pl.Float64))

    # Select columns to keep
    keep = [
        "TimestampStart", "Open", "High", "Low", "Close", "Volume",
        "ema1_fast", "ema1_slow", "atr1", "rsi", "vsma1",
        "atr5", "dc_high5_prev", "dc_low5_prev", "dc_exit5_low", "dc_exit5_high",
        "ema15_fast", "ema15_slow", "plus_di", "minus_di", "adx",
        "ema60_fast", "ema60_slow", "long_entry_signal", "short_entry_signal"
    ]
    return df.select([c for c in keep if c in df.columns])

def get_or_build_features(df_slice, spec, params, use_cuda=True):
    """Cache expensive feature computation with CUDA support"""
    cache_key = f"{spec.symbol}_{spec.start_date}_{spec.end_date}_{hash(str(sorted(params.items())))}_{use_cuda}"
    
    if cache_key not in FEATURE_CACHE:
        acceleration = "CUDA" if use_cuda and CUDA_AVAILABLE else "CPU"
        console.print(f"[yellow]Computing features for {spec.symbol} using {acceleration}[/yellow]")
        FEATURE_CACHE[cache_key] = build_feature_df(df_slice, params, use_cuda)
    else:
        console.print(f"[green]Using cached features for {spec.symbol}[/green]")
    
    return FEATURE_CACHE[cache_key]

# --- Backtrader Components ---
class FeatureData(bt.feeds.PolarsData):
    lines = ('ema1_fast','ema1_slow','atr1','rsi','vsma1','atr5','dc_high5_prev','dc_low5_prev','dc_exit5_low','dc_exit5_high','ema15_fast','ema15_slow','plus_di','minus_di','adx','ema60_fast','ema60_slow','long_entry_signal','short_entry_signal')
    params = dict(datetime='TimestampStart',open='Open',high='High',low='Low',close='Close',volume='Volume',openinterest=-1, **{k:k for k in lines})

def make_feature_feed(df_feat: pl.DataFrame, name: str) -> FeatureData:
    pdf = df_feat.to_pandas()
    feed = FeatureData(dataname=pdf)
    feed._name = f"{name}-features"
    return feed

class CUDAUltraFastVectorMACD_ADX(bt.Strategy):
    params = (
        ('risk_per_trade_pct', 0.0001),
        ('max_leverage', 2.0),
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
        ('reentry_cooldown_bars', 5),
        ('min_qty', 0.0),
        ('qty_step', 0.001),
        ('price_tick', 0.1),
        ('round_prices', True),
        ('can_short', False),
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
        ('ema_band_mult', 2.0),
        ('donchian_trail_period', 55),
        ('close_based_stop', True),
        ('trail_update_every', 2),
        ('max_bars_in_trade', 360),
        ('use_pyramiding', False),
        ('max_adds', 0),
        ('add_cooldown', 20),
        ('add_atr_mult', 1.0),
        ('add_min_R', 1.0),
        ('use_volume_filter', False),
        ('volume_filter_mult', 1.2),
        ('backtest', True),
        ('debug', False),
        ('regime_mode_long', 'ema'),
        ('regime_mode_short', 'neutral'),
        ('take_profit', 4.0),
        ('use_cuda', True),
    )
    
    def __init__(self):
        self.d = self.datas[0]
        
        # Pre-extract ALL data as numpy arrays (one-time cost)
        self._extract_data_arrays()
        
        # Pre-compute ALL signals with CUDA acceleration
        self._precompute_all_signals()
        
        # Execution state
        self.bar_idx = 0
        self.position_entry_idx = -1
        self.last_exit_bar = -1e9
        
    def _extract_data_arrays(self):
        """Extract all data once at startup"""
        try:
            data_len = len(self.d)
            
            def safe_extract(line, length):
                try:
                    return np.array([getattr(self.d, line)[i] for i in range(length)])
                except:
                    return np.full(length, np.nan)
            
            # OHLCV
            self.close_array = safe_extract('close', data_len)
            self.high_array = safe_extract('high', data_len)
            self.low_array = safe_extract('low', data_len)
            self.volume_array = safe_extract('volume', data_len)
            
            # Features  
            self.ema1_fast_array = safe_extract('ema1_fast', data_len)
            self.ema1_slow_array = safe_extract('ema1_slow', data_len)
            self.ema15_fast_array = safe_extract('ema15_fast', data_len)
            self.ema15_slow_array = safe_extract('ema15_slow', data_len)
            self.ema60_fast_array = safe_extract('ema60_fast', data_len)
            self.ema60_slow_array = safe_extract('ema60_slow', data_len)
            self.atr5_array = safe_extract('atr5', data_len)
            self.dc_high5_prev_array = safe_extract('dc_high5_prev', data_len)
            self.dc_low5_prev_array = safe_extract('dc_low5_prev', data_len)
            self.adx_array = safe_extract('adx', data_len)
            self.plus_di_array = safe_extract('plus_di', data_len)
            self.minus_di_array = safe_extract('minus_di', data_len)
            self.rsi_array = safe_extract('rsi', data_len)
            self.vsma1_array = safe_extract('vsma1', data_len)
            
        except Exception as e:
            console.print(f"[red]Error extracting data arrays: {e}[/red]")
            # Fallback
            self.close_array = np.zeros(100)
            self.long_signals = np.zeros(100, dtype=bool)
            self.short_signals = np.zeros(100, dtype=bool)
        
    def _precompute_all_signals(self):
        """CUDA-accelerated signal precomputation"""
        try:
            # Check data size and CUDA availability
            use_cuda = (self.p.use_cuda and CUDA_AVAILABLE and 
                       len(self.close_array) >= MIN_CUDA_SIZE)
            
            if use_cuda:
                console.print("[green]Using CUDA for signal computation[/green]")
                self.long_signals, self.short_signals = compute_signals_cuda(
                    self.close_array, self.high_array, self.low_array, self.volume_array,
                    self.ema1_fast_array, self.ema1_slow_array, 
                    self.ema15_fast_array, self.ema15_slow_array,
                    self.ema60_fast_array, self.ema60_slow_array,
                    self.atr5_array, self.dc_high5_prev_array, self.dc_low5_prev_array,
                    self.adx_array, self.plus_di_array, self.minus_di_array, 
                    self.rsi_array, self.vsma1_array,
                    float(self.p.adxth), float(self.p.rsi_overheat), float(self.p.rsi_oversold),
                    float(self.p.max_stretch_atr_mult), float(self.p.volume_filter_mult),
                    bool(self.p.use_volume_filter), int(self.p.confirm_bars)
                )
                manage_gpu_memory()  # Clean up GPU memory
            else:
                if not CUDA_AVAILABLE:
                    console.print("[yellow]CUDA not available, using CPU[/yellow]")
                else:
                    console.print(f"[yellow]Dataset too small ({len(self.close_array)} < {MIN_CUDA_SIZE}), using CPU[/yellow]")
                    
                self.long_signals, self.short_signals = compute_all_signals_vectorized(
                    self.close_array, self.high_array, self.low_array, self.volume_array,
                    self.ema1_fast_array, self.ema1_slow_array, 
                    self.ema15_fast_array, self.ema15_slow_array,
                    self.ema60_fast_array, self.ema60_slow_array,
                    self.atr5_array, self.dc_high5_prev_array, self.dc_low5_prev_array,
                    self.adx_array, self.plus_di_array, self.minus_di_array, 
                    self.rsi_array, self.vsma1_array,
                    float(self.p.adxth), float(self.p.rsi_overheat), float(self.p.rsi_oversold),
                    float(self.p.max_stretch_atr_mult), float(self.p.volume_filter_mult),
                    bool(self.p.use_volume_filter), int(self.p.confirm_bars)
                )
                
        except Exception as e:
            console.print(f"[red]Signal computation failed: {e}, using fallback[/red]")
            # Fallback
            self.long_signals = np.zeros(len(self.close_array), dtype=bool)
            self.short_signals = np.zeros(len(self.close_array), dtype=bool)
        
    def next(self):
        """Ultra-fast execution - just lookup pre-computed decisions"""
        
        if not self.position:
            # Check cooldown
            if (self.bar_idx - self.last_exit_bar) < self.p.reentry_cooldown_bars:
                self.bar_idx += 1
                return
                
            # Entry logic - just array lookup
            if self.bar_idx < len(self.long_signals) and self.long_signals[self.bar_idx]:
                size = self._get_precomputed_size(self.bar_idx)
                if size > 0:
                    self.buy(size=size)
                    self.position_entry_idx = self.bar_idx
                    
            elif (self.p.can_short and 
                  self.bar_idx < len(self.short_signals) and 
                  self.short_signals[self.bar_idx]):
                size = self._get_precomputed_size(self.bar_idx)
                if size > 0:
                    self.sell(size=size)
                    self.position_entry_idx = self.bar_idx
        else:
            # Exit logic - simplified for speed
            if self._should_exit_now():
                self.close()
                self.last_exit_bar = self.bar_idx
                self.position_entry_idx = -1
                
        self.bar_idx += 1
    
    def _get_precomputed_size(self, idx):
        """Get pre-computed position size"""
        if idx >= len(self.atr5_array):
            return 0.0
        
        available_cash = self.broker.getcash()
        portfolio_value = self.broker.getvalue()
        
        # Risk-based sizing
        risk_amount = portfolio_value * self.p.risk_per_trade_pct
        stop_distance = self.p.atr_stop_mult * self.atr5_array[idx]
        
        if stop_distance <= 1e-8:
            return 0.0
        
        # Calculate size based on risk
        risk_based_size = risk_amount / stop_distance
        
        # Calculate maximum size based on available cash and leverage
        max_notional = available_cash * self.p.max_leverage
        current_price = self.d.close[0]
        max_size_by_cash = max_notional / current_price if current_price > 0 else 0
        
        # Take the smaller of the two
        raw_size = min(risk_based_size, max_size_by_cash)
        
        return self._round_qty(raw_size)
    
    def _should_exit_now(self):
        """Fast exit decision using pre-computed data"""
        if self.position_entry_idx < 0 or self.bar_idx >= len(self.close_array):
            return False
            
        # Time-based exit
        if (self.bar_idx - self.position_entry_idx) > self.p.time_limit_bars:
            return True
            
        # Simple profit target exit
        current_price = self.d.close[0]
        entry_price = self.close_array[self.position_entry_idx] if self.position_entry_idx < len(self.close_array) else current_price
        
        if self.position.size > 0:  # Long
            profit_target = entry_price * (1 + self.p.tp_r_multiple * 0.01)
            return current_price >= profit_target
        else:  # Short  
            profit_target = entry_price * (1 - self.p.tp_r_multiple * 0.01)
            return current_price <= profit_target
        
    def _round_qty(self, s):
        step = max(self.p.qty_step, 0.001)
        return max(0.0, math.floor(s / step) * step)

# --- Scoring ---
def score_sharpe_dd(strat, lam_dd=0.03):
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
        trades = ta.get("total", {}).get("total", 0)
    except Exception: 
        trades = 0
    score = sharpe - lam_dd * mdd
    return score, dict(mdd=mdd, sharpe=sharpe, trades=trades)

# --- Ultra-Fast Backtest Runner ---
def run_single_backtest_eval_cuda(df, spec, init_cash, commission, params, exchange=None, use_cuda=True):
    """CUDA-accelerated backtest runner"""
    cerebro = bt.Cerebro(
        oldbuysell=True,
        runonce=True,      # Critical
        stdstats=False,    # Disable stats collection
        exactbars=1,       # Memory optimization  
        live=False,        # No live checks
        writer=None,       # No writing
        tradehistory=False # Don't track individual trades
    )
    
    try:
        # Use cached features with CUDA support
        df_slice = slice_df_to_spec(df, spec)
        feat_df = get_or_build_features(df_slice, spec, params, use_cuda)
        
        feed = make_feature_feed(feat_df, name=f"{spec.symbol}")
        cerebro.adddata(feed)
        
        # Add metadata from params
        strategy_params = dict(params)
        m = meta_for(spec.symbol)
        strategy_params.update({
            'can_short': (str(exchange).lower() == "mexc") if exchange else False,
            'min_qty': m["min_qty"],
            'qty_step': m["qty_step"], 
            'price_tick': m["price_tick"],
            'use_cuda': use_cuda
        })
        
        # Use the CUDA-accelerated strategy
        cerebro.addstrategy(CUDAUltraFastVectorMACD_ADX, **strategy_params)
        
        # Minimal analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown") 
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        
        cerebro.broker.setcash(init_cash)
        cerebro.broker.setcommission(commission=commission)
        
        start_time = time.time()
        strats = cerebro.run(maxcpus=1)
        strat = strats[0]
        score, metrics = score_sharpe_dd(strat)
        final_value = cerebro.broker.getvalue()
        end_time = time.time()
        
        acceleration = "CUDA" if use_cuda and CUDA_AVAILABLE else "CPU"
        console.print(f"[cyan]{acceleration} backtest completed in {end_time - start_time:.3f}s[/cyan]")
        
        # Aggressive cleanup
        del cerebro, feed, strats
        gc.collect()
        manage_gpu_memory()
        
        return score, metrics, final_value
        
    except Exception as e:
        console.print(f"[red]CUDA backtest failed: {e}[/red]")
        manage_gpu_memory()
        return -999.0, {"error": 1.0}, 0.0

def run_backtest_with_plot(df, spec, params, init_cash=INIT_CASH, commission=COMMISSION_PER_TRANSACTION, exchange=None, plot=False, use_cuda=True):
    """Run backtest with plotting capability and CUDA support"""
    m = meta_for(spec.symbol)
    
    cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
    df_slice = slice_df_to_spec(df, spec)
    
    feat_df = get_or_build_features(df_slice, spec, dict(params), use_cuda)
    feed = make_feature_feed(feat_df, name=spec.symbol)
    cerebro.adddata(feed)

    strategy_params = dict(params)
    strategy_params.update(dict(
        can_short=(str(exchange).lower() == "mexc") if exchange else False,
        min_qty=m["min_qty"], 
        qty_step=m["qty_step"], 
        price_tick=m["price_tick"],
        use_cuda=use_cuda
    ))
    cerebro.addstrategy(CUDAUltraFastVectorMACD_ADX, **strategy_params)
    
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    acceleration = "CUDA" if use_cuda and CUDA_AVAILABLE else "CPU"
    console.print(f"Running {acceleration} backtest for {spec.symbol} ({spec.start_date} to {spec.end_date})...")
    
    start_time = time.time()
    results = cerebro.run(maxcpus=1)
    end_time = time.time()
    
    strat = results[0]
    
    mdd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0)
    sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    sharpe = float(sr) if sr is not None else 0.0
    ta = strat.analyzers.trades.get_analysis()
    trades = ta.get('total', {}).get('total', 0)
    wins = ta.get('won', {}).get('total', 0)
    
    console.print(f"\n--- {acceleration} Backtest Results for {spec.symbol} ---")
    console.print(f"Execution Time: {end_time - start_time:.3f}s")
    console.print(f"Trades: {trades}, Wins: {wins}, Win Rate: {(wins/trades*100 if trades>0 else 0):.1f}%")
    console.print(f"Sharpe: {sharpe:.3f}, MaxDD: {mdd:.2f}%, Final Value: ${cerebro.broker.getvalue():,.2f}")
    
    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')
    
    manage_gpu_memory()
    return dict(trades=trades, sharpe=sharpe, maxdd=mdd, value=cerebro.broker.getvalue())

# --- Optuna Integration ---
def make_objective_cuda(specs, df_map, init_cash, commission, exchange=None, min_trades_per_spec=10, use_cuda=True):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "tf5m_breakout_period": trial.suggest_int("tf5m_breakout_period", 20, 80, step=5),
            "adxth": trial.suggest_int("adxth", 15, 30),
            "confirm_bars": trial.suggest_int("confirm_bars", 1, 3),
            "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.0001, 0.001),
            "atr_stop_mult": trial.suggest_float("atr_stop_mult", 1.5, 4.0, step=0.25),
            "tp_r_multiple": trial.suggest_float("tp_r_multiple", 1.5, 5.0, step=0.5),
            "use_partial_exits": trial.suggest_categorical("use_partial_exits", [True, False]),
            "partial_exit_r": trial.suggest_float("partial_exit_r", 0.8, 2.5, step=0.1),
            "partial_exit_pct": trial.suggest_float("partial_exit_pct", 0.3, 0.7, step=0.1),
            "time_limit_bars": trial.suggest_int("time_limit_bars", 60, 360, step=30),
            "trail_mode": trial.suggest_categorical("trail_mode", ["chandelier", "donchian"]),
            "trail_atr_mult": trial.suggest_float("trail_atr_mult", 2.5, 5.0, step=0.25),
            "move_to_breakeven_R": trial.suggest_float("move_to_breakeven_R", 0.5, 1.5, step=0.25),
            "regime_mode_long": trial.suggest_categorical("regime_mode_long", ["ema", "price_vs_slow", "off"]),
            "regime_mode_short": trial.suggest_categorical("regime_mode_short", ["neutral", "ema", "off"]),
        }
        
        # Get default parameters and merge
        default_params = {
            'risk_per_trade_pct': 0.0001, 'max_leverage': 2.0, 'use_dynamic_exits': True,
            'tp_r_multiple': 2.0, 'use_partial_exits': True, 'partial_exit_r': 1.5,
            'partial_exit_pct': 0.5, 'time_limit_bars': 120, 'atr_stop_mult': 2.5,
            'use_trailing_stop': True, 'trail_mode': 'chandelier', 'trail_atr_mult': 4.0,
            'move_to_breakeven_R': 1.0, 'reentry_cooldown_bars': 5, 'min_qty': 0.0,
            'qty_step': 0.001, 'price_tick': 0.1, 'round_prices': True, 'can_short': False,
            'rsi_oversold': 25, 'use_htf': True, 'tf5m_breakout_period': 55,
            'tf15m_adx_period': 14, 'tf15m_ema_fast': 50, 'tf15m_ema_slow': 200,
            'tf60m_ema_fast': 50, 'tf60m_ema_slow': 200, 'ema_fast': 20, 'ema_slow': 50,
            'ema_trend': 200, 'atr_period': 14, 'rsi_overheat': 75, 'adxth': 20,
            'confirm_bars': 2, 'max_stretch_atr_mult': 1.0, 'ema_band_mult': 2.0,
            'donchian_trail_period': 55, 'close_based_stop': True, 'trail_update_every': 2,
            'max_bars_in_trade': 360, 'use_pyramiding': False, 'max_adds': 0,
            'add_cooldown': 20, 'add_atr_mult': 1.0, 'add_min_R': 1.0,
            'use_volume_filter': False, 'volume_filter_mult': 1.2, 'backtest': True,
            'debug': False, 'regime_mode_long': 'ema', 'regime_mode_short': 'neutral',
            'take_profit': 4.0, 'use_cuda': use_cuda
        }
        final_params = {**default_params, **params}

        scores = []
        for spec in specs:
            df = df_map[spec.symbol]
            score, metrics, _ = run_single_backtest_eval_cuda(df, spec, init_cash, commission, final_params, exchange, use_cuda)
            
            trades = metrics.get('trades', 0)
            if trades < min_trades_per_spec: 
                score = -5.0
            scores.append(score)

            # Store metrics as trial attributes
            for k, v in metrics.items(): 
                trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_{k}", v)

        return sum(scores) / len(scores) if scores else -999.0
    
    return objective

def optimize_parameters_cuda(specs, n_trials=100, n_jobs=1, init_cash=INIT_CASH, commission=COMMISSION_PER_TRANSACTION, 
                            storage_string=None, study_name="cuda_ultra_fast_study", seed=42, exchange="MEXC", use_cuda=True):
    
    storage = ensure_storage_or_sqlite(storage_string, study_name)
    df_map = preload_polars(specs)
    
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage, 
                               study_name=study_name, load_if_exists=True)
    
    objective = make_objective_cuda(specs, df_map, init_cash, commission, exchange=exchange, use_cuda=use_cuda)
    
    acceleration = "CUDA" if use_cuda and CUDA_AVAILABLE else "CPU"
    console.print(f"Starting {acceleration} optimization: {n_trials} trials, {n_jobs} jobs")
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=True)
    
    table = Table(title=f"Best Parameters ({acceleration})")
    table.add_column("Parameter")
    table.add_column("Value")
    for k, v in study.best_params.items(): 
        table.add_row(k, str(v))
    console.print(table)
    
    return study, study.best_params

def profile_backtest_cuda(df, spec, params, init_cash, commission, exchange, use_cuda=True):
    """Profile the CUDA backtest to find bottlenecks"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = run_single_backtest_eval_cuda(df, spec, init_cash, commission, params, exchange, use_cuda)
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
    return result

if __name__ == "__main__":
    try:
        # Test data specs
        holdout_start, holdout_end = "2023-06-12", "2025-05-31"
        holdout_spec = DataSpec("BTC", interval="1m", start_date=holdout_start, end_date=holdout_end)
        
        # Load data
        df_map = preload_polars([holdout_spec])
        df_hold = df_map["BTC"]
        
        # Default parameters for testing
        default_params = {
            'tf5m_breakout_period': 35, 'adxth': 20, 'confirm_bars': 2,
            'risk_per_trade_pct': 0.0005, 'atr_stop_mult': 2.5, 'tp_r_multiple': 3.0,
            'use_partial_exits': True, 'partial_exit_r': 1.5, 'partial_exit_pct': 0.5,
            'time_limit_bars': 180, 'trail_mode': 'chandelier', 'trail_atr_mult': 4.0,
            'move_to_breakeven_R': 1.0, 'regime_mode_long': 'ema', 'regime_mode_short': 'neutral',
            'max_leverage': 2.0, 'use_dynamic_exits': True, 'use_trailing_stop': True,
            'reentry_cooldown_bars': 5, 'rsi_oversold': 25, 'use_htf': True,
            'tf15m_adx_period': 14, 'tf15m_ema_fast': 50, 'tf15m_ema_slow': 200,
            'tf60m_ema_fast': 50, 'tf60m_ema_slow': 200, 'ema_fast': 20, 'ema_slow': 50,
            'ema_trend': 200, 'atr_period': 14, 'rsi_overheat': 75, 'max_stretch_atr_mult': 1.0,
            'ema_band_mult': 2.0, 'donchian_trail_period': 55, 'close_based_stop': True,
            'trail_update_every': 2, 'max_bars_in_trade': 360, 'use_pyramiding': False,
            'max_adds': 0, 'add_cooldown': 20, 'add_atr_mult': 1.0, 'add_min_R': 1.0,
            'use_volume_filter': False, 'volume_filter_mult': 1.2, 'backtest': True,
            'debug': False, 'take_profit': 4.0
        }
        
        console.print("[magenta]Running ultra-fast backtest with profiling...[/magenta]")
        
        # Profile the backtest
        result = profile_backtest_cuda(df_hold, holdout_spec, default_params, INIT_CASH, 
                                COMMISSION_PER_TRANSACTION, "MEXC")
        console.print(f"Profiled result: {result}")
        
        # Run with plotting
        console.print("[magenta]Running backtest with detailed output...[/magenta]")
        detailed_result = run_backtest_with_plot(df_hold, holdout_spec, default_params, 
                                                INIT_CASH, COMMISSION_PER_TRANSACTION, 
                                                "MEXC", plot=True)
        console.print(f"Detailed result: {detailed_result}")
        
    except Exception as e:
        console.print(f"An error occurred: {e}")
        traceback.print_exc()
    except KeyboardInterrupt:
        console.print("Process interrupted by user.")