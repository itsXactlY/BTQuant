# optuna_hyperopt_polars_vec.py
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
from rich.console import Console
from rich.table import Table
from pathlib import Path
import multiprocessing as mp
import hashlib

# optuna_hyperopt_polars_vec.py
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

# Allow plotting in interactive environments
import matplotlib

from dependencies.backtrader import order

console = Console()

# Your DB accessors (expected in your environment)
from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData
from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC

# --- Globals & Configuration ---
INIT_CASH = 100_000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"

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
    if df.is_empty(): return df
    start, end = spec.start_date, spec.end_date
    if start: df = df.filter(pl.col("TimestampStart") >= pl.lit(start).str.to_datetime())
    if end:   df = df.filter(pl.col("TimestampStart") <= pl.lit(end).str.to_datetime())
    return df

def build_cache(specs: List[DataSpec], force_refresh=True):
    console.print(f"[bold blue]Building cache for {len(set(s.symbol for s in specs))} symbols...[/bold blue]")
    t0 = time.time()
    preload_polars(specs, force_refresh=force_refresh)

# --- Polars Feature Engineering ---
def ema(expr: pl.Expr, n: int) -> pl.Expr: return expr.ewm_mean(alpha=2.0/(n+1.0), adjust=False)
def wilder(expr: pl.Expr, n: int) -> pl.Expr: return expr.ewm_mean(alpha=1.0/max(1,n), adjust=False)

def add_atr(lf: pl.LazyFrame, n: int, prefix: str) -> pl.LazyFrame:
    tr = pl.max_horizontal((pl.col("High")-pl.col("Low")), (pl.col("High")-pl.col("Close").shift(1)).abs(), (pl.col("Low")-pl.col("Close").shift(1)).abs())
    return lf.with_columns(wilder(tr, n).alias(f"{prefix}_atr"))

def add_rsi(lf: pl.LazyFrame, n: int = 14, col: str = "Close") -> pl.LazyFrame:
    diff = pl.col("Close") - pl.col("Close").shift(1)
    up, down = pl.when(diff > 0).then(diff).otherwise(0.0), pl.when(diff < 0).then(-diff).otherwise(0.0)
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

def build_feature_df(df_1m: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    p = params
    lf_1m = df_1m.lazy().sort("TimestampStart")
    lf_1m_feat = lf_1m.with_columns(
        ema(pl.col("Close"), p["ema_fast"]).alias("ema1_fast"),
        ema(pl.col("Close"), p["ema_slow"]).alias("ema1_slow"),
        pl.col("Volume").rolling_mean(20, min_samples=20).alias("vsma1")
    )
    lf_1m_feat = add_atr(lf_1m_feat, p["atr_period"], "atr1")
    lf_1m_feat = add_rsi(lf_1m_feat, 14)

    # 5m features
    lf_5m = resample(lf_1m, "5m")
    lf_5m = add_atr(lf_5m, p["atr_period"], "atr5")
    lf_5m = lf_5m.with_columns(
        pl.col("High").rolling_max(p["tf5m_breakout_period"]).shift(1).alias("dc_high5_prev"),
        pl.col("Low").rolling_min(p["tf5m_breakout_period"]).shift(1).alias("dc_low5_prev"),
        pl.col("Low").rolling_min(p["donchian_trail_period"]).alias("dc_exit5_low"),
        pl.col("High").rolling_max(p["donchian_trail_period"]).alias("dc_exit5_high"),
    )

    # 15m features
    lf_15m = resample(lf_1m, "15m").with_columns(
        ema(pl.col("Close"), p["tf15m_ema_fast"]).alias("ema15_fast"),
        ema(pl.col("Close"), p["tf15m_ema_slow"]).alias("ema15_slow")
    )
    lf_15m = add_adx(lf_15m, p["tf15m_adx_period"])

    # 60m features
    lf_60m = resample(lf_1m, "60m").with_columns(
        ema(pl.col("Close"), p["tf60m_ema_fast"]).alias("ema60_fast"),
        ema(pl.col("Close"), p["tf60m_ema_slow"]).alias("ema60_slow")
    )

    # Join all features, selecting only the columns we need
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

    # Add entry signals
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
         vol_ok & (pl.col("rsi") < p["rsi_overheat"])
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

    # Select only the columns we want to keep
    keep = [
        "TimestampStart", "Open", "High", "Low", "Close", "Volume",
        "ema1_fast", "ema1_slow", "atr1", "rsi", "vsma1",
        "atr5", "dc_high5_prev", "dc_low5_prev", "dc_exit5_low", "dc_exit5_high",
        "ema15_fast", "ema15_slow", "plus_di", "minus_di", "adx",
        "ema60_fast", "ema60_slow", "long_entry_signal", "short_entry_signal"
    ]
    return df.select([c for c in keep if c in df.columns])

# --- Backtrader Components ---
class FeatureData(bt.feeds.PolarsData):
    lines = ('ema1_fast','ema1_slow','atr1','rsi','vsma1','atr5','dc_high5_prev','dc_low5_prev','dc_exit5_low','dc_exit5_high','ema15_fast','ema15_slow','plus_di','minus_di','adx','ema60_fast','ema60_slow','long_entry_signal','short_entry_signal')
    params = dict(datetime='TimestampStart',open='Open',high='High',low='Low',close='Close',volume='Volume',openinterest=-1, **{k:k for k in lines})

def make_feature_feed(df_feat: pl.DataFrame, name: str) -> FeatureData:
    pdf = df_feat.to_pandas()
    feed = FeatureData(dataname=pdf)
    feed._name = f"{name}-features"
    return feed

class VectorMACD_ADX(bt.Strategy):
    params = (
        ('risk_per_trade_pct', 0.005),
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
        ('take_profit', 4.0),  # % based TP if not using dynamic exits
    )

    def __init__(self):
        self.d = self.datas[0]
        self._reset_position_state()
        self.active_orders = []  # Track active orders

    def _round_qty(self, s):
        step = self.p.qty_step if self.p.qty_step > 0 else 0.001
        q = math.floor(max(0.0, s) / step) * step
        if q == 0.0 and s > 0:
            q = step
        return 0.0 if self.p.min_qty > 0 and q < self.p.min_qty else q

    def _round_price(self, p):
        return round(p / self.p.price_tick) * self.p.price_tick if self.p.round_prices and self.p.price_tick > 0 else p

    def _risk_based_size(self, e, s):
        eq = self.broker.getvalue()
        risk = eq * self.p.risk_per_trade_pct
        dist = max(1e-8, abs(e - s))
        s_raw = min(risk / dist if dist > 0 else 0.0, (eq * self.p.max_leverage) / max(e, 1e-8))
        return self._round_qty(s_raw)

    def _reset_position_state(self):
        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.partial_exit_done = False
        self.trail_stop_active = False

    def _enter(self, direction: int):
        entry = self._round_price(self.d.close[0])
        stop_dist = self.p.atr_stop_mult * self.d.atr5[0]
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
            order = self.buy(size=size)
        else:
            order = self.sell(size=size)

        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.entry_bar = len(self)
        if direction > 0:
            self.run_high = self.d.high[0]
        else:
            self.run_low = self.d.low[0]

        self.active_orders.append(order)  # Track the order

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                console.print(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                console.print(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            console.print(f'Order {order.Status[order.status]}')
        if order in self.active_orders:
            self.active_orders.remove(order)

    def _R(self):
        if not self.position:
            return 0.0
        current_price = self.d.close[0]
        direction = 1 if self.position.size > 0 else -1
        return (current_price - self.init_stop) * direction / self.initial_risk

    def _update_trailing_stop(self):
        if not self.position:
            self.trail_stop = None
            return
        if self.position.size > 0:
            self.run_high = max(self.run_high or self.d.high[0], self.d.high[0])
        else:
            self.run_low = min(self.run_low or self.d.low[0], self.d.low[0])

        cand = None
        if self.p.trail_mode == "chandelier":
            cand = (self.run_high - self.p.trail_atr_mult * self.d.atr5[0]) if self.position.size > 0 else (self.run_low + self.p.trail_atr_mult * self.d.atr5[0])
        elif self.p.trail_mode == "donchian":
            cand = self.d.dc_exit5_low[0] if self.position.size > 0 else self.d.dc_exit5_high[0]

        if cand is not None:
            cand = max(cand, self.init_stop or -1e18) if self.position.size > 0 else min(cand, self.init_stop or 1e18)
            ae = None
            if self.active_orders:
                pending_orders = [o for o in self.active_orders if o.status in [order.Submitted, order.Accepted]]
                if pending_orders:
                    ae = sum(o.executed.price * o.executed.size for o in pending_orders) / sum(o.executed.size for o in pending_orders)
            if ae and self._R() >= self.p.move_to_breakeven_R:
                cand = max(cand, ae) if self.position.size > 0 else min(cand, ae)
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
            if self.d.long_entry_signal[0]:
                self._enter(1)
            elif self.p.can_short and self.d.short_entry_signal[0]:
                self._enter(-1)
        else:
            current_price = self.d.close[0]
            direction = 1 if self.position.size > 0 else -1

            if self.p.use_partial_exits and not self.partial_exit_done and self.partial_tp_price and (
                (direction > 0 and current_price >= self.partial_tp_price) or (direction < 0 and current_price <= self.partial_tp_price)
            ):
                partial_size = self._round_qty(self.position.size * self.p.partial_exit_pct)
                if partial_size > 0:
                    if direction > 0:
                        self.sell(size=partial_size)
                    else:
                        self.buy(size=partial_size)
                    self.partial_exit_done = True
                    self.trail_stop_active = True

            if (direction > 0 and current_price >= self.take_profit_price) or (direction < 0 and current_price <= self.take_profit_price):
                self.close()
                return
            if self.p.time_limit_bars > 0 and (len(self) - self.entry_bar) > self.p.time_limit_bars and not self.partial_exit_done:
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


# --- Scoring ---
def score_sharpe_dd(strat, lam_dd=0.03):
    try:
        sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr) if sr is not None and not math.isnan(sr) else 0.0
    except Exception: sharpe = 0.0
    try:
        mdd = float(strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0))
    except Exception: mdd = 0.0
    try:
        ta = strat.analyzers.trades.get_analysis()
        trades = ta.get("total", {}).get("total", 0)
    except Exception: trades = 0
    score = sharpe - lam_dd * mdd
    return score, dict(mdd=mdd, sharpe=sharpe, trades=trades)

# --- Backtest Runners ---
def run_backtest(strategy_class,df,spec,params,init_cash=INIT_CASH,commission=COMMISSION_PER_TRANSACTION,exchange=None,slippage_bps=5.0,plot=False,debug=False):
    m = meta_for(spec.symbol)
    min_qty, qty_step, price_tick = m["min_qty"], m["qty_step"], m["price_tick"]
    
    cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
    df_slice = slice_df_to_spec(df, spec)
    
    feat_df = build_feature_df(df_slice, dict(params))
    feed = make_feature_feed(feat_df, name=spec.symbol)
    cerebro.adddata(feed)

    strat_kwargs = dict(params)
    strat_kwargs.update(dict(
        backtest=True,
        can_short=(str(exchange).lower() == "mexc") if exchange else False,
        min_qty=min_qty, qty_step=qty_step, price_tick=price_tick, debug=debug
    ))
    cerebro.addstrategy(strategy_class, **strat_kwargs)
    
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    try: cerebro.broker.set_slippage_perc(perc=slippage_bps/10000.0)
    except: pass
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    console.print(f"Running backtest for {spec.symbol} ({spec.start_date} to {spec.end_date})...")
    results = cerebro.run(maxcpus=1)
    strat = results[0]
    
    mdd = strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0)
    sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    sharpe = float(sr) if sr is not None else 0.0
    ta = strat.analyzers.trades.get_analysis()
    trades = ta.get('total', {}).get('total', 0)
    wins = ta.get('won', {}).get('total', 0)
    
    console.print(f"\n--- Backtest Results for {spec.symbol} ---")
    console.print(f"Trades: {trades}, Wins: {wins}, Win Rate: {(wins/trades*100 if trades>0 else 0):.1f}%")
    console.print(f"Sharpe: {sharpe:.3f}, MaxDD: {mdd:.2f}%, Final Value: ${cerebro.broker.getvalue():,.2f}")
    
    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')
        
    return dict(trades=trades, sharpe=sharpe, maxdd=mdd, value=cerebro.broker.getvalue())

def run_single_backtest_eval(strategy_class,df,spec,init_cash,commission,params,exchange=None,slippage_bps=5.0,min_qty=None,qty_step=None,price_tick=None):
    m = meta_for(spec.symbol); min_qty=min_qty or m["min_qty"]; qty_step=qty_step or m["qty_step"]; price_tick=price_tick or m["price_tick"]
    cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
    df_slice = slice_df_to_spec(df, spec)

    feat_df = build_feature_df(df_slice, dict(params))
    feed = make_feature_feed(feat_df, name=f"{spec.symbol}")
    cerebro.adddata(feed)
    
    sp = dict(params)
    sp.update(dict(can_short=(str(exchange).lower()=="mexc") if exchange else False, min_qty=min_qty, qty_step=qty_step, price_tick=price_tick))
    cerebro.addstrategy(strategy_class, **sp)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    
    cerebro.broker.setcommission(commission=commission); cerebro.broker.setcash(init_cash)
    try: cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except: pass

    try: strats = cerebro.run(maxcpus=1); strat = strats[0]
    except Exception as e:
        console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]"); return -999.0, {"error": 1.0}, 0.0
    
    score, metrics = score_sharpe_dd(strat)
    final_value = cerebro.broker.getvalue()
    del cerebro, feed, strats, strat; gc.collect()
    return score, metrics, final_value

# --- Optuna ---
def make_objective(strategy_class,specs,df_map,init_cash,commission,exchange=None,min_trades_per_spec=10,use_median=False,asset_weights=None) -> Callable:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "tf5m_breakout_period": trial.suggest_int("tf5m_breakout_period", 20, 80, step=5),
            "adxth": trial.suggest_int("adxth", 20, 30),
            "confirm_bars": trial.suggest_int("confirm_bars", 1, 3),
            "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.002, 0.01),
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
        
        default_params = dict(strategy_class.params._getitems())
        final_params = {**default_params, **params}

        scores, weights = [], []
        for spec in specs:
            base_symbol = spec.symbol
            df = df_map[base_symbol]
            score, metrics, _ = run_single_backtest_eval(strategy_class,df,spec,init_cash,commission,final_params,exchange)
            
            trades = metrics.get('trades', 0)
            if trades < min_trades_per_spec: score = -5.0
            scores.append(score)
            weights.append((asset_weights or {}).get(base_symbol, 1.0))

            for k,v in metrics.items(): trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_{k}", v)

        if not scores: return -999.0
        if use_median: import numpy as np; return float(np.median(scores))
        else: wsum = sum(weights) or 1.0; return sum(s*w for s,w in zip(scores,weights))/wsum
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
    storage_string: str,  # <-- Expects a string
    study_name: str,
    trials_for_worker: int,
    init_cash: float,
    comm: float,
    exch: str,
    seed: int,
):
    try:
        storage = ensure_storage_or_sqlite(storage_string, study_name)  # <-- Create storage in worker
        optimize(
            strategy_class=strategy_class,
            specs=specs,
            n_trials=trials_for_worker,
            n_jobs=1,
            init_cash=init_cash,
            commission=comm,
            pruner="hyperband",
            storage_string=storage_string,  # <-- Pass the connection string
            study_name=study_name,
            seed=seed,
            exchange=exch,
        )
    except Exception as e:
        console.print(f"[red]Worker {worker_id} crashed: {e}[/red]")

def launch_multiprocess_optimize(
    strategy_class,
    specs,
    storage_string,  # <-- Pass the connection string, not the storage object
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
                storage_string,  # <-- Pass the connection string
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
    storage = ensure_storage_or_sqlite(storage_string, study_name)  # <-- Create storage here
    study = optuna.load_study(study_name=study_name, storage=storage)
    console.print(f"[green]Multiprocess optimize done. Best value: {study.best_value:.4f}[/green]")
    console.print(study.best_params)

# --- Main execution block ---
if __name__ == "__main__":
    try:
        bull_start, bull_end = "2020-09-28", "2021-05-31"
        bear_start, bear_end = "2022-05-28", "2023-06-23"
        holdout_start, holdout_end = "2023-06-12", "2025-05-31"
        train_specs = expand_specs_by_ranges([
            DataSpec("BTC", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
            DataSpec("ETH", "1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
        ])
        build_cache(train_specs, force_refresh=False)

        sanity_spec = DataSpec("BTC", "1m", start_date="2024-01-01", end_date="2024-01-15")
        df_map_sanity = preload_polars([sanity_spec])
        sanity_params = {
            'tf5m_breakout_period': 20,
            'adxth': 18,
            'confirm_bars': 1,
            'atr_stop_mult': 2.5,
            'trail_mode': 'chandelier',
            'trail_atr_mult': 4.0,
            'move_to_breakeven_R': 0.5,
            'risk_per_trade_pct': 0.002,
            'rsi_overheat': 80,
            'use_pyramiding': False,
            'use_volume_filter': False,
            'volume_filter_mult': 1.2,
            'ema_fast': 20,
            'ema_slow': 50,
            'ema_trend': 200,
            'atr_period': 14,
            'donchian_trail_period': 55,
            'ema_band_mult': 2.0,
            'trail_update_every': 2,
            'max_adds': 0,
            'add_cooldown': 20,
            'add_atr_mult': 1.0,
            'add_min_R': 1.0,
            'close_based_stop': True,
            'use_htf': True,
            'regime_mode_long': 'price_vs_slow',
            'regime_mode_short': 'neutral',
            'rsi_oversold': 25,
            'use_dynamic_exits': True,
            'tp_r_multiple': 2.0,
            'use_partial_exits': True,
            'partial_exit_r': 1.5,
            'partial_exit_pct': 0.5,
            'time_limit_bars': 120,
            'tf15m_ema_fast': 50,
            'tf15m_ema_slow': 200,
            'tf15m_adx_period': 14,
            'tf60m_ema_fast': 50,
            'tf60m_ema_slow': 200,
            'max_stretch_atr_mult': 1.0,
        }

        console.print("[yellow]Sanity backtest (BTC 2024-01-01..15) with vector features...[/yellow]")
        run_backtest(VectorMACD_ADX, df_map_sanity["BTC"], sanity_spec, sanity_params, init_cash=10000, exchange="MEXC", plot=True, debug=False)

        study_name = "Optimized_1m_MTF_VEC_V3"
        storage = ensure_storage_or_sqlite(MSSQL_ODBC, study_name)
        launch_multiprocess_optimize(
            strategy_class=VectorMACD_ADX,
            specs=train_specs,
            storage_string=MSSQL_ODBC,  # <-- Pass the storage object
            study_name=study_name, # <-- Pass the study name
            total_trials=200,       # <-- Use `trials` instead of `total_trials`
            workers=8,
            init_cash=INIT_CASH,
            comm=COMMISSION_PER_TRANSACTION,
            exch="MEXC",
            seed_base=42,          # <-- Use `seed` instead of `seed_base`
        )

        storage = ensure_storage_or_sqlite(MSSQL_ODBC, study_name)
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params = study.best_params
        holdout_spec = DataSpec("BNB", interval="1m", start_date=holdout_start, end_date=holdout_end)
        df_hold = preload_polars([holdout_spec])["BNB"]
        console.print("[magenta]Holdout backtest with best params on BNB...[/magenta]")
        run_backtest(VectorMACD_ADX, df_hold, holdout_spec, best_params, init_cash=10000, exchange="MEXC", plot=True, debug=False)
    except Exception as e:
        console.print(f"An error occurred: {e}")
        traceback.print_exc()
    except KeyboardInterrupt:
        console.print("Process interrupted by user.")