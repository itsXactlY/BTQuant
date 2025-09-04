# # optuna_hyperopt_polars_vec.py
# import os
# import gc
# import math
# import time
# import urllib.parse
# from dataclasses import dataclass
# from typing import Dict, Any, List, Tuple, Optional, Callable

# import backtrader as bt
# import optuna
# import polars as pl
# from rich.console import Console
# from rich.table import Table
# from pathlib import Path
# import multiprocessing as mp
# import hashlib

# # Headless plotting
# import matplotlib
# matplotlib.use('Agg')

# console = Console()

# # Your DB accessors (expected in your environment)
# from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData
# from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC

# INIT_CASH = 100_000.0
# COMMISSION_PER_TRANSACTION = 0.00075
# DEFAULT_COLLATERAL = "USDT"

# # Parquet cache dir (override with env BTQ_CACHE_DIR)
# CACHE_DIR = Path(os.getenv("BTQ_CACHE_DIR", ".btq_cache"))
# CACHE_DIR.mkdir(parents=True, exist_ok=True)

# # ---------- Exchange metadata (qty step / tick) ----------
# SYMBOL_META = {
#     "BTC": dict(qty_step=0.001, price_tick=0.1, min_qty=0.0),
#     "ETH": dict(qty_step=0.001, price_tick=0.01, min_qty=0.0),
#     # Add more if needed
# }
# def meta_for(symbol: str):
#     return SYMBOL_META.get(symbol.upper(), dict(qty_step=0.001, price_tick=0.1, min_qty=0.0))


# # --------------------- Data spec ---------------------
# @dataclass(frozen=True)
# class DataSpec:
#     symbol: str
#     interval: str
#     start_date: Optional[str] = None
#     end_date: Optional[str] = None
#     ranges: Optional[List[Tuple[str, str]]] = None
#     collateral: str = DEFAULT_COLLATERAL

#     def spec_id(self) -> str:
#         if self.ranges:
#             rstr = ",".join([f"{s}:{e}" for s, e in self.ranges])
#         else:
#             rstr = f"{self.start_date or ''}:{self.end_date or ''}"
#         raw = f"{self.symbol}|{self.interval}|{self.collateral}|{rstr}"
#         return hashlib.md5(raw.encode()).hexdigest()[:16]


# # --------------------- Storage helpers ---------------------
# def mssql_url_from_odbc(odbc_connection_string: str, database_override: Optional[str] = None) -> str:
#     parts = {}
#     for chunk in odbc_connection_string.split(";"):
#         if "=" in chunk and chunk.strip():
#             k, v = chunk.split("=", 1)
#             parts[k.strip().upper()] = v.strip()

#     server = parts.get("SERVER", "localhost")
#     database = database_override or parts.get("DATABASE", "OptunaBT")
#     uid = parts.get("UID", "SA")
#     pwd = parts.get("PWD", "")
#     driver_raw = parts.get("DRIVER", "{ODBC Driver 18 for SQL Server}").strip("{}")
#     driver = urllib.parse.quote_plus(driver_raw)

#     extra = {"driver": driver, "Encrypt": "yes", "TrustServerCertificate": parts.get("TRUSTSERVERCERTIFICATE", "yes")}
#     query = "&".join(f"{k}={v}" for k, v in extra.items() if v is not None)
#     encoded_pwd = urllib.parse.quote_plus(pwd)
#     return f"mssql+pyodbc://{uid}:{encoded_pwd}@{server}/{database}?{query}"

# def ensure_storage_or_sqlite(storage_string: Optional[str], study_name: str) -> optuna.storages.RDBStorage:
#     if storage_string is None:
#         sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
#         console.print(f"[yellow]No storage provided → using SQLite at {sqlite_path}[/yellow]")
#         return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")

#     if not storage_string.lower().startswith("mssql+pyodbc://"):
#         storage_url = mssql_url_from_odbc(storage_string)
#     else:
#         storage_url = storage_string

#     try:
#         console.print(f"Optuna storage URL: {storage_url.split('@')[0]}@***")
#         return optuna.storages.RDBStorage(
#             url=storage_url,
#             engine_kwargs={"pool_pre_ping": True, "pool_recycle": 300, "connect_args": {"timeout": 30}},
#         )
#     except Exception as e:
#         console.print(f"[red]RDB storage init failed: {e}[/red]")
#         sqlite_path = CACHE_DIR / f"optuna_{study_name}.db"
#         console.print(f"[yellow]Falling back to SQLite at {sqlite_path}[/yellow]")
#         return optuna.storages.RDBStorage(url=f"sqlite:///{sqlite_path}")


# # --------------------- Expand ranges (keep symbol) ---------------------
# def expand_specs_by_ranges(specs: List[DataSpec]) -> List[DataSpec]:
#     expanded: List[DataSpec] = []
#     for s in specs:
#         if s.ranges:
#             for rs, re in s.ranges:
#                 expanded.append(DataSpec(
#                     symbol=s.symbol, interval=s.interval,
#                     start_date=rs, end_date=re, ranges=None, collateral=s.collateral
#                 ))
#         else:
#             expanded.append(s)
#     return expanded


# # --------------------- Polars loader with cache ---------------------
# class PolarsDataLoader:
#     def __init__(self, cache_dir: Path = CACHE_DIR):
#         self.cache_dir = cache_dir
#         self.cache_dir.mkdir(parents=True, exist_ok=True)

#     def _cache_key(self, symbol: str, interval: str, collateral: str, start_date: Optional[str], end_date: Optional[str]) -> str:
#         raw = f"{symbol}|{interval}|{collateral}|{start_date or ''}:{end_date or ''}"
#         h = hashlib.md5(raw.encode()).hexdigest()[:12]
#         return f"{symbol}_{interval}_{collateral}_{h}.parquet"

#     def _cache_path(self, symbol: str, interval: str, collateral: str, start_date: Optional[str], end_date: Optional[str]) -> Path:
#         return self.cache_dir / self._cache_key(symbol, interval, collateral, start_date, end_date)

#     def _load_from_cache(self, path: Path) -> Optional[pl.DataFrame]:
#         if not path.exists():
#             return None
#         try:
#             return pl.scan_parquet(str(path)).collect()
#         except Exception as e:
#             console.print(f"[yellow]Cache read failed {path.name}: {e}[/yellow]")
#             return None

#     def _save_to_cache(self, path: Path, df: pl.DataFrame) -> None:
#         try:
#             df.write_parquet(str(path), compression="zstd", statistics=True)
#             console.print(f"[green]Cached -> {path}[/green]")
#         except Exception as e:
#             console.print(f"[yellow]Cache write failed {path.name}: {e}[/yellow]")

#     def _fetch_from_database(self, symbol: str, interval: str, collateral: str, start_date: Optional[str], end_date: Optional[str]) -> pl.DataFrame:
#         df = get_database_data(
#             ticker=symbol, start_date=start_date, end_date=end_date,
#             time_resolution=interval, pair=collateral
#         )
#         if df is None or df.is_empty():
#             raise ValueError(f"No data for {symbol} {interval} {start_date}->{end_date}")
#         return df.sort("TimestampStart")

#     def load_union_by_symbol(self, symbol: str, interval: str, collateral: str, start_end_pairs: List[Tuple[str, str]], use_cache: bool = True) -> pl.DataFrame:
#         if not start_end_pairs:
#             raise ValueError("start_end_pairs must not be empty")
#         starts = [s for s, _ in start_end_pairs if s]
#         ends = [e for _, e in start_end_pairs if e]
#         start = min(starts) if starts else None
#         end = max(ends) if ends else None

#         cache_path = self._cache_path(symbol, interval, collateral, start, end)
#         if use_cache:
#             df = self._load_from_cache(cache_path)
#             if df is not None:
#                 console.print(f"[cyan]Loaded from cache: {symbol} ({start}..{end})[/cyan]")
#                 return self._ensure_types(df)

#         df = self._fetch_from_database(symbol, interval, collateral, start, end)
#         df = self._ensure_types(df)
#         if use_cache:
#             self._save_to_cache(cache_path, df)
#         return df

#     @staticmethod
#     def _ensure_types(df: pl.DataFrame) -> pl.DataFrame:
#         if "TimestampStart" in df.columns and df.schema["TimestampStart"] != pl.Datetime:
#             try:
#                 df = df.with_columns(pl.col("TimestampStart").cast(pl.Datetime, strict=False))
#             except Exception:
#                 df = df.with_columns(pl.col("TimestampStart").str.strptime(pl.Datetime, strict=False))
#         return df


# def slice_df_to_spec(df: pl.DataFrame, spec: DataSpec) -> pl.DataFrame:
#     if df.is_empty():
#         return df

#     def _slice_one(s: Optional[str], e: Optional[str]) -> pl.DataFrame:
#         out = df
#         if s:
#             out = out.filter(pl.col("TimestampStart") >= pl.lit(s).str.strptime(pl.Datetime, strict=False))
#         if e:
#             out = out.filter(pl.col("TimestampStart") <= pl.lit(e).str.strptime(pl.Datetime, strict=False))
#         return out

#     if spec.ranges:
#         parts = []
#         for s, e in spec.ranges:
#             parts.append(_slice_one(s, e))
#         if not parts:
#             return df
#         return pl.concat(parts).sort("TimestampStart")

#     return _slice_one(spec.start_date, spec.end_date)


# # --------------------- Polars indicators + feature builder ---------------------
# def ema(expr: pl.Expr, n: int) -> pl.Expr:
#     alpha = 2.0 / (n + 1.0)
#     return expr.ewm_mean(alpha=alpha, adjust=False)

# def wilder(expr: pl.Expr, n: int) -> pl.Expr:
#     return expr.ewm_mean(alpha=1.0 / max(1, n), adjust=False)

# def add_atr(lf: pl.LazyFrame, n: int, prefix: str) -> pl.LazyFrame:
#     return (
#         lf.with_columns([
#             (pl.col("High") - pl.col("Low")).alias("_hl"),
#             (pl.col("High") - pl.col("Close").shift(1)).abs().alias("_hpc"),
#             (pl.col("Low") - pl.col("Close").shift(1)).abs().alias("_lpc"),
#         ])
#         .with_columns(pl.max_horizontal("_hl", "_hpc", "_lpc").alias(f"_{prefix}_tr"))
#         .with_columns(wilder(pl.col(f"_{prefix}_tr"), n).alias(f"{prefix}_atr"))
#         .drop(["_hl", "_hpc", "_lpc", f"_{prefix}_tr"])
#     )

# def add_rsi(lf: pl.LazyFrame, n: int = 14, col: str = "Close") -> pl.LazyFrame:
#     diff = pl.col(col) - pl.col(col).shift(1)
#     up = pl.when(diff > 0).then(diff).otherwise(0.0)
#     down = pl.when(diff < 0).then(-diff).otherwise(0.0)
#     return (
#         lf.with_columns([
#             wilder(up, n).alias("_avg_up"),
#             wilder(down, n).alias("_avg_down"),
#         ])
#         .with_columns([
#             (pl.col("_avg_up") / (pl.col("_avg_down") + 1e-12)).alias("_rs"),  # This line was missing!
#         ])
#         .with_columns([
#             (100 - 100 / (1 + pl.col("_rs"))).alias("rsi"),
#         ])
#         .drop(["_avg_up", "_avg_down", "_rs"])
#     )

# def add_adx(lf: pl.LazyFrame, n: int, col_high="High", col_low="Low", col_close="Close") -> pl.LazyFrame:
#     upmove = pl.col(col_high) - pl.col(col_high).shift(1)
#     downmove = pl.col(col_low).shift(1) - pl.col(col_low)
#     plus_dm = pl.when((upmove > downmove) & (upmove > 0)).then(upmove).otherwise(0.0)
#     minus_dm = pl.when((downmove > upmove) & (downmove > 0)).then(downmove).otherwise(0.0)

#     tr = pl.max_horizontal(
#         pl.col(col_high) - pl.col(col_low),
#         (pl.col(col_high) - pl.col(col_close).shift(1)).abs(),
#         (pl.col(col_low) - pl.col(col_close).shift(1)).abs(),
#     )
#     return (
#         lf.with_columns([
#             wilder(tr, n).alias("_tr_n"),
#             wilder(plus_dm, n).alias("_pdm_n"),
#             wilder(minus_dm, n).alias("_mdm_n"),
#         ])
#         .with_columns([
#             (100 * pl.col("_pdm_n") / (pl.col("_tr_n") + 1e-12)).alias("plus_di"),
#             (100 * pl.col("_mdm_n") / (pl.col("_tr_n") + 1e-12)).alias("minus_di"),
#         ])
#         .with_columns([
#             (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() /
#             (pl.col("plus_di") + pl.col("minus_di") + 1e-12)).alias("_dx")
#         ])
#         .with_columns([
#             wilder(pl.col("_dx"), n).alias("adx")
#         ])
#         .drop(["_tr_n", "_pdm_n", "_mdm_n", "_dx"])
#     )

# def resample(lf: pl.LazyFrame, every: str) -> pl.LazyFrame:
#     return (
#         lf.group_by_dynamic(index_column="TimestampStart", every=every, period=every, closed="right")
#         .agg([
#             pl.first("Open").alias("Open"),
#             pl.max("High").alias("High"),
#             pl.min("Low").alias("Low"),
#             pl.last("Close").alias("Close"),
#             pl.sum("Volume").alias("Volume"),
#         ])
#         .sort("TimestampStart")
#     )

# def build_feature_df(df_1m: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
#     # Unpack params with defaults
#     ema_fast = int(params.get("ema_fast", 20))
#     ema_slow = int(params.get("ema_slow", 50))
#     ema_trend = int(params.get("ema_trend", 200))
#     atr_period = int(params.get("atr_period", 14))
#     tf5m_breakout_period = int(params.get("tf5m_breakout_period", 55))
#     donchian_trail_period = int(params.get("donchian_trail_period", 55))
#     tf15m_adx_period = int(params.get("tf15m_adx_period", 14))
#     tf15m_ema_fast = int(params.get("tf15m_ema_fast", 50))
#     tf15m_ema_slow = int(params.get("tf15m_ema_slow", 200))
#     tf60m_ema_fast = int(params.get("tf60m_ema_fast", 50))
#     tf60m_ema_slow = int(params.get("tf60m_ema_slow", 200))
#     adxth = float(params.get("adxth", 20))
#     confirm_bars = int(params.get("confirm_bars", 2))
#     rsi_overheat = float(params.get("rsi_overheat", 75))
#     rsi_oversold = float(params.get("rsi_oversold", 25))
#     max_stretch_atr_mult = float(params.get("max_stretch_atr_mult", 1.0))
#     use_volume_filter = bool(params.get("use_volume_filter", False))
#     volume_filter_mult = float(params.get("volume_filter_mult", 1.2))
#     regime_mode_long = str(params.get("regime_mode_long", "ema"))
#     regime_mode_short = str(params.get("regime_mode_short", "neutral"))

#     lf_1m = pl.LazyFrame(df_1m).sort("TimestampStart")

#     # 1m features
#     lf_1m_feat = (
#         lf_1m.with_columns([
#             ema(pl.col("Close"), ema_fast).alias("ema1_fast"),
#             ema(pl.col("Close"), ema_slow).alias("ema1_slow"),
#             ema(pl.col("Close"), ema_trend).alias("ema1_trend"),
#             pl.col("Volume").rolling_mean(window_size=20, min_samples=20).alias("vsma1"),
#         ])
#     )
#     lf_1m_feat = add_atr(lf_1m_feat, n=atr_period, prefix="atr1")
#     lf_1m_feat = add_rsi(lf_1m_feat, n=14, col="Close")

#     # 5m
#     lf_5m = resample(lf_1m, every="5m")
#     lf_5m = add_atr(lf_5m, n=atr_period, prefix="atr5")
#     lf_5m = lf_5m.with_columns([
#         pl.col("High").rolling_max(window_size=tf5m_breakout_period).alias("dc_high5"),
#         pl.col("Low").rolling_min(window_size=tf5m_breakout_period).alias("dc_low5"),
#         pl.col("High").rolling_max(window_size=donchian_trail_period).alias("dc_exit5_high"),
#         pl.col("Low").rolling_min(window_size=donchian_trail_period).alias("dc_exit5_low"),
#     ]).with_columns([
#         pl.col("dc_high5").shift(1).alias("dc_high5_prev"),
#         pl.col("dc_low5").shift(1).alias("dc_low5_prev"),
#     ])

#     # 15m
#     lf_15m = resample(lf_1m, every="15m")
#     lf_15m = lf_15m.with_columns([
#         ema(pl.col("Close"), tf15m_ema_fast).alias("ema15_fast"),
#         ema(pl.col("Close"), tf15m_ema_slow).alias("ema15_slow"),
#     ])
#     lf_15m = add_adx(lf_15m, n=tf15m_adx_period, col_high="High", col_low="Low", col_close="Close")

#     # 60m
#     lf_60m = resample(lf_1m, every="60m")
#     lf_60m = lf_60m.with_columns([
#         ema(pl.col("Close"), tf60m_ema_fast).alias("ema60_fast"),
#         ema(pl.col("Close"), tf60m_ema_slow).alias("ema60_slow"),
#     ])

#     # Join to 1m
#     lf_all = (
#         lf_1m_feat.join_asof(
#             lf_5m.select(["TimestampStart", "atr5_atr", "dc_high5_prev", "dc_low5_prev", "dc_exit5_low", "dc_exit5_high"]),
#             on="TimestampStart", strategy="backward", suffix="_5m")
#         .join_asof(
#             lf_15m.select(["TimestampStart", "adx", "plus_di", "minus_di", "ema15_fast", "ema15_slow"]),
#             on="TimestampStart", strategy="backward", suffix="_15m")
#         .join_asof(
#             lf_60m.select(["TimestampStart", "ema60_fast", "ema60_slow"]),
#             on="TimestampStart", strategy="backward", suffix="_60m")
#     )

#     # Base booleans
#     lf_all = lf_all.with_columns([
#         pl.col("Close").rolling_min(window_size=confirm_bars).alias("lastN_min_close"),
#         pl.col("Close").rolling_max(window_size=confirm_bars).alias("lastN_max_close"),
#     ]).with_columns([
#         (pl.col("lastN_min_close") > pl.col("dc_high5_prev")).alias("breakout_up"),
#         (pl.col("lastN_max_close") < pl.col("dc_low5_prev")).alias("breakdown_down"),
#         ((pl.col("Close") - pl.col("dc_high5_prev")) > max_stretch_atr_mult * pl.col("atr5_atr")).alias("stretched_up"),
#         ((pl.col("dc_low5_prev") - pl.col("Close")) > max_stretch_atr_mult * pl.col("atr5_atr")).alias("stretched_down"),
#         (pl.col("Volume") > volume_filter_mult * pl.col("vsma1")).fill_null(False).alias("volume_ok"),
#         (pl.col("adx") >= adxth).alias("adx_ok"),
#         (pl.col("plus_di") > pl.col("minus_di")).alias("di_long"),
#         (pl.col("minus_di") > pl.col("plus_di")).alias("di_short"),
#         (pl.col("ema15_fast") > pl.col("ema15_slow")).alias("ema15_up"),
#         (pl.col("ema15_fast") < pl.col("ema15_slow")).alias("ema15_down"),
#         (pl.col("ema1_fast") > pl.col("ema1_slow")).alias("ema1_up"),
#         (pl.col("ema1_fast") < pl.col("ema1_slow")).alias("ema1_down"),
#         (pl.col("ema60_fast") > pl.col("ema60_slow")).alias("regime_ema_up"),
#         (pl.col("ema60_fast") < pl.col("ema60_slow")).alias("regime_ema_down"),
#         (pl.col("Close") > pl.col("ema60_slow")).alias("regime_px_above_slow"),
#         (pl.col("Close") < pl.col("ema60_slow")).alias("regime_px_below_slow"),
#         (pl.col("rsi") < rsi_overheat).fill_null(False).alias("rsi_not_hot"),
#         (pl.col("rsi") > rsi_oversold).fill_null(False).alias("rsi_not_cold"),
#     ])

#     # Composite entry signals (use params to embed regime mode and volume filter)
#     regime_long = (pl.col("regime_px_above_slow") if regime_mode_long == "price_vs_slow" else pl.col("regime_ema_up"))
#     regime_short = (pl.lit(True) if regime_mode_short == "neutral" else pl.col("regime_ema_down"))

#     vol_ok_expr = pl.when(pl.lit(use_volume_filter)).then(pl.col("volume_ok")).otherwise(True)

#     lf_all = lf_all.with_columns([
#         (regime_long & pl.col("adx_ok") & pl.col("di_long") & pl.col("ema15_up") & pl.col("ema1_up")
#         & pl.col("breakout_up") & (~pl.col("stretched_up")) & vol_ok_expr & pl.col("rsi_not_hot")).alias("long_entry_signal"),
#         (regime_short & pl.col("adx_ok") & pl.col("di_short") & pl.col("ema15_down") & pl.col("ema1_down")
#         & pl.col("breakdown_down") & (~pl.col("stretched_down")) & vol_ok_expr & pl.col("rsi_not_cold")).alias("short_entry_signal"),
#     ])

#     df = lf_all.collect()
#     # Rename ATR cols for clarity
#     df = df.rename({"atr1_atr": "atr1", "atr5_atr": "atr5"})
#     # Keep only needed columns
#     keep = [
#         "TimestampStart", "Open", "High", "Low", "Close", "Volume",
#         "ema1_fast", "ema1_slow", "ema1_trend", "atr1", "rsi", "vsma1",
#         "atr5", "dc_high5_prev", "dc_low5_prev", "dc_exit5_low", "dc_exit5_high",
#         "ema15_fast", "ema15_slow", "plus_di", "minus_di", "adx",
#         "ema60_fast", "ema60_slow",
#         "breakout_up", "breakdown_down", "long_entry_signal", "short_entry_signal"
#     ]
#     cols = [c for c in keep if c in df.columns]
#     return df.select(cols)


# # --------------------- Feature feed ---------------------
# class FeatureData(bt.feeds.PolarsData):
#     lines = (
#         'ema1_fast','ema1_slow','ema1_trend','atr1','rsi','vsma1',
#         'atr5','dc_high5_prev','dc_low5_prev','dc_exit5_low','dc_exit5_high',
#         'ema15_fast','ema15_slow','plus_di','minus_di','adx',
#         'ema60_fast','ema60_slow',
#         'breakout_up','breakdown_down','long_entry_signal','short_entry_signal'
#     )
#     params = dict(
#         datetime='TimestampStart', open='Open', high='High', low='Low', close='Close', volume='Volume', openinterest=-1,
#         ema1_fast='ema1_fast', ema1_slow='ema1_slow', ema1_trend='ema1_trend', atr1='atr1', rsi='rsi', vsma1='vsma1',
#         atr5='atr5', dc_high5_prev='dc_high5_prev', dc_low5_prev='dc_low5_prev', dc_exit5_low='dc_exit5_low', dc_exit5_high='dc_exit5_high',
#         ema15_fast='ema15_fast', ema15_slow='ema15_slow', plus_di='plus_di', minus_di='minus_di', adx='adx',
#         ema60_fast='ema60_fast', ema60_slow='ema60_slow',
#         breakout_up='breakout_up', breakdown_down='breakdown_down',
#         long_entry_signal='long_entry_signal', short_entry_signal='short_entry_signal',
#     )

# def make_feature_feed(df_feat: pl.DataFrame, name: str = "feat") -> FeatureData:
#     pdf = df_feat.to_pandas()
#     feed = FeatureData(dataname=pdf)
#     try:
#         feed._name = f"{name}-features"
#     except Exception:
#         pass
#     return feed


# # --------------------- Preload Polars per symbol ---------------------
# def preload_polars(specs: List[DataSpec], force_refresh: bool = False) -> Dict[str, pl.DataFrame]:
#     loader = PolarsDataLoader()
#     by_symbol: Dict[str, List[Tuple[str, str]]] = {}
#     for s in specs:
#         if s.ranges:
#             for rs, re in s.ranges:
#                 by_symbol.setdefault(s.symbol, []).append((rs, re))
#         else:
#             by_symbol.setdefault(s.symbol, []).append((s.start_date or "", s.end_date or ""))

#     df_map: Dict[str, pl.DataFrame] = {}
#     for symbol, pairs in by_symbol.items():
#         # Use first spec props (interval/collateral assumed same across specs set)
#         one = next(x for x in specs if x.symbol == symbol)
#         df = loader.load_union_by_symbol(symbol, one.interval, one.collateral, pairs, use_cache=not force_refresh)
#         df_map[symbol] = df
#     return df_map

# def build_cache(specs: List[DataSpec], force_refresh=True):
#     console.print(f"[bold blue]Building cache for {len(specs)} symbols...[/bold blue]")
#     t0 = time.time()
#     df_map = preload_polars(specs, force_refresh=force_refresh)
#     total_rows = sum(int(df.height) for df in df_map.values())
#     console.print(f"[green]✓ Cache build complete. {len(df_map)} symbols, {total_rows:,} rows. Took {time.time()-t0:.2f}s[/green]")


# # --------------------- Vectorized Strategy ---------------------
# class VectorMACD_ADX(bt.Strategy):
#     params = (
#         # Sizing
#         ('percent_sizer', 0.05),
#         ('risk_per_trade_pct', 0.0025),
#         ('max_leverage', 2.0),
#         ('min_qty', 0.0),
#         ('qty_step', 0.001),      # FIX: realistic default
#         ('price_tick', 0.1),      # FIX: realistic default
#         ('round_prices', True),

#         # Shorts
#         ('can_short', False),
#         ('regime_mode_long', 'ema'),      # 'ema' | 'price_vs_slow'
#         ('regime_mode_short', 'neutral'), # 'neutral' | 'ema'
#         ('rsi_oversold', 25),

#         # Feature periods (must match feature builder)
#         ('use_htf', True),
#         ('tf5m_breakout_period', 55),
#         ('tf15m_adx_period', 14),
#         ('tf15m_ema_fast', 50),
#         ('tf15m_ema_slow', 200),
#         ('tf60m_ema_fast', 50),
#         ('tf60m_ema_slow', 200),

#         # 1m baseline
#         ('ema_fast', 20),
#         ('ema_slow', 50),
#         ('ema_trend', 200),
#         ('atr_period', 14),
#         ('rsi_overheat', 75),

#         # Entries/Stops/TP
#         ('adxth', 20),
#         ('confirm_bars', 2),
#         ('max_stretch_atr_mult', 1.0),
#         ('atr_stop_mult', 2.5),
#         ('take_profit', 4.0),      # % per leg

#         # Trailing
#         ('use_trailing_stop', True),
#         ('trail_mode', 'chandelier'),  # chandelier | ema_band | donchian
#         ('trail_atr_mult', 4.0),
#         ('ema_band_mult', 2.0),
#         ('donchian_trail_period', 55),  # already baked in features
#         ('close_based_stop', True),
#         ('move_to_breakeven_R', 1.0),
#         ('trail_update_every', 2),
#         ('max_bars_in_trade', 6*60),
#         ('reentry_cooldown_bars', 5),

#         # Pyramiding
#         ('use_pyramiding', False),
#         ('max_adds', 0),
#         ('add_cooldown', 20),
#         ('add_atr_mult', 1.0),
#         ('add_min_R', 1.0),

#         # Volume filter
#         ('use_volume_filter', False),
#         ('volume_filter_mult', 1.2),

#         ('backtest', True),
#         ('debug', False),
#     )

#     def __init__(self):
#         self.d = self.datas[0]
#         # State
#         self.entry_bar = None
#         self.trail_stop = None
#         self.init_stop = None
#         self.initial_risk = None
#         self.run_high = None
#         self.run_low = None
#         self.last_trail_update = -10**9
#         self.n_adds = 0
#         self.last_add_bar = -10**9
#         self.last_exit_bar = -10**9

#         self.active_orders = []
#         self.entry_prices = []
#         self.sizes = []

#     # ---------- Sizing / Rounding ----------
#     def _round_qty(self, size: float) -> float:
#         step = float(self.p.qty_step) if self.p.qty_step else 0.001
#         if step <= 0:
#             step = 0.001
#         q = math.floor(max(0.0, float(size)) / step) * step  # round DOWN only
#         if self.p.min_qty and q < float(self.p.min_qty):
#             return 0.0
#         return q

#     def _round_price(self, price: float) -> float:
#         if not (self.p.round_prices and self.p.price_tick):
#             return float(price)
#         tick = float(self.p.price_tick)
#         if tick <= 0:
#             return float(price)
#         return round(price / tick) * tick

#     def _risk_based_size(self, entry: float, stop: float) -> float:
#         eq = float(self.broker.getvalue())
#         risk = eq * float(self.p.risk_per_trade_pct or 0.0)
#         dist = max(1e-8, abs(float(entry) - float(stop)))
#         s_risk = risk / dist if dist > 0 else 0.0
#         s_lev = (eq * float(self.p.max_leverage)) / max(float(entry), 1e-8)
#         s_raw = max(0.0, min(s_risk, s_lev))
#         s = self._round_qty(s_raw)
#         if s <= 0.0:
#             return 0.0
#         # Clamp to leverage post-rounding
#         max_units = (eq * float(self.p.max_leverage)) / max(float(entry), 1e-8)
#         if s > max_units:
#             step = float(self.p.qty_step) if self.p.qty_step else 0.001
#             s = math.floor(max_units / step) * step
#             if self.p.min_qty and s < float(self.p.min_qty):
#                 s = 0.0
#         return s

#     # ---------- Helpers ----------
#     def _avg_entry(self) -> Optional[float]:
#         if not self.active_orders:
#             return None
#         total = sum(o["size"] for o in self.active_orders)
#         if total <= 0:
#             return None
#         return sum(o["entry"] * o["size"] for o in self.active_orders) / total

#     def _R(self) -> float:
#         if self.initial_risk is None or self.initial_risk <= 0:
#             return 0.0
#         px = float(self.d.close[0])
#         ae = self._avg_entry() or px
#         if self.position.size > 0:
#             return (px - ae) / self.initial_risk
#         elif self.position.size < 0:
#             return (ae - px) / self.initial_risk
#         return 0.0

#     def _volume_filter_ok(self) -> bool:
#         if not self.p.use_volume_filter:
#             return True
#         base = float(self.d.vsma1[0]) if not math.isnan(float(self.d.vsma1[0])) else None
#         if base is None or base <= 0:
#             return False
#         return float(self.d.volume[0]) > self.p.volume_filter_mult * base

#     # ---------- Trailing ----------
#     def _update_trailing_stop(self):
#         if not self.position:
#             self.trail_stop = None
#             return

#         if self.position.size > 0:
#             self.run_high = max(self.run_high or float(self.d.high[0]), float(self.d.high[0]))
#         else:
#             self.run_low  = min(self.run_low or float(self.d.low[0]), float(self.d.low[0]))

#         if (len(self) - self.last_trail_update) < self.p.trail_update_every:
#             return

#         candidate = None
#         if self.p.trail_mode == "chandelier":
#             if self.position.size > 0:
#                 candidate = float((self.run_high or float(self.d.high[0])) - self.p.trail_atr_mult * float(self.d.atr5[0]))
#             else:
#                 candidate = float((self.run_low  or float(self.d.low[0]))  + self.p.trail_atr_mult * float(self.d.atr5[0]))
#         elif self.p.trail_mode == "ema_band":
#             if self.position.size > 0:
#                 candidate = float(self.d.ema1_fast[0] - self.p.ema_band_mult * float(self.d.atr1[0]))
#             else:
#                 candidate = float(self.d.ema1_fast[0] + self.p.ema_band_mult * float(self.d.atr1[0]))
#         elif self.p.trail_mode == "donchian":
#             candidate = float(self.d.dc_exit5_low[0] if self.position.size > 0 else self.d.dc_exit5_high[0])

#         if candidate is not None:
#             # lock to at least init stop
#             candidate = max(candidate, self.init_stop or -1e18) if self.position.size > 0 else min(candidate, self.init_stop or 1e18)
#             ae = self._avg_entry()
#             if ae and self._R() >= self.p.move_to_breakeven_R:
#                 candidate = max(candidate, ae) if self.position.size > 0 else min(candidate, ae)
#             self.trail_stop = candidate if self.trail_stop is None else (
#                 max(self.trail_stop, candidate) if self.position.size > 0 else min(self.trail_stop, candidate)
#             )
#             self.last_trail_update = len(self)

#     def _stop_hit(self) -> bool:
#         if self.trail_stop is None:
#             return False
#         if self.p.close_based_stop:
#             return (self.d.close[0] <= self.trail_stop) if self.position.size > 0 else (self.d.close[0] >= self.trail_stop)
#         else:
#             return (self.d.low[0] <= self.trail_stop) if self.position.size > 0 else (self.d.high[0] >= self.trail_stop)

#     # ---------- Conditions ----------
#     def regime_ok_long(self) -> bool:
#         if self.p.regime_mode_long == 'price_vs_slow':
#             return bool(self.d.close[0] > self.d.ema60_slow[0])
#         return bool(self.d.ema60_fast[0] > self.d.ema60_slow[0])

#     def regime_ok_short(self) -> bool:
#         if not self.p.can_short:
#             return False
#         if self.p.regime_mode_short == 'neutral':
#             return True
#         return bool(self.d.ema60_fast[0] < self.d.ema60_slow[0])

#     def trend_ok_long(self) -> bool:
#         return bool(self.d.adx[0] >= self.p.adxth and self.d.plus_di[0] > self.d.minus_di[0]
#                     and self.d.ema15_fast[0] > self.d.ema15_slow[0] and self.d.ema1_fast[0] > self.d.ema1_slow[0])

#     def trend_ok_short(self) -> bool:
#         return bool(self.d.adx[0] >= self.p.adxth and self.d.minus_di[0] > self.d.plus_di[0]
#                     and self.d.ema15_fast[0] < self.d.ema15_slow[0] and self.d.ema1_fast[0] < self.d.ema1_slow[0])

#     def breakout_up(self) -> bool:
#         if self.p.use_volume_filter and not self._volume_filter_ok():
#             return False
#         if float(self.d.rsi[0]) >= self.p.rsi_overheat:
#             return False
#         stretched = (float(self.d.close[0]) - float(self.d.dc_high5_prev[0])) > self.p.max_stretch_atr_mult * float(self.d.atr5[0])
#         return bool(self.d.breakout_up[0] and not stretched)

#     def breakdown_down(self) -> bool:
#         if self.p.use_volume_filter and not self._volume_filter_ok():
#             return False
#         if float(self.d.rsi[0]) <= self.p.rsi_oversold:
#             return False
#         stretched = (float(self.d.dc_low5_prev[0]) - float(self.d.close[0])) > self.p.max_stretch_atr_mult * float(self.d.atr5[0])
#         return bool(self.d.breakdown_down[0] and not stretched)

#     # ---------- Orders ----------
#     def _enter_long(self):
#         entry = float(self._round_price(self.d.close[0]))
#         init_stop = self._round_price(entry - self.p.atr_stop_mult * float(self.d.atr5[0]))
#         size = self._risk_based_size(entry, init_stop)
#         if size <= 0:
#             cash = float(self.broker.getcash())
#             budget = cash * float(self.p.percent_sizer or 0.0)
#             step = float(self.p.qty_step) if self.p.qty_step else 0.001
#             size = math.floor((budget / max(entry, 1e-8)) / step) * step
#             if self.p.min_qty and size < float(self.p.min_qty):
#                 size = 0.0
#             if size <= 0.0:
#                 return
#         tp = self._round_price(entry * (1 + self.p.take_profit / 100.0))

#         self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))
#         self.buy(size=size, exectype=bt.Order.Market)
#         self.init_stop = init_stop
#         self.trail_stop = init_stop
#         self.initial_risk = max(1e-8, entry - init_stop)
#         self.run_high = float(self.d.high[0]); self.run_low = None
#         self.entry_bar = len(self); self.last_add_bar = len(self)

#         if self.p.debug:
#             console.print(f"ENTER LONG {size} @ {entry} | SL={init_stop} | TP={tp}")

#     def _enter_short(self):
#         if not self.p.can_short:
#             return
#         entry = float(self._round_price(self.d.close[0]))
#         init_stop = self._round_price(entry + self.p.atr_stop_mult * float(self.d.atr5[0]))
#         size = self._risk_based_size(entry, init_stop)
#         if size <= 0:
#             cash = float(self.broker.getcash())
#             budget = cash * float(self.p.percent_sizer or 0.0)
#             step = float(self.p.qty_step) if self.p.qty_step else 0.001
#             size = math.floor((budget / max(entry, 1e-8)) / step) * step
#             if self.p.min_qty and size < float(self.p.min_qty):
#                 size = 0.0
#             if size <= 0.0:
#                 return
#         tp = self._round_price(entry * (1 - self.p.take_profit / 100.0))

#         self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))
#         self.sell(size=size, exectype=bt.Order.Market)
#         self.init_stop = init_stop
#         self.trail_stop = init_stop
#         self.initial_risk = max(1e-8, init_stop - entry)
#         self.run_low = float(self.d.low[0]); self.run_high = None
#         self.entry_bar = len(self); self.last_add_bar = len(self)

#         if self.p.debug:
#             console.print(f"ENTER SHORT {size} @ {entry} | SL={init_stop} | TP={tp}")

#     def _can_pyramid(self) -> bool:
#         if not (self.p.use_pyramiding and self.position):
#             return False
#         if self.n_adds >= self.p.max_adds:
#             return False
#         if (len(self) - self.last_add_bar) < self.p.add_cooldown:
#             return False
#         if self._R() < self.p.add_min_R:
#             return False
#         if self.position.size > 0:
#             return float(self.d.close[0]) >= ((self.run_high or float(self.d.high[0])) + self.p.add_atr_mult * float(self.d.atr5[0]))
#         else:
#             return float(self.d.close[0]) <= ((self.run_low  or float(self.d.low[0]))  - self.p.add_atr_mult * float(self.d.atr5[0]))

#     def _do_pyramid(self):
#         entry = float(self._round_price(self.d.close[0]))
#         stop = float(self.trail_stop or (entry - self.p.atr_stop_mult * float(self.d.atr5[0]) if self.position.size > 0
#                                          else entry + self.p.atr_stop_mult * float(self.d.atr5[0])))
#         size = self._round_qty(self._risk_based_size(entry, stop) / 2.0)
#         if size <= 0:
#             return
#         if self.position.size > 0:
#             tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
#             self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))
#             self.buy(size=size, exectype=bt.Order.Market)
#         else:
#             tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
#             self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))
#             self.sell(size=size, exectype=bt.Order.Market)
#         self.n_adds += 1
#         self.last_add_bar = len(self)
#         if self.p.debug:
#             console.print(f"PYRAMID add #{self.n_adds} {size} @ {entry}")

#     def _take_profits_and_trail(self):
#         if not self.active_orders:
#             return
#         current = float(self.d.close[0])
#         to_remove = []

#         # Per-leg TP
#         for idx, o in enumerate(self.active_orders):
#             if o["dir"] > 0 and current >= o["tp"]:
#                 self.sell(size=o["size"], exectype=bt.Order.Market)
#                 to_remove.append(idx)
#                 if self.p.debug:
#                     prof = (current / o["entry"] - 1) * 100
#                     console.print(f"TP LONG: -{o['size']} @ {current} (+{prof:.2f}%)")
#             elif o["dir"] < 0 and current <= o["tp"]:
#                 self.buy(size=o["size"], exectype=bt.Order.Market)
#                 to_remove.append(idx)
#                 if self.p.debug:
#                     prof = (1 - current / o["entry"]) * 100
#                     console.print(f"TP SHORT: +{o['size']} @ {current} (+{prof:.2f}%)")

#         # Trailing stop
#         if self.p.use_trailing_stop:
#             self._update_trailing_stop()
#             if self._stop_hit():
#                 qty = sum(o["size"] for o in self.active_orders if (o["dir"] > 0 and self.position.size > 0) or (o["dir"] < 0 and self.position.size < 0))
#                 if qty > 0:
#                     if self.position.size > 0:
#                         self.sell(size=qty, exectype=bt.Order.Market)
#                     else:
#                         self.buy(size=qty, exectype=bt.Order.Market)
#                 to_remove = list(range(len(self.active_orders)))

#         if to_remove:
#             for i in reversed(to_remove):
#                 self.active_orders.pop(i)
#                 self.sizes = [o["size"] for o in self.active_orders]
#                 self.entry_prices = [o["entry"] for o in self.active_orders]

#             if not self.active_orders:
#                 self._reset_position_state()

#     def _reset_position_state(self):
#         self.entry_bar = None
#         self.trail_stop = None
#         self.init_stop = None
#         self.initial_risk = None
#         self.run_high = None
#         self.run_low = None
#         self.last_trail_update = -10**9
#         self.n_adds = 0
#         self.last_add_bar = -10**9

#     def next(self):
#         # Flat -> Entry
#         if not self.position:
#             # Basic cooldown after exit
#             if (len(self) - self.last_exit_bar) < self.p.reentry_cooldown_bars:
#                 return

#             if self.p.use_htf:
#                 if self.regime_ok_long() and self.trend_ok_long() and self.breakout_up():
#                     self._enter_long()
#                 elif self.p.can_short and self.regime_ok_short() and self.trend_ok_short() and self.breakdown_down():
#                     self._enter_short()
#             else:
#                 # Fallback: only use 1m features and breakout
#                 if self.d.ema1_fast[0] > self.d.ema1_slow[0] and self.breakout_up():
#                     self._enter_long()
#                 elif self.p.can_short and self.d.ema1_fast[0] < self.d.ema1_slow[0] and self.breakdown_down():
#                     self._enter_short()
#         else:
#             # Manage position
#             self._take_profits_and_trail()
#             # Optional pyramiding
#             if self.p.use_pyramiding and self._can_pyramid():
#                 self._do_pyramid()

#     def notify_trade(self, trade):
#         if trade.isclosed:
#             self.last_exit_bar = len(self)

# # --------------------- Feed helpers ---------------------
# def make_feed_from_df(df: pl.DataFrame, spec: DataSpec) -> MSSQLData:
#     feed = MSSQLData(
#         dataname=df,
#         datetime="TimestampStart", open="Open", high="High", low="Low", close="Close", volume="Volume",
#         timeframe=bt.TimeFrame.Minutes, compression=1
#     )
#     try:
#         feed.p.timeframe = bt.TimeFrame.Minutes
#         feed.p.compression = 1
#     except Exception:
#         setattr(feed, "_timeframe", bt.TimeFrame.Minutes)
#         setattr(feed, "_compression", 1)
#     try:
#         sr = f"{spec.start_date or (spec.ranges[0][0] if spec.ranges else '')}"
#         er = f"{spec.end_date or (spec.ranges[-1][1] if spec.ranges else '')}"
#         feed._name = f"{spec.symbol}-{spec.interval} {sr}..{er}"
#         feed._dataname = f"{spec.symbol}{spec.collateral}"
#     except Exception:
#         pass
#     return feed

# def add_mtf_feeds(cerebro: bt.Cerebro, base_feed, add_5m=True, add_15m=True, add_60m=True):
#     if add_5m:
#         cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=5,  name='5m',  boundoff=1)
#     if add_15m:
#         cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=15, name='15m', boundoff=1)
#     if add_60m:
#         cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=60, name='60m', boundoff=1)


# # --------------------- Scoring ---------------------
# def score_sharpe_dd(strat, lam_dd=0.03):
#     try:
#         sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
#         sharpe = float(sr) if sr is not None and not math.isnan(sr) else 0.0
#     except Exception:
#         sharpe = 0.0
#     try:
#         draw = strat.analyzers.drawdown.get_analysis()
#         mdd = float(draw.get("max", {}).get("drawdown", 0.0))
#     except Exception:
#         mdd = 0.0
#     try:
#         ta = strat.analyzers.trades.get_analysis()
#         trades = ta.get("total", {}).get("total", 0) or 0
#         won = ta.get("won", {}).get("total", 0) or 0
#         lost = ta.get("lost", {}).get("total", 0) or 0
#         win_rate = (won / trades) if trades > 0 else 0.0
#     except Exception:
#         trades, win_rate = 0, 0.0
#     score = sharpe - lam_dd * (mdd / 100.0)
#     metrics = dict(mdd=mdd, sharpe=sharpe, trades=trades, win_rate=win_rate)
#     return score, metrics


# # --------------------- Quick backtest ---------------------
# def quick_backtest(
#     df: pl.DataFrame,
#     spec: DataSpec,
#     params: Dict[str, Any],
#     init_cash: float = INIT_CASH,
#     commission: float = COMMISSION_PER_TRANSACTION,
#     exchange: Optional[str] = None,
#     slippage_bps: float = 5.0,
#     min_qty: float = None, qty_step: float = None, price_tick: Optional[float] = None,
#     plot: bool = False,
#     debug: bool = False,
#     use_features: bool = True
# ):
#     m = meta_for(spec.symbol)
#     if min_qty is None: min_qty = m["min_qty"]
#     if qty_step is None: qty_step = m["qty_step"]
#     if price_tick is None: price_tick = m["price_tick"]

#     cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
#     df_slice = slice_df_to_spec(df, spec)

#     # if use_features:
#     feat_params = dict(params)  # copy
#     feat_df = build_feature_df(df_slice, feat_params)
#     feed = make_feature_feed(feat_df, name=f"{spec.symbol}")
#     cerebro.adddata(feed)
#     strat_class = VectorMACD_ADX  # Use the new vectorized strategy
#     # else:
#     #     # For non-feature mode, use the original strategy
#     #     feed = make_feed_from_df(df_slice, spec)
#     #     cerebro.adddata(feed)
#     #     add_mtf_feeds(cerebro, feed)
#     #     # Import and use the original strategy class
#     #     from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX4 as strat_class


#     sp = dict(params)
#     sp.update(dict(
#         can_short=(str(exchange).lower() == "mexc") if exchange else False,
#         min_qty=min_qty, qty_step=qty_step, price_tick=price_tick, debug=debug
#     ))

#     console.print("Strategy params:")
#     console.print(sp)

#     cerebro.addstrategy(strat_class, backtest=True, **sp)
#     cerebro.broker.setcash(init_cash)
#     cerebro.broker.setcommission(commission=commission)
#     try:
#         cerebro.broker.set_slippage_perc(perc=slippage_bps/10000.0)
#     except Exception:
#         pass

#     cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
#     cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
#     cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
#     cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
#     if debug:
#         cerebro.addobserver(bt.observers.Trades)

#     res = cerebro.run(maxcpus=1)
#     strat = res[0]

#     try:
#         maxdd = float(strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0))
#     except Exception:
#         maxdd = 0.0
#     try:
#         sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
#         sharpe = float(sr) if sr is not None else 0.0
#     except Exception:
#         sharpe = 0.0
#     ta = strat.analyzers.trades.get_analysis()
#     trades = ta.get('total', {}).get('total', 0) if ta else 0

#     console.print(f"Trades={trades}, Sharpe={sharpe:.3f}, MaxDD={maxdd:.2f}%, Value={cerebro.broker.getvalue():.2f}")
#     if plot:
#         cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')
#     return dict(trades=trades, sharpe=sharpe, maxdd=maxdd, value=cerebro.broker.getvalue())


# # --------------------- Single backtest eval (Optuna) ---------------------
# def run_single_backtest_eval(
#     strategy_class,
#     df: pl.DataFrame,
#     spec: DataSpec,
#     init_cash: float,
#     commission: float,
#     params: Dict[str, Any],
#     exchange: Optional[str] = None,
#     slippage_bps: float = 5.0,
#     min_qty: float = None,
#     qty_step: float = None,
#     price_tick: Optional[float] = None,
#     score_fn: Callable = score_sharpe_dd,
#     use_features: bool = True
# ) -> Tuple[float, Dict[str, float], float]:
#     m = meta_for(spec.symbol)
#     if min_qty is None: min_qty = m["min_qty"]
#     if qty_step is None: qty_step = m["qty_step"]
#     if price_tick is None: price_tick = m["price_tick"]

#     cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
#     df_slice = slice_df_to_spec(df, spec)

#     if use_features:
#         feat_df = build_feature_df(df_slice, dict(params))
#         feed = make_feature_feed(feat_df, name=f"{spec.symbol}")
#         cerebro.adddata(feed)
#         strat_class = VectorMACD_ADX
#     else:
#         feed = make_feed_from_df(df_slice, spec)
#         cerebro.adddata(feed)
#         add_mtf_feeds(cerebro, feed)
#         strat_class = strategy_class

#     sp = dict(params)
#     sp.update(dict(
#         can_short=(str(exchange).lower() == "mexc") if exchange else False,
#         min_qty=min_qty, qty_step=qty_step, price_tick=price_tick
#     ))
#     cerebro.addstrategy(strat_class, backtest=True, **sp)

#     cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Minutes, annualize=True)
#     cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
#     cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
#     cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

#     cerebro.broker.setcommission(commission=commission)
#     try:
#         cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
#     except Exception:
#         pass
#     cerebro.broker.setcash(init_cash)

#     try:
#         strats = cerebro.run(maxcpus=1)
#         strat = strats[0]
#     except Exception as e:
#         console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]")
#         return -999.0, {"error": 1.0}, 0.0

#     score, metrics = score_fn(strat)
#     final_value = cerebro.broker.getvalue()
#     del cerebro, feed, strats, strat
#     gc.collect()
#     return score, metrics, final_value


# # --------------------- Objective ---------------------
# def make_objective(
#     strategy_class,
#     specs: List[DataSpec],
#     df_map: Dict[str, pl.DataFrame],
#     init_cash: float,
#     commission: float,
#     exchange: Optional[str] = None,
#     slippage_bps: float = 5.0,
#     min_qty: float = None, qty_step: float = None, price_tick: Optional[float] = None,
#     scoring: str = "sharpe_dd",
#     asset_weights: Optional[Dict[str, float]] = None,
#     min_trades_per_spec: int = 10,
#     use_median: bool = False,
#     use_features: bool = True
# ) -> Callable[[optuna.Trial], float]:
#     def objective(trial: optuna.Trial) -> float:
#         # Params
#         params = {
#             "tf5m_breakout_period": trial.suggest_int("tf5m_breakout_period", 35, 75, step=5),
#             "adxth": trial.suggest_int("adxth", 18, 28),
#             "confirm_bars": trial.suggest_int("confirm_bars", 1, 3),
#             "max_stretch_atr_mult": trial.suggest_float("max_stretch_atr_mult", 0.8, 1.3, step=0.1),
#             "atr_stop_mult": trial.suggest_float("atr_stop_mult", 2.0, 3.0, step=0.25),
#             "trail_mode": trial.suggest_categorical("trail_mode", ["chandelier","ema_band","donchian"]),
#             "trail_atr_mult": trial.suggest_float("trail_atr_mult", 3.0, 5.0, step=0.25),
#             "ema_band_mult": trial.suggest_float("ema_band_mult", 1.6, 2.6, step=0.2),
#             "donchian_trail_period": trial.suggest_int("donchian_trail_period", 45, 75, step=5),
#             "move_to_breakeven_R": trial.suggest_float("move_to_breakeven_R", 0.5, 1.5, step=0.25),
#             "trail_update_every": trial.suggest_int("trail_update_every", 2, 6),
#             "ema_fast": trial.suggest_int("ema_fast", 18, 26),
#             "ema_slow": trial.suggest_int("ema_slow", 40, 60),
#             "ema_trend": trial.suggest_int("ema_trend", 150, 220),
#             "atr_period": trial.suggest_int("atr_period", 12, 18),
#             "take_profit": trial.suggest_float("take_profit", 3.0, 8.0, step=0.5),
#             "use_volume_filter": trial.suggest_categorical("use_volume_filter", [False, True]),
#             "volume_filter_mult": trial.suggest_float("volume_filter_mult", 1.1, 1.8, step=0.1),
#             "use_pyramiding": trial.suggest_categorical("use_pyramiding", [False, True]),
#             "max_adds": trial.suggest_int("max_adds", 0, 2),
#             "add_cooldown": trial.suggest_int("add_cooldown", 15, 40, step=5),
#             "add_atr_mult": trial.suggest_float("add_atr_mult", 0.5, 1.5, step=0.25),
#             "add_min_R": trial.suggest_float("add_min_R", 0.5, 1.5, step=0.25),
#             "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.001, 0.005, step=0.0005),
#             "close_based_stop": True,
#             "use_htf": True,
#             # Regime modes
#             "regime_mode_long": trial.suggest_categorical("regime_mode_long", ["ema", "price_vs_slow"]),
#             "regime_mode_short": trial.suggest_categorical("regime_mode_short", ["neutral", "ema"]),
#             "rsi_overheat": trial.suggest_int("rsi_overheat", 70, 85),
#             "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 35),
#         }

#         per_spec_scores = []
#         per_spec_weights = []

#         for spec in specs:
#             df = df_map[spec.symbol]
#             score, metrics, _ = run_single_backtest_eval(
#                 strategy_class=strategy_class,
#                 df=df, spec=spec,
#                 init_cash=init_cash, commission=commission,
#                 params=params,
#                 exchange=exchange, slippage_bps=slippage_bps,
#                 min_qty=min_qty, qty_step=qty_step, price_tick=price_tick,
#                 score_fn=score_sharpe_dd,
#                 use_features=use_features
#             )
#             trades = metrics.get('trades', 0)
#             if trades < min_trades_per_spec:
#                 score = -5.0
#             per_spec_scores.append(score)
#             w = (asset_weights or {}).get(spec.symbol, 1.0)
#             per_spec_weights.append(w)

#             trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_score", score)
#             trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_trades", trades)
#             trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_mdd", metrics.get("mdd", 0.0))
#             trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_sharpe", metrics.get("sharpe", 0.0))

#         if not per_spec_scores:
#             return -999.0

#         if use_median:
#             import numpy as np
#             agg = float(np.median(per_spec_scores))
#         else:
#             wsum = sum(per_spec_weights) or 1.0
#             agg = sum(s*w for s, w in zip(per_spec_scores, per_spec_weights)) / wsum

#         return agg

#     return objective


# # --------------------- Pruner ---------------------
# def make_pruner(num_steps: int, pruner: Optional[str], n_jobs: int):
#     if pruner == "hyperband":
#         return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max(1, num_steps), reduction_factor=3, bootstrap_count=max(2 * n_jobs, 8))
#     elif pruner == "sha":
#         return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
#     elif pruner == "median":
#         return optuna.pruners.MedianPruner(n_warmup_steps=1)
#     else:
#         return None


# # --------------------- Optimize (single-process) ---------------------
# def optimize(
#     strategy_class,
#     specs: List[DataSpec],
#     n_trials: int = 30,
#     n_jobs: int = 1,
#     init_cash: float = INIT_CASH,
#     commission: float = COMMISSION_PER_TRANSACTION,
#     pruner: Optional[str] = "hyperband",
#     storage_string: Optional[str] = None,
#     study_name: str = "strategy_opt",
#     seed: int = 42,
#     exchange: Optional[str] = None,
#     use_features: bool = True,
# ):
#     storage = ensure_storage_or_sqlite(storage_string, study_name)
#     console.print("[bold blue]Preloading data (cache-aware, union per symbol)...[/bold blue]")
#     df_map = preload_polars(specs)  # union per symbol
#     console.print(f"[green]✓ Loaded {len(df_map)} symbols[/green]")

#     pruner_obj = make_pruner(num_steps=len(specs), pruner=pruner, n_jobs=n_jobs)
#     sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True, warn_independent_sampling=False)

#     study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner_obj, storage=storage, study_name=study_name, load_if_exists=True)

#     m = meta_for(specs[0].symbol)
#     objective = make_objective(
#         strategy_class, specs, df_map, init_cash, commission,
#         exchange=exchange, slippage_bps=5,
#         min_qty=m["min_qty"], qty_step=m["qty_step"], price_tick=m["price_tick"],
#         scoring="sharpe_dd", use_features=use_features
#     )

#     console.print(f"Starting Optuna: trials={n_trials}, n_jobs={n_jobs}, pruner={pruner}, features={use_features}")
#     study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=True)
#     console.print(f"[bold green]Done. Best value: {study.best_value:.4f}[/bold green]")

#     table = Table(title="Best Parameters", show_lines=True)
#     table.add_column("Parameter"); table.add_column("Value")
#     for k, v in study.best_params.items():
#         table.add_row(k, str(v))
#     console.print(table)

#     if hasattr(study.best_trial, 'user_attrs'):
#         console.print("\n[bold cyan]Best Trial Performance (per spec):[/bold cyan]")
#         for attr, value in study.best_trial.user_attrs.items():
#             if any(tag in attr for tag in ["_score", "_trades", "_sharpe", "_mdd"]):
#                 console.print(f"{attr}: {value}")

#     return study, study.best_params


# # --------------------- Multi-process launcher ---------------------
# def _worker_optimize(
#     worker_id: int,
#     strategy_class,
#     specs: List[DataSpec],
#     storage_string: Optional[str],
#     study_name: str,
#     trials_for_worker: int,
#     init_cash: float,
#     commission: float,
#     exchange: Optional[str],
#     seed_base: int,
#     use_features: bool,
# ):
#     try:
#         optimize(
#             strategy_class=strategy_class,
#             specs=specs,
#             n_trials=trials_for_worker,
#             n_jobs=1,
#             init_cash=init_cash,
#             commission=commission,
#             pruner="hyperband",
#             storage_string=storage_string,
#             study_name=study_name,
#             seed=seed_base + worker_id,
#             exchange=exchange,
#             use_features=use_features,
#         )
#     except Exception as e:
#         console.print(f"[red]Worker {worker_id} crashed: {e}[/red]")

# def launch_multiprocess_optimize(
#     strategy_class,
#     specs: List[DataSpec],
#     storage_string: Optional[str],
#     study_name: str,
#     total_trials: int,
#     workers: int,
#     init_cash: float = INIT_CASH,
#     commission: float = COMMISSION_PER_TRANSACTION,
#     exchange: Optional[str] = None,
#     param_mode: str = "mtf",
#     seed_base: int = 42,
#     use_features: bool = True,
# ):
#     if workers < 1:
#         workers = 1
#     trials_per_worker = math.ceil(total_trials / workers)

#     console.print(f"[bold magenta]Launching {workers} workers @ {trials_per_worker} trials each (total≈{trials_per_worker*workers})[/bold magenta]")

#     # Warm cache
#     preload_polars(specs, force_refresh=False)

#     try:
#         mp.set_start_method("spawn", force=True)
#     except RuntimeError:
#         pass

#     procs: List[mp.Process] = []
#     for wid in range(workers):
#         p = mp.Process(
#             target=_worker_optimize,
#             args=(wid, strategy_class, specs, storage_string, study_name,
#                 trials_per_worker, init_cash, commission, exchange, seed_base, use_features)
#         )
#         p.start()
#         procs.append(p)
#     for p in procs:
#         p.join()

#     storage = ensure_storage_or_sqlite(storage_string, study_name)
#     study = optuna.load_study(study_name=study_name, storage=storage)
#     console.print(f"[green]Multiprocess optimize done. Best value: {study.best_value:.4f}[/green]")
#     console.print(study.best_params)


# # --------------------- Legacy helpers (bulk) ---------------------
# def fetch_single_data(coin, start_date, end_date, interval, collateral="USDT"):
#     try:
#         loader = PolarsDataLoader()
#         df = loader.load_union_by_symbol(coin, interval, collateral, [(start_date, end_date)], use_cache=True)
#         spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
#         feed = make_feed_from_df(slice_df_to_spec(df, spec), spec)
#         feed._dataname = f"{coin}{collateral}"
#         return feed
#     except Exception as e:
#         console.print(f"[red]Error fetching data for {coin}: {str(e)}[/red]")
#         return None

# def fetch_all_data_sequential(coins: List[str], start_date: str, end_date: str, interval: str, collateral: str = "USDT", progress_callback: Optional[Callable] = None) -> Dict[str, Optional[MSSQLData]]:
#     loader = PolarsDataLoader()
#     total = len(coins)
#     out: Dict[str, Optional[MSSQLData]] = {}
#     for i, coin in enumerate(coins, 1):
#         try:
#             df = loader.load_union_by_symbol(coin, interval, collateral, [(start_date, end_date)], use_cache=True)
#             spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
#             feed = make_feed_from_df(slice_df_to_spec(df, spec), spec)
#             feed._dataname = f"{coin}{collateral}"
#             out[coin] = feed
#             if progress_callback:
#                 progress_callback(i, total, coin, "success")
#         except Exception as e:
#             console.print(f"[red]Failed to load {coin}: {e}[/red]")
#             out[coin] = None
#             if progress_callback:
#                 progress_callback(i, total, coin, "failed")
#     return out


# def backtest(
#     strategy,
#     data=None,
#     coin=None,
#     start_date=None,
#     end_date=None,
#     interval=None,
#     collateral="USDT",
#     commission=COMMISSION_PER_TRANSACTION,
#     init_cash=INIT_CASH,
#     plot=True,
#     quantstats=False,
#     asset_name=None,
#     bulk=False,
#     show_progress=True,
#     exchange=None,
#     slippage_bps=5,
#     min_qty: float = None, qty_step: float = None, price_tick: Optional[float] = None,
#     params=None,
#     add_mtf_resamples=False,  # not used in feature mode
#     use_features=True,
#     **kwargs,
# ):
#     from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer

#     m = meta_for(coin or "BTC")
#     if min_qty is None: min_qty = m["min_qty"]
#     if qty_step is None: qty_step = m["qty_step"]
#     if price_tick is None: price_tick = m["price_tick"]

#     # ---------- Data ----------
#     if data is None:
#         if not all([coin, start_date, end_date, interval]):
#             raise ValueError("If data is not provided, coin, start_date, end_date, and interval are required")
#         if show_progress and not bulk:
#             console.print(f"🔄 [bold blue]Fetching data for {coin}...[/bold blue]")

#         loader = PolarsDataLoader()
#         df = loader.load_union_by_symbol(coin, interval, collateral, [(start_date, end_date)], use_cache=True)
#         spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
#         df_slice = slice_df_to_spec(df, spec)

#         cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)

#         if use_features:
#             feat_df = build_feature_df(df_slice, dict(params or {}))
#             data_feed = make_feature_feed(feat_df, name=f"{coin}")
#             cerebro.adddata(data_feed)
#             strat_class = VectorMACD_ADX
#         else:
#             data_feed = make_feed_from_df(df_slice, spec)
#             cerebro.adddata(data_feed)
#             if add_mtf_resamples:
#                 add_mtf_feeds(cerebro, data_feed)
#             # from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX4 as strat_class

#         if show_progress and not bulk:
#             console.print(f"✅ [bold green]Data ready for {coin}[/bold green]")
#     else:
#         from backtrader.feed import DataBase
#         if isinstance(data, DataBase):
#             data_feed = data
#             cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
#             cerebro.adddata(data_feed)
#             strat_class = VectorMACD_ADX # if use_features else strategy
#         else:
#             raise ValueError("Unsupported data type. Provide a Backtrader feed or leave data=None.")

#     # ---------- Strategy parameters ----------
#     strat_kwargs = {'backtest': True}
#     strat_kwargs['can_short'] = (str(exchange).lower() == "mexc") if exchange is not None else False
#     strat_kwargs['min_qty'] = min_qty
#     strat_kwargs['qty_step'] = qty_step
#     strat_kwargs['price_tick'] = price_tick

#     if isinstance(params, dict):
#         for k, v in params.items():
#             strat_kwargs[k] = v

#     for k, v in kwargs.items():
#         strat_kwargs[k] = v

#     cerebro.addstrategy(strat_class, **strat_kwargs)

#     # ---------- Broker & analyzers ----------
#     cerebro.broker.setcash(init_cash)
#     cerebro.broker.setcommission(commission=commission)
#     try:
#         cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
#     except Exception:
#         pass

#     cerebro.addanalyzer(TimeReturn, _name='time_return')
#     cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
#     cerebro.addanalyzer(DrawDown, _name='drawdown')
#     cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
#     cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

#     if show_progress and not bulk:
#         display_name = asset_name or (f"{coin}/{collateral}" if coin else "Asset")
#         console.print(f"🔄 [bold green]Running backtest for {display_name}...")

#     strategy_result = cerebro.run(maxcpus=1)[0]

#     # ---------- Metrics ----------
#     dd_info = strategy_result.analyzers.drawdown.get_analysis()
#     max_drawdown = dd_info.get('max', {}).get('drawdown', 0.0)
#     trade_analyzer = strategy_result.analyzers.trade_analyzer.get_analysis()

#     # ---------- Returns for QuantStats using Polars ----------
#     tr = strategy_result.analyzers.time_return.get_analysis()

#     try:
#         import pandas as pd
#         if tr:
#             dates = list(tr.keys())
#             returns_values = list(tr.values())
#             returns_df = pl.DataFrame({'date': dates, 'returns': returns_values})
#             if returns_df.dtypes[0] == pl.String:
#                 returns_df = returns_df.with_columns([pl.col('date').str.strptime(pl.Datetime, strict=False)]).sort('date').drop_nulls()
#             elif returns_df.dtypes[0] in [pl.Datetime, pl.Date]:
#                 returns_df = returns_df.sort('date').drop_nulls()
#             else:
#                 returns_df = returns_df.with_columns([pl.col('date').cast(pl.Datetime)]).sort('date').drop_nulls()
#             returns_pd = returns_df.to_pandas().set_index('date')['returns']
#         else:
#             returns_pd = pd.Series(dtype=float)
#     except Exception:
#         returns_pd = None

#     if quantstats and returns_pd is not None and not returns_pd.empty:
#         try:
#             try:
#                 import quantstats_lumi as quantstats
#             except Exception:
#                 console.print("[yellow]quantstats_lumi not found, falling back to quantstats[/yellow]")

#             from datetime import datetime
#             current_date = datetime.now().strftime("%Y-%m-%d")
#             current_time = datetime.now().strftime("%H-%M-%S")
#             if asset_name:
#                 coin_name = asset_name.replace('/', '_')
#             elif coin:
#                 coin_name = f"{coin}_{collateral}"
#             elif hasattr(data_feed, '_dataname'):
#                 coin_name = str(data_feed._dataname)
#             else:
#                 coin_name = "Unknown_Asset"
#             folder = os.path.join("QuantStats")
#             os.makedirs(folder, exist_ok=True)
#             filename = os.path.join(folder, f"{coin_name}_{current_date}_{current_time}.html")
#             quantstats.reports.html(returns_pd, output=filename, title=f'QuantStats_{coin_name}_{current_date}')
#         except Exception as e:
#             console.print(f"[yellow]QuantStats report generation failed: {e}[/yellow]")

#     if plot:
#         cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')

#     final_value = cerebro.broker.getvalue()
#     del cerebro, data_feed, strategy_result
#     gc.collect()

#     return final_value


# # --------------------- Example usage ---------------------
# if __name__ == "__main__":
#     try:
#         bull_start = "2020-09-28"; bull_end = "2021-05-31"
#         bear_start = "2022-05-28"; bear_end = "2023-06-23"

#         # 1) Training universe with regimes
#         base_specs = [
#             DataSpec("BTC", interval="1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
#             DataSpec("ETH", interval="1m", ranges=[(bull_start, bull_end), (bear_start, bear_end)]),
#         ]

#         # 2) Build cache once (union per symbol)
#         build_cache(base_specs, force_refresh=False)

#         # 3) Quick sanity backtest on a short window with vector features
#         sanity_spec = DataSpec("BTC", interval="1m", ranges=[("2024-01-01","2024-01-15")])
#         df_map = preload_polars([sanity_spec])
#         sanity_params = dict(
#             # regime_mode_long='price_vs_slow',  # Changed from 'use_regime_long'
#             tf60m_ema_fast=20, 
#             tf60m_ema_slow=60,
#             tf5m_breakout_period=20, 
#             confirm_bars=1, 
#             adxth=18,
#             atr_stop_mult=2.5, 
#             trail_mode='chandelier', 
#             trail_atr_mult=4.0,
#             move_to_breakeven_R=0.5, 
#             risk_per_trade_pct=0.002,
#             ema_fast=20, 
#             ema_slow=50, 
#             ema_trend=200, 
#             atr_period=14,
#             rsi_overheat=80, 
#             use_pyramiding=False, 
#             use_volume_filter=False,
#         )
#         console.print("[yellow]Quick sanity backtest (BTC 2024-01-01..2024-01-15) with vector features...[/yellow]")
#         quick_backtest(df_map["BTC"], sanity_spec, sanity_params, exchange="MEXC", plot=True, debug=True, use_features=True)

#         # 4) Multi-process optimization on vector features
#         study_name = "Optimized_1m_MTF_MACD_ADX_VEC"
#         specs = expand_specs_by_ranges(base_specs)
#         build_cache(specs, force_refresh=False)

#         launch_multiprocess_optimize(
#             strategy_class=VectorMACD_ADX,   # strategy class ignored in feature mode but kept for signature
#             specs=specs,
#             storage_string=MSSQL_ODBC,       # None => SQLite fallback
#             study_name=study_name,
#             total_trials=50,
#             workers=2,
#             init_cash=INIT_CASH,
#             commission=COMMISSION_PER_TRANSACTION,
#             exchange="MEXC",
#             seed_base=42,
#             use_features=True,
#         )

#         # 5) Holdout with best params
#         storage = ensure_storage_or_sqlite(MSSQL_ODBC, study_name)
#         study = optuna.load_study(study_name=study_name, storage=storage)
#         best_params = study.best_params

#         holdout_spec = DataSpec("BTC", interval="1m", ranges=[("2023-06-12", "2025-05-31")])
#         df_hold = preload_polars([holdout_spec])["BTC"]
#         console.print("[magenta]Holdout backtest with best params (vector features)...[/magenta]")
#         quick_backtest(df_hold, holdout_spec, best_params, exchange="MEXC", plot=False, debug=False, use_features=True)

#     except Exception as e:
#         console.print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()
#     except KeyboardInterrupt:
#         console.print("Process interrupted by user.")


'''BACKUP 2'''
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

console = Console()

# Your DB accessors (expected in your environment)
from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData
from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC

INIT_CASH = 100_000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"

# Parquet cache dir (override with env BTQ_CACHE_DIR)
CACHE_DIR = Path(os.getenv("BTQ_CACHE_DIR", ".btq_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Exchange metadata (qty step / tick) ----------
SYMBOL_META = {
    "BTC": dict(qty_step=0.001, price_tick=0.1, min_qty=0.0),
    "ETH": dict(qty_step=0.001, price_tick=0.01, min_qty=0.0),
    # Add more if needed
}
def meta_for(symbol: str):
    return SYMBOL_META.get(symbol.upper(), dict(qty_step=0.001, price_tick=0.1, min_qty=0.0))


# --------------------- Data spec ---------------------
@dataclass(frozen=True)
class DataSpec:
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL

    def spec_id(self) -> str:
        if self.ranges:
            rstr = ",".join([f"{s}:{e}" for s, e in self.ranges])
        else:
            rstr = f"{self.start_date or ''}:{self.end_date or ''}"
        raw = f"{self.symbol}|{self.interval}|{self.collateral}|{rstr}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# --------------------- Storage helpers ---------------------
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


# --------------------- Expand ranges (keep symbol) ---------------------
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


# --------------------- Polars loader with cache ---------------------
class PolarsDataLoader:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, symbol: str, interval: str, collateral: str, start_date: Optional[str], end_date: Optional[str]) -> str:
        raw = f"{symbol}|{interval}|{collateral}|{start_date or ''}:{end_date or ''}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{symbol}_{interval}_{collateral}_{h}.parquet"

    def _cache_path(self, symbol: str, interval: str, collateral: str, start_date: Optional[str], end_date: Optional[str]) -> Path:
        return self.cache_dir / self._cache_key(symbol, interval, collateral, start_date, end_date)

    def _load_from_cache(self, path: Path) -> Optional[pl.DataFrame]:
        if not path.exists():
            return None
        try:
            return pl.scan_parquet(str(path)).collect()
        except Exception as e:
            console.print(f"[yellow]Cache read failed {path.name}: {e}[/yellow]")
            return None

    def _save_to_cache(self, path: Path, df: pl.DataFrame) -> None:
        try:
            df.write_parquet(str(path), compression="zstd", statistics=True)
            console.print(f"[green]Cached -> {path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Cache write failed {path.name}: {e}[/yellow]")

    def _fetch_from_database(self, symbol: str, interval: str, collateral: str, start_date: Optional[str], end_date: Optional[str]) -> pl.DataFrame:
        df = get_database_data(
            ticker=symbol, start_date=start_date, end_date=end_date,
            time_resolution=interval, pair=collateral
        )
        if df is None or df.is_empty():
            raise ValueError(f"No data for {symbol} {interval} {start_date}->{end_date}")
        return df.sort("TimestampStart")

    def load_union_by_symbol(self, symbol: str, interval: str, collateral: str, start_end_pairs: List[Tuple[str, str]], use_cache: bool = True) -> pl.DataFrame:
        if not start_end_pairs:
            raise ValueError("start_end_pairs must not be empty")
        starts = [s for s, _ in start_end_pairs if s]
        ends = [e for _, e in start_end_pairs if e]
        start = min(starts) if starts else None
        end = max(ends) if ends else None

        cache_path = self._cache_path(symbol, interval, collateral, start, end)
        if use_cache:
            df = self._load_from_cache(cache_path)
            if df is not None:
                console.print(f"[cyan]Loaded from cache: {symbol} ({start}..{end})[/cyan]")
                return self._ensure_types(df)

        df = self._fetch_from_database(symbol, interval, collateral, start, end)
        df = self._ensure_types(df)
        if use_cache:
            self._save_to_cache(cache_path, df)
        return df

    @staticmethod
    def _ensure_types(df: pl.DataFrame) -> pl.DataFrame:
        if "TimestampStart" in df.columns and df.schema["TimestampStart"] != pl.Datetime:
            try:
                df = df.with_columns(pl.col("TimestampStart").cast(pl.Datetime, strict=False))
            except Exception:
                df = df.with_columns(pl.col("TimestampStart").str.strptime(pl.Datetime, strict=False))
        return df


def slice_df_to_spec(df: pl.DataFrame, spec: DataSpec) -> pl.DataFrame:
    if df.is_empty():
        return df

    def _slice_one(s: Optional[str], e: Optional[str]) -> pl.DataFrame:
        out = df
        if s:
            out = out.filter(pl.col("TimestampStart") >= pl.lit(s).str.strptime(pl.Datetime, strict=False))
        if e:
            out = out.filter(pl.col("TimestampStart") <= pl.lit(e).str.strptime(pl.Datetime, strict=False))
        return out

    if spec.ranges:
        parts = []
        for s, e in spec.ranges:
            parts.append(_slice_one(s, e))
        if not parts:
            return df
        return pl.concat(parts).sort("TimestampStart")

    return _slice_one(spec.start_date, spec.end_date)


# --------------------- Polars indicators + feature builder ---------------------
def ema(expr: pl.Expr, n: int) -> pl.Expr:
    alpha = 2.0 / (n + 1.0)
    return expr.ewm_mean(alpha=alpha, adjust=False)

def wilder(expr: pl.Expr, n: int) -> pl.Expr:
    return expr.ewm_mean(alpha=1.0 / max(1, n), adjust=False)

def add_atr(lf: pl.LazyFrame, n: int, prefix: str) -> pl.LazyFrame:
    return (
        lf.with_columns([
            (pl.col("High") - pl.col("Low")).alias("_hl"),
            (pl.col("High") - pl.col("Close").shift(1)).abs().alias("_hpc"),
            (pl.col("Low") - pl.col("Close").shift(1)).abs().alias("_lpc"),
        ])
        .with_columns(pl.max_horizontal("_hl", "_hpc", "_lpc").alias(f"_{prefix}_tr"))
        .with_columns(wilder(pl.col(f"_{prefix}_tr"), n).alias(f"{prefix}_atr"))
        .drop(["_hl", "_hpc", "_lpc", f"_{prefix}_tr"])
    )

def add_rsi(lf: pl.LazyFrame, n: int = 14, col: str = "Close") -> pl.LazyFrame:
    diff = pl.col(col) - pl.col(col).shift(1)
    up = pl.when(diff > 0).then(diff).otherwise(0.0)
    down = pl.when(diff < 0).then(-diff).otherwise(0.0)
    return (
        lf.with_columns([
            wilder(up, n).alias("_avg_up"),
            wilder(down, n).alias("_avg_down"),
        ])
        .with_columns([
            (pl.col("_avg_up") / (pl.col("_avg_down") + 1e-12)).alias("_rs"),  # This line was missing!
        ])
        .with_columns([
            (100 - 100 / (1 + pl.col("_rs"))).alias("rsi"),
        ])
        .drop(["_avg_up", "_avg_down", "_rs"])
    )

def add_adx(lf: pl.LazyFrame, n: int, col_high="High", col_low="Low", col_close="Close") -> pl.LazyFrame:
    upmove = pl.col(col_high) - pl.col(col_high).shift(1)
    downmove = pl.col(col_low).shift(1) - pl.col(col_low)
    plus_dm = pl.when((upmove > downmove) & (upmove > 0)).then(upmove).otherwise(0.0)
    minus_dm = pl.when((downmove > upmove) & (downmove > 0)).then(downmove).otherwise(0.0)

    tr = pl.max_horizontal(
        pl.col(col_high) - pl.col(col_low),
        (pl.col(col_high) - pl.col(col_close).shift(1)).abs(),
        (pl.col(col_low) - pl.col(col_close).shift(1)).abs(),
    )
    return (
        lf.with_columns([
            wilder(tr, n).alias("_tr_n"),
            wilder(plus_dm, n).alias("_pdm_n"),
            wilder(minus_dm, n).alias("_mdm_n"),
        ])
        .with_columns([
            (100 * pl.col("_pdm_n") / (pl.col("_tr_n") + 1e-12)).alias("plus_di"),
            (100 * pl.col("_mdm_n") / (pl.col("_tr_n") + 1e-12)).alias("minus_di"),
        ])
        .with_columns([
            (100 * (pl.col("plus_di") - pl.col("minus_di")).abs() /
            (pl.col("plus_di") + pl.col("minus_di") + 1e-12)).alias("_dx")
        ])
        .with_columns([
            wilder(pl.col("_dx"), n).alias("adx")
        ])
        .drop(["_tr_n", "_pdm_n", "_mdm_n", "_dx"])
    )

def resample(lf: pl.LazyFrame, every: str) -> pl.LazyFrame:
    return (
        lf.group_by_dynamic(index_column="TimestampStart", every=every, period=every, closed="right")
        .agg([
            pl.first("Open").alias("Open"),
            pl.max("High").alias("High"),
            pl.min("Low").alias("Low"),
            pl.last("Close").alias("Close"),
            pl.sum("Volume").alias("Volume"),
        ])
        .sort("TimestampStart")
    )

def build_feature_df(df_1m: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
    # Unpack params with defaults
    ema_fast = int(params.get("ema_fast", 20))
    ema_slow = int(params.get("ema_slow", 50))
    ema_trend = int(params.get("ema_trend", 200))
    atr_period = int(params.get("atr_period", 14))
    tf5m_breakout_period = int(params.get("tf5m_breakout_period", 55))
    donchian_trail_period = int(params.get("donchian_trail_period", 55))
    tf15m_adx_period = int(params.get("tf15m_adx_period", 14))
    tf15m_ema_fast = int(params.get("tf15m_ema_fast", 50))
    tf15m_ema_slow = int(params.get("tf15m_ema_slow", 200))
    tf60m_ema_fast = int(params.get("tf60m_ema_fast", 50))
    tf60m_ema_slow = int(params.get("tf60m_ema_slow", 200))
    adxth = float(params.get("adxth", 20))
    confirm_bars = int(params.get("confirm_bars", 2))
    rsi_overheat = float(params.get("rsi_overheat", 75))
    rsi_oversold = float(params.get("rsi_oversold", 25))
    max_stretch_atr_mult = float(params.get("max_stretch_atr_mult", 1.0))
    use_volume_filter = bool(params.get("use_volume_filter", False))
    volume_filter_mult = float(params.get("volume_filter_mult", 1.2))
    regime_mode_long = str(params.get("regime_mode_long", "ema"))
    regime_mode_short = str(params.get("regime_mode_short", "neutral"))

    lf_1m = pl.LazyFrame(df_1m).sort("TimestampStart")

    # 1m features
    lf_1m_feat = (
        lf_1m.with_columns([
            ema(pl.col("Close"), ema_fast).alias("ema1_fast"),
            ema(pl.col("Close"), ema_slow).alias("ema1_slow"),
            ema(pl.col("Close"), ema_trend).alias("ema1_trend"),
            pl.col("Volume").rolling_mean(window_size=20, min_samples=20).alias("vsma1"),
        ])
    )
    lf_1m_feat = add_atr(lf_1m_feat, n=atr_period, prefix="atr1")
    lf_1m_feat = add_rsi(lf_1m_feat, n=14, col="Close")

    # 5m
    lf_5m = resample(lf_1m, every="5m")
    lf_5m = add_atr(lf_5m, n=atr_period, prefix="atr5")
    lf_5m = lf_5m.with_columns([
        pl.col("High").rolling_max(window_size=tf5m_breakout_period).alias("dc_high5"),
        pl.col("Low").rolling_min(window_size=tf5m_breakout_period).alias("dc_low5"),
        pl.col("High").rolling_max(window_size=donchian_trail_period).alias("dc_exit5_high"),
        pl.col("Low").rolling_min(window_size=donchian_trail_period).alias("dc_exit5_low"),
    ]).with_columns([
        pl.col("dc_high5").shift(1).alias("dc_high5_prev"),
        pl.col("dc_low5").shift(1).alias("dc_low5_prev"),
    ])

    # 15m
    lf_15m = resample(lf_1m, every="15m").with_columns([
        ema(pl.col("Close"), tf15m_ema_fast).alias("ema15_fast"),
        ema(pl.col("Close"), tf15m_ema_slow).alias("ema15_slow"),
    ])
    lf_15m = add_adx(lf_15m, n=tf15m_adx_period, col_high="High", col_low="Low", col_close="Close")

    # 60m
    lf_60m = resample(lf_1m, every="60m").with_columns([
        ema(pl.col("Close"), tf60m_ema_fast).alias("ema60_fast"),
        ema(pl.col("Close"), tf60m_ema_slow).alias("ema60_slow"),
    ])

    # Join to 1m
    lf_all = (
        lf_1m_feat.join_asof(
            lf_5m.select(["TimestampStart", "atr5_atr", "dc_high5_prev", "dc_low5_prev", "dc_exit5_low", "dc_exit5_high"]),
            on="TimestampStart", strategy="backward", suffix="_5m")
        .join_asof(
            lf_15m.select(["TimestampStart", "adx", "plus_di", "minus_di", "ema15_fast", "ema15_slow"]),
            on="TimestampStart", strategy="backward", suffix="_15m")
        .join_asof(
            lf_60m.select(["TimestampStart", "ema60_fast", "ema60_slow"]),
            on="TimestampStart", strategy="backward", suffix="_60m")
    )

    # Base booleans
    lf_all = lf_all.with_columns([
        pl.col("Close").rolling_min(window_size=confirm_bars, min_samples=confirm_bars).alias("lastN_min_close"),
        pl.col("Close").rolling_max(window_size=confirm_bars, min_samples=confirm_bars).alias("lastN_max_close"),
    ]).with_columns([
        (pl.col("lastN_min_close") > pl.col("dc_high5_prev")).alias("breakout_up"),
        (pl.col("lastN_max_close") < pl.col("dc_low5_prev")).alias("breakdown_down"),
        ((pl.col("Close") - pl.col("dc_high5_prev")) > max_stretch_atr_mult * pl.col("atr5_atr")).alias("stretched_up"),
        ((pl.col("dc_low5_prev") - pl.col("Close")) > max_stretch_atr_mult * pl.col("atr5_atr")).alias("stretched_down"),
        (pl.col("Volume") > volume_filter_mult * pl.col("vsma1")).fill_null(False).alias("volume_ok"),
        (pl.col("adx") >= adxth).alias("adx_ok"),
        (pl.col("plus_di") > pl.col("minus_di")).alias("di_long"),
        (pl.col("minus_di") > pl.col("plus_di")).alias("di_short"),
        (pl.col("ema15_fast") > pl.col("ema15_slow")).alias("ema15_up"),
        (pl.col("ema15_fast") < pl.col("ema15_slow")).alias("ema15_down"),
        (pl.col("ema1_fast") > pl.col("ema1_slow")).alias("ema1_up"),
        (pl.col("ema1_fast") < pl.col("ema1_slow")).alias("ema1_down"),
        (pl.col("ema60_fast") > pl.col("ema60_slow")).alias("regime_ema_up"),
        (pl.col("ema60_fast") < pl.col("ema60_slow")).alias("regime_ema_down"),
        (pl.col("Close") > pl.col("ema60_slow")).alias("regime_px_above_slow"),
        (pl.col("Close") < pl.col("ema60_slow")).alias("regime_px_below_slow"),
        (pl.col("rsi") < rsi_overheat).fill_null(False).alias("rsi_not_hot"),
        (pl.col("rsi") > rsi_oversold).fill_null(False).alias("rsi_not_cold"),
    ])

    # Fill nulls on all boolean flags so Backtrader never sees None
    lf_all = lf_all.with_columns([
        pl.col("breakout_up").fill_null(False),
        pl.col("breakdown_down").fill_null(False),
        pl.col("stretched_up").fill_null(False),
        pl.col("stretched_down").fill_null(False),
        pl.col("adx_ok").fill_null(False),
        pl.col("di_long").fill_null(False),
        pl.col("di_short").fill_null(False),
        pl.col("ema15_up").fill_null(False),
        pl.col("ema15_down").fill_null(False),
        pl.col("ema1_up").fill_null(False),
        pl.col("ema1_down").fill_null(False),
        pl.col("regime_ema_up").fill_null(False),
        pl.col("regime_ema_down").fill_null(False),
        pl.col("regime_px_above_slow").fill_null(False),
        pl.col("regime_px_below_slow").fill_null(False),
    ])

    # Composite entry signals (respect params for regime mode and volume filter)
    regime_long = (pl.col("regime_px_above_slow") if regime_mode_long == "price_vs_slow" else pl.col("regime_ema_up"))
    regime_short = (pl.lit(True) if regime_mode_short == "neutral" else pl.col("regime_ema_down"))
    vol_ok_expr = pl.when(pl.lit(use_volume_filter)).then(pl.col("volume_ok")).otherwise(True)

    lf_all = lf_all.with_columns([
        (regime_long & pl.col("adx_ok") & pl.col("di_long") & pl.col("ema15_up") & pl.col("ema1_up")
         & pl.col("breakout_up") & (~pl.col("stretched_up")) & vol_ok_expr & pl.col("rsi_not_hot")).alias("long_entry_signal"),
        (regime_short & pl.col("adx_ok") & pl.col("di_short") & pl.col("ema15_down") & pl.col("ema1_down")
         & pl.col("breakdown_down") & (~pl.col("stretched_down")) & vol_ok_expr & pl.col("rsi_not_cold")).alias("short_entry_signal"),
    ]).with_columns([
        pl.col("long_entry_signal").fill_null(False),
        pl.col("short_entry_signal").fill_null(False),
    ])

    df = lf_all.collect()

    # Rename ATR cols for clarity
    df = df.rename({"atr1_atr": "atr1", "atr5_atr": "atr5"})

    # Drop warmup rows where HTF features are not available
    required_cols = [
        "atr5", "dc_high5_prev", "dc_low5_prev", "dc_exit5_low", "dc_exit5_high",
        "ema15_fast", "ema15_slow", "plus_di", "minus_di", "adx",
        "ema60_fast", "ema60_slow", "ema1_fast", "ema1_slow", "atr1", "rsi", "vsma1"
    ]
    df = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in required_cols]))

    # Cast signal lines to float to avoid float(None) issues in feeds
    df = df.with_columns([
        pl.col("breakout_up").cast(pl.Float64),
        pl.col("breakdown_down").cast(pl.Float64),
        pl.col("long_entry_signal").cast(pl.Float64),
        pl.col("short_entry_signal").cast(pl.Float64),
    ])

    # Keep only needed columns
    keep = [
        "TimestampStart", "Open", "High", "Low", "Close", "Volume",
        "ema1_fast", "ema1_slow", "ema1_trend", "atr1", "rsi", "vsma1",
        "atr5", "dc_high5_prev", "dc_low5_prev", "dc_exit5_low", "dc_exit5_high",
        "ema15_fast", "ema15_slow", "plus_di", "minus_di", "adx",
        "ema60_fast", "ema60_slow",
        "breakout_up", "breakdown_down", "long_entry_signal", "short_entry_signal"
    ]
    cols = [c for c in keep if c in df.columns]
    return df.select(cols)

# --------------------- Feature feed ---------------------
class FeatureData(bt.feeds.PolarsData):
    lines = (
        'ema1_fast','ema1_slow','ema1_trend','atr1','rsi','vsma1',
        'atr5','dc_high5_prev','dc_low5_prev','dc_exit5_low','dc_exit5_high',
        'ema15_fast','ema15_slow','plus_di','minus_di','adx',
        'ema60_fast','ema60_slow',
        'breakout_up','breakdown_down','long_entry_signal','short_entry_signal'
    )
    params = dict(
        datetime='TimestampStart', open='Open', high='High', low='Low', close='Close', volume='Volume', openinterest=-1,
        ema1_fast='ema1_fast', ema1_slow='ema1_slow', ema1_trend='ema1_trend', atr1='atr1', rsi='rsi', vsma1='vsma1',
        atr5='atr5', dc_high5_prev='dc_high5_prev', dc_low5_prev='dc_low5_prev', dc_exit5_low='dc_exit5_low', dc_exit5_high='dc_exit5_high',
        ema15_fast='ema15_fast', ema15_slow='ema15_slow', plus_di='plus_di', minus_di='minus_di', adx='adx',
        ema60_fast='ema60_fast', ema60_slow='ema60_slow',
        breakout_up='breakout_up', breakdown_down='breakdown_down',
        long_entry_signal='long_entry_signal', short_entry_signal='short_entry_signal',
    )

def make_feature_feed(df_feat: pl.DataFrame, name: str = "feat") -> FeatureData:
    pdf = df_feat.to_pandas()
    feed = FeatureData(dataname=pdf)
    try:
        feed._name = f"{name}-features"
    except Exception:
        pass
    return feed


# --------------------- Preload Polars per symbol ---------------------
def preload_polars(specs: List[DataSpec], force_refresh: bool = False) -> Dict[str, pl.DataFrame]:
    loader = PolarsDataLoader()
    by_symbol: Dict[str, List[Tuple[str, str]]] = {}
    for s in specs:
        if s.ranges:
            for rs, re in s.ranges:
                by_symbol.setdefault(s.symbol, []).append((rs, re))
        else:
            by_symbol.setdefault(s.symbol, []).append((s.start_date or "", s.end_date or ""))

    df_map: Dict[str, pl.DataFrame] = {}
    for symbol, pairs in by_symbol.items():
        # Use first spec props (interval/collateral assumed same across specs set)
        one = next(x for x in specs if x.symbol == symbol)
        df = loader.load_union_by_symbol(symbol, one.interval, one.collateral, pairs, use_cache=not force_refresh)
        df_map[symbol] = df
    return df_map

def build_cache(specs: List[DataSpec], force_refresh=True):
    console.print(f"[bold blue]Building cache for {len(specs)} symbols...[/bold blue]")
    t0 = time.time()
    df_map = preload_polars(specs, force_refresh=force_refresh)
    total_rows = sum(int(df.height) for df in df_map.values())
    console.print(f"[green]✓ Cache build complete. {len(df_map)} symbols, {total_rows:,} rows. Took {time.time()-t0:.2f}s[/green]")


# --------------------- Vectorized Strategy ---------------------
class VectorMACD_ADX(bt.Strategy):
    params = (
        # Sizing
        ('percent_sizer', 0.05),
        ('risk_per_trade_pct', 0.0025),
        ('max_leverage', 2.0),
        ('min_qty', 0.0),
        ('qty_step', 0.001),      # FIX: realistic default
        ('price_tick', 0.1),      # FIX: realistic default
        ('round_prices', True),

        # Shorts
        ('can_short', False),
        ('regime_mode_long', 'ema'),      # 'ema' | 'price_vs_slow'
        ('regime_mode_short', 'neutral'), # 'neutral' | 'ema'
        ('rsi_oversold', 25),

        # Feature periods (must match feature builder)
        ('use_htf', True),
        ('tf5m_breakout_period', 55),
        ('tf15m_adx_period', 14),
        ('tf15m_ema_fast', 50),
        ('tf15m_ema_slow', 200),
        ('tf60m_ema_fast', 50),
        ('tf60m_ema_slow', 200),

        # 1m baseline
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('ema_trend', 200),
        ('atr_period', 14),
        ('rsi_overheat', 75),

        # Entries/Stops/TP
        ('adxth', 20),
        ('confirm_bars', 2),
        ('max_stretch_atr_mult', 1.0),
        ('atr_stop_mult', 2.5),
        ('take_profit', 4.0),      # % per leg

        # Trailing
        ('use_trailing_stop', True),
        ('trail_mode', 'chandelier'),  # chandelier | ema_band | donchian
        ('trail_atr_mult', 4.0),
        ('ema_band_mult', 2.0),
        ('donchian_trail_period', 55),  # already baked in features
        ('close_based_stop', True),
        ('move_to_breakeven_R', 1.0),
        ('trail_update_every', 2),
        ('max_bars_in_trade', 6*60),
        ('reentry_cooldown_bars', 5),

        # Pyramiding
        ('use_pyramiding', False),
        ('max_adds', 0),
        ('add_cooldown', 20),
        ('add_atr_mult', 1.0),
        ('add_min_R', 1.0),

        # Volume filter
        ('use_volume_filter', False),
        ('volume_filter_mult', 1.2),

        ('backtest', True),
        ('debug', False),
    )

    def __init__(self):
        self.d = self.datas[0]
        # State
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

        self.active_orders = []
        self.entry_prices = []
        self.sizes = []

    # ---------- Sizing / Rounding ----------
    def _round_qty(self, size: float) -> float:
        step = float(self.p.qty_step) if self.p.qty_step else 0.001
        if step <= 0:
            step = 0.001
        q = math.floor(max(0.0, float(size)) / step) * step  # round DOWN only
        if self.p.min_qty and q < float(self.p.min_qty):
            return 0.0
        return q

    def _round_price(self, price: float) -> float:
        if not (self.p.round_prices and self.p.price_tick):
            return float(price)
        tick = float(self.p.price_tick)
        if tick <= 0:
            return float(price)
        return round(price / tick) * tick

    def _risk_based_size(self, entry: float, stop: float) -> float:
        eq = float(self.broker.getvalue())
        risk = eq * float(self.p.risk_per_trade_pct or 0.0)
        dist = max(1e-8, abs(float(entry) - float(stop)))
        s_risk = risk / dist if dist > 0 else 0.0
        s_lev = (eq * float(self.p.max_leverage)) / max(float(entry), 1e-8)
        s_raw = max(0.0, min(s_risk, s_lev))
        s = self._round_qty(s_raw)
        if s <= 0.0:
            return 0.0
        # Clamp to leverage post-rounding
        max_units = (eq * float(self.p.max_leverage)) / max(float(entry), 1e-8)
        if s > max_units:
            step = float(self.p.qty_step) if self.p.qty_step else 0.001
            s = math.floor(max_units / step) * step
            if self.p.min_qty and s < float(self.p.min_qty):
                s = 0.0
        return s

    # ---------- Helpers ----------
    def _avg_entry(self) -> Optional[float]:
        if not self.active_orders:
            return None
        total = sum(o["size"] for o in self.active_orders)
        if total <= 0:
            return None
        return sum(o["entry"] * o["size"] for o in self.active_orders) / total

    def _R(self) -> float:
        if self.initial_risk is None or self.initial_risk <= 0:
            return 0.0
        px = float(self.d.close[0])
        ae = self._avg_entry() or px
        if self.position.size > 0:
            return (px - ae) / self.initial_risk
        elif self.position.size < 0:
            return (ae - px) / self.initial_risk
        return 0.0

    def _volume_filter_ok(self) -> bool:
        if not self.p.use_volume_filter:
            return True
        base = float(self.d.vsma1[0]) if not math.isnan(float(self.d.vsma1[0])) else None
        if base is None or base <= 0:
            return False
        return float(self.d.volume[0]) > self.p.volume_filter_mult * base

    # ---------- Trailing ----------
    def _update_trailing_stop(self):
        if not self.position:
            self.trail_stop = None
            return

        if self.position.size > 0:
            self.run_high = max(self.run_high or float(self.d.high[0]), float(self.d.high[0]))
        else:
            self.run_low  = min(self.run_low or float(self.d.low[0]), float(self.d.low[0]))

        if (len(self) - self.last_trail_update) < self.p.trail_update_every:
            return

        candidate = None
        if self.p.trail_mode == "chandelier":
            if self.position.size > 0:
                candidate = float((self.run_high or float(self.d.high[0])) - self.p.trail_atr_mult * float(self.d.atr5[0]))
            else:
                candidate = float((self.run_low  or float(self.d.low[0]))  + self.p.trail_atr_mult * float(self.d.atr5[0]))
        elif self.p.trail_mode == "ema_band":
            if self.position.size > 0:
                candidate = float(self.d.ema1_fast[0] - self.p.ema_band_mult * float(self.d.atr1[0]))
            else:
                candidate = float(self.d.ema1_fast[0] + self.p.ema_band_mult * float(self.d.atr1[0]))
        elif self.p.trail_mode == "donchian":
            candidate = float(self.d.dc_exit5_low[0] if self.position.size > 0 else self.d.dc_exit5_high[0])

        if candidate is not None:
            # lock to at least init stop
            candidate = max(candidate, self.init_stop or -1e18) if self.position.size > 0 else min(candidate, self.init_stop or 1e18)
            ae = self._avg_entry()
            if ae and self._R() >= self.p.move_to_breakeven_R:
                candidate = max(candidate, ae) if self.position.size > 0 else min(candidate, ae)
            self.trail_stop = candidate if self.trail_stop is None else (
                max(self.trail_stop, candidate) if self.position.size > 0 else min(self.trail_stop, candidate)
            )
            self.last_trail_update = len(self)

    def _stop_hit(self) -> bool:
        if self.trail_stop is None:
            return False
        if self.p.close_based_stop:
            return (self.d.close[0] <= self.trail_stop) if self.position.size > 0 else (self.d.close[0] >= self.trail_stop)
        else:
            return (self.d.low[0] <= self.trail_stop) if self.position.size > 0 else (self.d.high[0] >= self.trail_stop)

    # ---------- Conditions ----------
    def regime_ok_long(self) -> bool:
        if self.p.regime_mode_long == 'price_vs_slow':
            return bool(self.d.close[0] > self.d.ema60_slow[0])
        return bool(self.d.ema60_fast[0] > self.d.ema60_slow[0])

    def regime_ok_short(self) -> bool:
        if not self.p.can_short:
            return False
        if self.p.regime_mode_short == 'neutral':
            return True
        return bool(self.d.ema60_fast[0] < self.d.ema60_slow[0])

    def trend_ok_long(self) -> bool:
        return bool(self.d.adx[0] >= self.p.adxth and self.d.plus_di[0] > self.d.minus_di[0]
                    and self.d.ema15_fast[0] > self.d.ema15_slow[0] and self.d.ema1_fast[0] > self.d.ema1_slow[0])

    def trend_ok_short(self) -> bool:
        return bool(self.d.adx[0] >= self.p.adxth and self.d.minus_di[0] > self.d.plus_di[0]
                    and self.d.ema15_fast[0] < self.d.ema15_slow[0] and self.d.ema1_fast[0] < self.d.ema1_slow[0])

    def breakout_up(self) -> bool:
        if self.p.use_volume_filter and not self._volume_filter_ok():
            return False
        if float(self.d.rsi[0]) >= self.p.rsi_overheat:
            return False
        stretched = (float(self.d.close[0]) - float(self.d.dc_high5_prev[0])) > self.p.max_stretch_atr_mult * float(self.d.atr5[0])
        return bool(self.d.breakout_up[0] and not stretched)

    def breakdown_down(self) -> bool:
        if self.p.use_volume_filter and not self._volume_filter_ok():
            return False
        if float(self.d.rsi[0]) <= self.p.rsi_oversold:
            return False
        stretched = (float(self.d.dc_low5_prev[0]) - float(self.d.close[0])) > self.p.max_stretch_atr_mult * float(self.d.atr5[0])
        return bool(self.d.breakdown_down[0] and not stretched)

    # ---------- Orders ----------
    def _enter_long(self):
        entry = float(self._round_price(self.d.close[0]))
        init_stop = self._round_price(entry - self.p.atr_stop_mult * float(self.d.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            cash = float(self.broker.getcash())
            budget = cash * float(self.p.percent_sizer or 0.0)
            step = float(self.p.qty_step) if self.p.qty_step else 0.001
            size = math.floor((budget / max(entry, 1e-8)) / step) * step
            if self.p.min_qty and size < float(self.p.min_qty):
                size = 0.0
            if size <= 0.0:
                return
        tp = self._round_price(entry * (1 + self.p.take_profit / 100.0))

        self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))
        self.buy(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.initial_risk = max(1e-8, entry - init_stop)
        self.run_high = float(self.d.high[0]); self.run_low = None
        self.entry_bar = len(self); self.last_add_bar = len(self)

        if self.p.debug:
            console.print(f"ENTER LONG {size} @ {entry} | SL={init_stop} | TP={tp}")

    def _enter_short(self):
        if not self.p.can_short:
            return
        entry = float(self._round_price(self.d.close[0]))
        init_stop = self._round_price(entry + self.p.atr_stop_mult * float(self.d.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            cash = float(self.broker.getcash())
            budget = cash * float(self.p.percent_sizer or 0.0)
            step = float(self.p.qty_step) if self.p.qty_step else 0.001
            size = math.floor((budget / max(entry, 1e-8)) / step) * step
            if self.p.min_qty and size < float(self.p.min_qty):
                size = 0.0
            if size <= 0.0:
                return
        tp = self._round_price(entry * (1 - self.p.take_profit / 100.0))

        self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))
        self.sell(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.initial_risk = max(1e-8, init_stop - entry)
        self.run_low = float(self.d.low[0]); self.run_high = None
        self.entry_bar = len(self); self.last_add_bar = len(self)

        if self.p.debug:
            console.print(f"ENTER SHORT {size} @ {entry} | SL={init_stop} | TP={tp}")

    def _can_pyramid(self) -> bool:
        if not (self.p.use_pyramiding and self.position):
            return False
        if self.n_adds >= self.p.max_adds:
            return False
        if (len(self) - self.last_add_bar) < self.p.add_cooldown:
            return False
        if self._R() < self.p.add_min_R:
            return False
        if self.position.size > 0:
            return float(self.d.close[0]) >= ((self.run_high or float(self.d.high[0])) + self.p.add_atr_mult * float(self.d.atr5[0]))
        else:
            return float(self.d.close[0]) <= ((self.run_low  or float(self.d.low[0]))  - self.p.add_atr_mult * float(self.d.atr5[0]))

    def _do_pyramid(self):
        entry = float(self._round_price(self.d.close[0]))
        stop = float(self.trail_stop or (entry - self.p.atr_stop_mult * float(self.d.atr5[0]) if self.position.size > 0
                                         else entry + self.p.atr_stop_mult * float(self.d.atr5[0])))
        size = self._round_qty(self._risk_based_size(entry, stop) / 2.0)
        if size <= 0:
            return
        if self.position.size > 0:
            tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
            self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))
            self.buy(size=size, exectype=bt.Order.Market)
        else:
            tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
            self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))
            self.sell(size=size, exectype=bt.Order.Market)
        self.n_adds += 1
        self.last_add_bar = len(self)
        if self.p.debug:
            console.print(f"PYRAMID add #{self.n_adds} {size} @ {entry}")

    def _take_profits_and_trail(self):
        if not self.active_orders:
            return
        current = float(self.d.close[0])
        to_remove = []

        # Per-leg TP
        for idx, o in enumerate(self.active_orders):
            if o["dir"] > 0 and current >= o["tp"]:
                self.sell(size=o["size"], exectype=bt.Order.Market)
                to_remove.append(idx)
                if self.p.debug:
                    prof = (current / o["entry"] - 1) * 100
                    console.print(f"TP LONG: -{o['size']} @ {current} (+{prof:.2f}%)")
            elif o["dir"] < 0 and current <= o["tp"]:
                self.buy(size=o["size"], exectype=bt.Order.Market)
                to_remove.append(idx)
                if self.p.debug:
                    prof = (1 - current / o["entry"]) * 100
                    console.print(f"TP SHORT: +{o['size']} @ {current} (+{prof:.2f}%)")

        # Trailing stop
        if self.p.use_trailing_stop:
            self._update_trailing_stop()
            if self._stop_hit():
                qty = sum(o["size"] for o in self.active_orders if (o["dir"] > 0 and self.position.size > 0) or (o["dir"] < 0 and self.position.size < 0))
                if qty > 0:
                    if self.position.size > 0:
                        self.sell(size=qty, exectype=bt.Order.Market)
                    else:
                        self.buy(size=qty, exectype=bt.Order.Market)
                to_remove = list(range(len(self.active_orders)))

        if to_remove:
            for i in reversed(to_remove):
                self.active_orders.pop(i)
                self.sizes = [o["size"] for o in self.active_orders]
                self.entry_prices = [o["entry"] for o in self.active_orders]

            if not self.active_orders:
                self._reset_position_state()

    def _reset_position_state(self):
        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.last_trail_update = -10**9
        self.n_adds = 0
        self.last_add_bar = -10**9

    def next(self):
        # Flat -> Entry
        if not self.position:
            # Basic cooldown after exit
            if (len(self) - self.last_exit_bar) < self.p.reentry_cooldown_bars:
                return

            if self.p.use_htf:
                if self.regime_ok_long() and self.trend_ok_long() and self.breakout_up():
                    self._enter_long()
                elif self.p.can_short and self.regime_ok_short() and self.trend_ok_short() and self.breakdown_down():
                    self._enter_short()
            else:
                # Fallback: only use 1m features and breakout
                if self.d.ema1_fast[0] > self.d.ema1_slow[0] and self.breakout_up():
                    self._enter_long()
                elif self.p.can_short and self.d.ema1_fast[0] < self.d.ema1_slow[0] and self.breakdown_down():
                    self._enter_short()
        else:
            # Manage position
            self._take_profits_and_trail()
            # Optional pyramiding
            if self.p.use_pyramiding and self._can_pyramid():
                self._do_pyramid()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.last_exit_bar = len(self)

# --------------------- Feed helpers ---------------------
def make_feed_from_df(df: pl.DataFrame, spec: DataSpec) -> MSSQLData:
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
        sr = f"{spec.start_date or (spec.ranges[0][0] if spec.ranges else '')}"
        er = f"{spec.end_date or (spec.ranges[-1][1] if spec.ranges else '')}"
        feed._name = f"{spec.symbol}-{spec.interval} {sr}..{er}"
        feed._dataname = f"{spec.symbol}{spec.collateral}"
    except Exception:
        pass
    return feed

def add_mtf_feeds(cerebro: bt.Cerebro, base_feed, add_5m=True, add_15m=True, add_60m=True):
    if add_5m:
        cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=5,  name='5m',  boundoff=1)
    if add_15m:
        cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=15, name='15m', boundoff=1)
    if add_60m:
        cerebro.resampledata(base_feed, timeframe=bt.TimeFrame.Minutes, compression=60, name='60m', boundoff=1)


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


# --------------------- Quick backtest ---------------------
# def quick_backtest(
#     df: pl.DataFrame,
#     strategy_class,
#     df_map: Dict[str, pl.DataFrame], # Pass the whole map
#     spec: DataSpec,                  # Pass the spec to test
#     params: Dict[str, Any],
#     init_cash: float = INIT_CASH,
#     commission: float = COMMISSION_PER_TRANSACTION,
#     exchange: Optional[str] = None,
#     slippage_bps: float = 5.0,
#     min_qty: float = None, qty_step: float = None, price_tick: Optional[float] = None,
#     plot: bool = True,
#     debug: bool = False,
#     use_features: bool = True
# ):
def quick_backtest(strategy_class,df,spec,params,init_cash=INIT_CASH,commission=COMMISSION_PER_TRANSACTION,exchange=None,slippage_bps=5.0,plot=False,debug=False):
    m = meta_for(spec.symbol)
    if min_qty is None: min_qty = m["min_qty"]
    if qty_step is None: qty_step = m["qty_step"]
    if price_tick is None: price_tick = m["price_tick"]

    cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
    df_slice = slice_df_to_spec(df, spec)

    if use_features:
        feat_params = dict(params)  # copy
        feat_df = build_feature_df(df_slice, feat_params)
        feed = make_feature_feed(feat_df, name=f"{spec.symbol}")
        cerebro.adddata(feed)
        strat_class = VectorMACD_ADX  # Use the new vectorized strategy
    # else:
    #     # For non-feature mode, use the original strategy
    #     feed = make_feed_from_df(df_slice, spec)
    #     cerebro.adddata(feed)
    #     add_mtf_feeds(cerebro, feed)
    #     # Import and use the original strategy class
    #     from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX4 as strat_class


    sp = dict(params)
    sp.update(dict(
        can_short=(str(exchange).lower() == "mexc") if exchange else False,
        min_qty=min_qty, qty_step=qty_step, price_tick=price_tick, debug=debug
    ))

    console.print("Strategy params:")
    console.print(sp)

    cerebro.addstrategy(strat_class, backtest=True, **sp)
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    try:
        cerebro.broker.set_slippage_perc(perc=slippage_bps/10000.0)
    except Exception:
        pass

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    if debug:
        cerebro.addobserver(bt.observers.Trades)

    res = cerebro.run(maxcpus=1)
    strat = res[0]

    try:
        maxdd = float(strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0))
    except Exception:
        maxdd = 0.0
    try:
        sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr) if sr is not None else 0.0
    except Exception:
        sharpe = 0.0
    ta = strat.analyzers.trades.get_analysis()
    trades = ta.get('total', {}).get('total', 0) if ta else 0


    cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')

    console.print(f"Trades={trades}, Sharpe={sharpe:.3f}, MaxDD={maxdd:.2f}%, Value={cerebro.broker.getvalue():.2f}")
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
    min_qty: float = None,
    qty_step: float = None,
    price_tick: Optional[float] = None,
    score_fn: Callable = score_sharpe_dd,
    use_features: bool = True
) -> Tuple[float, Dict[str, float], float]:
    m = meta_for(spec.symbol)
    if min_qty is None: min_qty = m["min_qty"]
    if qty_step is None: qty_step = m["qty_step"]
    if price_tick is None: price_tick = m["price_tick"]

    cerebro = bt.Cerebro(oldbuysell=True, runonce=True, stdstats=False, exactbars=1)
    df_slice = slice_df_to_spec(df, spec)

    if use_features:
        feat_df = build_feature_df(df_slice, dict(params))
        feed = make_feature_feed(feat_df, name=f"{spec.symbol}")
        cerebro.adddata(feed)
        strat_class = VectorMACD_ADX
    else:
        feed = make_feed_from_df(df_slice, spec)
        cerebro.adddata(feed)
        add_mtf_feeds(cerebro, feed)
        strat_class = strategy_class

    sp = dict(params)
    sp.update(dict(
        can_short=(str(exchange).lower() == "mexc") if exchange else False,
        min_qty=min_qty, qty_step=qty_step, price_tick=price_tick
    ))
    cerebro.addstrategy(strat_class, backtest=True, **sp)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Minutes, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.broker.setcommission(commission=commission)
    try:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception:
        pass
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
    min_qty: float = None, qty_step: float = None, price_tick: Optional[float] = None,
    scoring: str = "sharpe_dd",
    asset_weights: Optional[Dict[str, float]] = None,
    min_trades_per_spec: int = 10,
    use_median: bool = False,
    use_features: bool = True
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        # Params
        params = {
            "tf5m_breakout_period": trial.suggest_int("tf5m_breakout_period", 35, 75, step=5),
            "adxth": trial.suggest_int("adxth", 18, 28),
            "confirm_bars": trial.suggest_int("confirm_bars", 1, 3),
            "max_stretch_atr_mult": trial.suggest_float("max_stretch_atr_mult", 0.8, 1.3, step=0.1),
            "atr_stop_mult": trial.suggest_float("atr_stop_mult", 2.0, 3.0, step=0.25),
            "trail_mode": trial.suggest_categorical("trail_mode", ["chandelier","ema_band","donchian"]),
            "trail_atr_mult": trial.suggest_float("trail_atr_mult", 3.0, 5.0, step=0.25),
            "ema_band_mult": trial.suggest_float("ema_band_mult", 1.6, 2.6, step=0.2),
            "donchian_trail_period": trial.suggest_int("donchian_trail_period", 45, 75, step=5),
            "move_to_breakeven_R": trial.suggest_float("move_to_breakeven_R", 0.5, 1.5, step=0.25),
            "trail_update_every": trial.suggest_int("trail_update_every", 2, 6),
            "ema_fast": trial.suggest_int("ema_fast", 18, 26),
            "ema_slow": trial.suggest_int("ema_slow", 40, 60),
            "ema_trend": trial.suggest_int("ema_trend", 150, 220),
            "atr_period": trial.suggest_int("atr_period", 12, 18),
            "take_profit": trial.suggest_float("take_profit", 3.0, 8.0, step=0.5),
            "use_volume_filter": trial.suggest_categorical("use_volume_filter", [False, True]),
            "volume_filter_mult": trial.suggest_float("volume_filter_mult", 1.1, 1.8, step=0.1),
            "use_pyramiding": trial.suggest_categorical("use_pyramiding", [False, True]),
            "max_adds": trial.suggest_int("max_adds", 0, 2),
            "add_cooldown": trial.suggest_int("add_cooldown", 15, 40, step=5),
            "add_atr_mult": trial.suggest_float("add_atr_mult", 0.5, 1.5, step=0.25),
            "add_min_R": trial.suggest_float("add_min_R", 0.5, 1.5, step=0.25),
            "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.001, 0.005, step=0.0005),
            "close_based_stop": True,
            "use_htf": True,
            # Regime modes
            "regime_mode_long": trial.suggest_categorical("regime_mode_long", ["ema", "price_vs_slow"]),
            "regime_mode_short": trial.suggest_categorical("regime_mode_short", ["neutral", "ema"]),
            "rsi_overheat": trial.suggest_int("rsi_overheat", 70, 85),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 15, 35),
        }

        per_spec_scores = []
        per_spec_weights = []

        for spec in specs:
            df = df_map[spec.symbol]
            score, metrics, _ = run_single_backtest_eval(
                strategy_class=strategy_class,
                df=df, spec=spec,
                init_cash=init_cash, commission=commission,
                params=params,
                exchange=exchange, slippage_bps=slippage_bps,
                min_qty=min_qty, qty_step=qty_step, price_tick=price_tick,
                score_fn=score_sharpe_dd,
                use_features=use_features
            )
            trades = metrics.get('trades', 0)
            if trades < min_trades_per_spec:
                score = -5.0
            per_spec_scores.append(score)
            w = (asset_weights or {}).get(spec.symbol, 1.0)
            per_spec_weights.append(w)

            trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_score", score)
            trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_trades", trades)
            trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_mdd", metrics.get("mdd", 0.0))
            trial.set_user_attr(f"{spec.symbol}_{spec.start_date}_{spec.end_date}_sharpe", metrics.get("sharpe", 0.0))

        if not per_spec_scores:
            return -999.0

        if use_median:
            import numpy as np
            agg = float(np.median(per_spec_scores))
        else:
            wsum = sum(per_spec_weights) or 1.0
            agg = sum(s*w for s, w in zip(per_spec_scores, per_spec_weights)) / wsum

        return agg

    return objective


# --------------------- Pruner ---------------------
def make_pruner(num_steps: int, pruner: Optional[str], n_jobs: int):
    if pruner == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=max(1, num_steps), reduction_factor=3, bootstrap_count=max(2 * n_jobs, 8))
    elif pruner == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
    elif pruner == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=1)
    else:
        return None


# --------------------- Optimize (single-process) ---------------------
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
    use_features: bool = True,
):
    storage = ensure_storage_or_sqlite(storage_string, study_name)
    console.print("[bold blue]Preloading data (cache-aware, union per symbol)...[/bold blue]")
    df_map = preload_polars(specs)  # union per symbol
    console.print(f"[green]✓ Loaded {len(df_map)} symbols[/green]")

    pruner_obj = make_pruner(num_steps=len(specs), pruner=pruner, n_jobs=n_jobs)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True, warn_independent_sampling=False)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner_obj, storage=storage, study_name=study_name, load_if_exists=True)

    m = meta_for(specs[0].symbol)
    objective = make_objective(
        strategy_class, specs, df_map, init_cash, commission,
        exchange=exchange, slippage_bps=5,
        min_qty=m["min_qty"], qty_step=m["qty_step"], price_tick=m["price_tick"],
        scoring="sharpe_dd", use_features=use_features
    )

    console.print(f"Starting Optuna: trials={n_trials}, n_jobs={n_jobs}, pruner={pruner}, features={use_features}")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=True)
    console.print(f"[bold green]Done. Best value: {study.best_value:.4f}[/bold green]")

    table = Table(title="Best Parameters", show_lines=True)
    table.add_column("Parameter"); table.add_column("Value")
    for k, v in study.best_params.items():
        table.add_row(k, str(v))
    console.print(table)

    if hasattr(study.best_trial, 'user_attrs'):
        console.print("\n[bold cyan]Best Trial Performance (per spec):[/bold cyan]")
        for attr, value in study.best_trial.user_attrs.items():
            if any(tag in attr for tag in ["_score", "_trades", "_sharpe", "_mdd"]):
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
    seed_base: int,
    use_features: bool,
):
    try:
        optimize(
            strategy_class=strategy_class,
            specs=specs,
            n_trials=trials_for_worker,
            n_jobs=1,
            init_cash=init_cash,
            commission=commission,
            pruner="hyperband",
            storage_string=storage_string,
            study_name=study_name,
            seed=seed_base + worker_id,
            exchange=exchange,
            use_features=use_features,
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
    use_features: bool = True,
):
    if workers < 1:
        workers = 1
    trials_per_worker = math.ceil(total_trials / workers)

    console.print(f"[bold magenta]Launching {workers} workers @ {trials_per_worker} trials each (total≈{trials_per_worker*workers})[/bold magenta]")

    # Warm cache
    preload_polars(specs, force_refresh=False)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    procs: List[mp.Process] = []
    for wid in range(workers):
        p = mp.Process(
            target=_worker_optimize,
            args=(wid, strategy_class, specs, storage_string, study_name,
                trials_per_worker, init_cash, commission, exchange, seed_base, use_features)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    storage = ensure_storage_or_sqlite(storage_string, study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    console.print(f"[green]Multiprocess optimize done. Best value: {study.best_value:.4f}[/green]")
    console.print(study.best_params)


# --------------------- Legacy helpers (bulk) ---------------------
def fetch_single_data(coin, start_date, end_date, interval, collateral="USDT"):
    try:
        loader = PolarsDataLoader()
        df = loader.load_union_by_symbol(coin, interval, collateral, [(start_date, end_date)], use_cache=True)
        spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
        feed = make_feed_from_df(slice_df_to_spec(df, spec), spec)
        feed._dataname = f"{coin}{collateral}"
        return feed
    except Exception as e:
        console.print(f"[red]Error fetching data for {coin}: {str(e)}[/red]")
        return None

def fetch_all_data_sequential(coins: List[str], start_date: str, end_date: str, interval: str, collateral: str = "USDT", progress_callback: Optional[Callable] = None) -> Dict[str, Optional[MSSQLData]]:
    loader = PolarsDataLoader()
    total = len(coins)
    out: Dict[str, Optional[MSSQLData]] = {}
    for i, coin in enumerate(coins, 1):
        try:
            df = loader.load_union_by_symbol(coin, interval, collateral, [(start_date, end_date)], use_cache=True)
            spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
            feed = make_feed_from_df(slice_df_to_spec(df, spec), spec)
            feed._dataname = f"{coin}{collateral}"
            out[coin] = feed
            if progress_callback:
                progress_callback(i, total, coin, "success")
        except Exception as e:
            console.print(f"[red]Failed to load {coin}: {e}[/red]")
            out[coin] = None
            if progress_callback:
                progress_callback(i, total, coin, "failed")
    return out

# --------------------- Example usage ---------------------
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
        default_params = dict(VectorMACD_ADX.params._getitems())
        sanity_overrides = {'regime_mode_long':'price_vs_slow','tf60m_ema_fast':20,'tf60m_ema_slow':60,'tf5m_breakout_period':20,'confirm_bars':1,'adxth':18,'atr_stop_mult':2.5,'trail_mode':'chandelier','trail_atr_mult':4.0,'move_to_breakeven_R':0.5,'risk_per_trade_pct':0.002,'rsi_overheat':80,'use_pyramiding':False,'use_volume_filter':False}
        sanity_params = {**default_params, **sanity_overrides}
        
        console.print("[yellow]Sanity backtest (BTC 2024-01-01..15) with vector features...[/yellow]")
        quick_backtest(VectorMACD_ADX, df_map_sanity["BTC"], sanity_spec, sanity_params, init_cash=10000, exchange="MEXC", plot=False, debug=False)

        study_name = "Optimized_1m_MTF_VEC_V3"
        launch_multiprocess_optimize(
            strategy_class=VectorMACD_ADX,
            specs=train_specs,
            storage_string=MSSQL_ODBC,
            study_name=study_name,
            total_trials=200,
            workers=8,
            init_cash=INIT_CASH,
            commission=COMMISSION_PER_TRANSACTION,
            exchange="MEXC",
            seed_base=42,
        )

        storage = ensure_storage_or_sqlite(MSSQL_ODBC, study_name)
        study = optuna.load_study(study_name=study_name, storage=storage)
        best_params = study.best_params

        holdout_spec = DataSpec("BNB", interval="1m", start_date=holdout_start, end_date=holdout_end)
        df_hold = preload_polars([holdout_spec])["BNB"]
        console.print("[magenta]Holdout backtest with best params on BNB...[/magenta]")
        quick_backtest(VectorMACD_ADX, df_hold, holdout_spec, best_params, init_cash=10000, exchange="MEXC", plot=True, debug=False)

    except Exception as e:
        console.print(f"An error occurred: {e}")
        # traceback.print_exc()
    except KeyboardInterrupt:
        console.print("Process interrupted by user.")