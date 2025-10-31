
import polars as pl
import numpy as np
from typing import Dict
import ccxt
from rich.console import Console
from datetime import datetime, timedelta

console = Console()

def load_and_cache_data(config: Dict) -> pl.DataFrame:

    symbol = config['symbol']
    interval = config['interval']
    start_str = config['start_date']
    end_str = config['end_date']
    cache_path = config['data_dir'] / f"{symbol}_{interval}.parquet"

    start_dt = datetime.strptime(start_str, '%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = datetime.strptime(end_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59, microsecond=999999)

    start_lit = pl.lit(start_dt).cast(pl.Datetime)
    end_lit = pl.lit(end_dt).cast(pl.Datetime)

    if cache_path.exists():
        try:
            df = pl.read_parquet(cache_path)
            if len(df) > 0 and 'timestamp' in df.columns and df['timestamp'].dtype == pl.Datetime:
                df_filtered = df.filter((pl.col('timestamp') >= start_lit) & (pl.col('timestamp') <= end_lit))
                if len(df_filtered) > 10:
                    console.print(f"✅ Loaded cached data: {len(df_filtered):,} bars (dtype: {df['timestamp'].dtype})")
                    return df_filtered
        except Exception as e:
            console.print(f"⚠️ Cache invalid ({e}); reloading...")

    console.print(f"📥 Fetching {symbol} {interval} data from {start_str} to {end_str}...")
    exchange = ccxt.binance()

    ohlcv = []
    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    while since_ms < end_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, interval, since=since_ms, limit=1000)
            if not batch:
                break
            ohlcv.extend(batch)

            if interval == '4h':
                since_ms = batch[-1][0] + (4 * 60 * 60 * 1000)
            elif interval == '1h':
                since_ms = batch[-1][0] + (60 * 60 * 1000)
            else:
                since_ms = batch[-1][0] + 1
        except Exception as e:
            console.print(f"⚠️ Fetch error: {e}; using partial data.")
            break

    if len(ohlcv) < 10:
        console.print("❌ Insufficient real data; using synthetic fallback for testing.")

        dates = []
        current = start_dt
        interval_delta = timedelta(hours=4) if interval == '4h' else timedelta(minutes=15)
        while current <= end_dt:
            dates.append(current)
            current += interval_delta
        n_bars = len(dates)
        returns = np.random.normal(0, 0.015, n_bars)
        closes = 10000 * np.exp(np.cumsum(returns))
        highs = closes * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
        opens = np.roll(closes * (1 + np.random.normal(0, 0.002, n_bars)), 1)
        opens[0] = closes[0]
        volumes = np.random.uniform(500, 5000, n_bars)

        ohlcv = [
            [int(date.timestamp() * 1000), opens[i], highs[i], lows[i], closes[i], volumes[i]]
            for i, date in enumerate(dates)
        ]

    timestamps_ms = [row[0] for row in ohlcv]
    opens = [row[1] for row in ohlcv]
    highs = [row[2] for row in ohlcv]
    lows = [row[3] for row in ohlcv]
    closes = [row[4] for row in ohlcv]
    volumes = [row[5] for row in ohlcv]

    df = pl.DataFrame({
        'timestamp_sec': [ts / 1000.0 for ts in timestamps_ms],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }).unique(subset=['timestamp_sec'])

    df = df.with_columns(
        pl.from_epoch('timestamp_sec', time_unit='s').alias('timestamp')
    ).drop('timestamp_sec')

    df = df.with_columns([
        pl.max_horizontal('open', 'close').clip(pl.col('high')).alias('high'),
        pl.min_horizontal('open', 'close').clip(pl.col('low')).alias('low')
    ]).with_columns([

        pl.when((pl.col('high') < pl.col('open')) | (pl.col('high') < pl.col('close')) |
                (pl.col('low') > pl.col('open')) | (pl.col('low') > pl.col('close')) |
                (pl.col('high') < pl.col('low'))).then(None).otherwise(pl.col('timestamp')).alias('valid_ts')
    ]).filter(pl.col('valid_ts').is_not_null()).drop('valid_ts')

    df = df.select(['timestamp', 'open', 'high', 'low', 'close', 'volume']).sort('timestamp')

    df = df.filter((pl.col('timestamp') >= start_lit) & (pl.col('timestamp') <= end_lit))

    if len(df) == 0:
        raise ValueError(f"No data after filtering. Check dates: {start_str} to {end_str}. Fetched {len(ohlcv)} raw bars.")

    df = df.with_columns(pl.col('timestamp').cast(pl.Datetime('ns')))

    df.write_parquet(cache_path)
    console.print(f"💾 Cached {len(df):,} valid bars to {cache_path} (timestamp dtype: {df['timestamp'].dtype})")
    return df
