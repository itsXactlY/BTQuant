import polars as pl
import connectorx as cx # pip install connectorx
import time
from datetime import datetime, timedelta
from fastquant.dontcommit import MSSQLData, connection_string

def benchmark_mssql_fastquant(connection_string, start_date, end_date):
    start_time = time.time()
    
    if start_date is None:
        min_date = datetime(2000, 1, 1)
        max_date = datetime(2100, 1, 1)
        df = MSSQLData.get_data_from_db(
            connection_string=connection_string,
            coin="BTCUSDT_klines",
            timeframe="1m",
            start_date=min_date,
            end_date=max_date
        )
    else:
        df = MSSQLData.get_data_from_db(
            connection_string=connection_string,
            coin="BTCUSDT_klines",
            timeframe="1m",
            start_date=start_date,
            end_date=end_date
        )
    
    end_time = time.time()
    return df, end_time - start_time

def benchmark_polars_cx(connection_string, start_date, end_date):
    cx_connection_string = f"mssql://sa:<password>@localhost/BacktraderData"
    
    if start_date is None:
        query = """
        SELECT *
        FROM BTCUSDT_klines
        WHERE Timeframe = '1m'
        ORDER BY TimestampStart
        """
    else:
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        query = f"""
        SELECT *
        FROM BTCUSDT_klines
        WHERE Timeframe = '1m'
        AND TimestampStart BETWEEN {start_timestamp} AND {end_timestamp}
        ORDER BY TimestampStart
        """
    
    start_time = time.time()
    df = pl.from_arrow(
        cx.read_sql(
            conn=cx_connection_string,
            query=query,
            return_type="arrow"
        )
    )
    
    df = df.with_columns([
        pl.col("TimestampStart").cast(pl.Datetime(time_unit="ms")),
        pl.col("TimestampEnd").cast(pl.Datetime(time_unit="ms")),
        pl.col("Open").cast(pl.Float64),
        pl.col("High").cast(pl.Float64),
        pl.col("Low").cast(pl.Float64),
        pl.col("Close").cast(pl.Float64),
        pl.col("Volume").cast(pl.Float64),
        pl.col("QuoteVolume").cast(pl.Float64),
        pl.col("Trades").cast(pl.Int64),
        pl.col("TakerBaseVolume").cast(pl.Float64),
        pl.col("TakerQuoteVolume").cast(pl.Float64)
    ])
    
    end_time = time.time()
    return df, end_time - start_time

def format_benchmark_results(period_name, df_fast, time_fast, df_polars, time_polars):
    """Create a formatted benchmark results string"""
    width = 80
    lines = []
    lines.append(f"\n{'=' * width}")
    lines.append(f"Period: {period_name}".center(width))
    lines.append('-' * width)
    
    # FastQuant results
    lines.append("FastQuant MSSQLData:")
    lines.append(f"  Time: {time_fast:.4f} seconds")
    lines.append(f"  Records: {len(df_fast):,}")
    lines.append(f"  Columns: {len(df_fast.columns)} {list(df_fast.columns)}")
    
    # Polars results
    lines.append("\nPolars + ConnectorX:")
    lines.append(f"  Time: {time_polars:.4f} seconds")
    lines.append(f"  Records: {len(df_polars):,}")
    lines.append(f"  Columns: {len(df_polars.columns)} {list(df_polars.columns)}")
    
    # Performance comparison
    speedup = time_fast / time_polars
    lines.append("\nPerformance:")
    lines.append(f"  Speedup ratio: {speedup:.2f}x {'(Polars faster)' if speedup > 1 else '(FastQuant faster)'}")
    
    # Data consistency
    lines.append("\nConsistency Check:")
    lines.append(f"  Row counts: {'✓' if len(df_fast) == len(df_polars) else '✗'}")
    lines.append(f"  Columns match: {'✓' if len(df_fast.columns) == len(df_polars.columns) else '✗'}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    test_periods = [
        (timedelta(hours=1), "1 Hour"),
        (timedelta(days=1), "1 Day"),
        (timedelta(days=7), "1 Week"),
        (timedelta(days=30), "1 Month"),
        (timedelta(days=365), "1 Year"),
        ("FULL", "FULL TABLE")
    ]
    
    print("\nBENCHMARK: C++ FastQuant vs Polars + ConnectorX")
    print(f"{'=' * 80}")
    
    for period, period_name in test_periods:
        if period == "FULL":
            start_date = None
            end_date = None
        else:
            end_date = datetime(2024, 12, 10)
            start_date = end_date - period
        
        df_fast, time_fast = benchmark_mssql_fastquant(connection_string, start_date, end_date)
        df_polars, time_polars = benchmark_polars_cx(connection_string, start_date, end_date)
        
        results = format_benchmark_results(period_name, df_fast, time_fast, df_polars, time_polars)
        print(results)