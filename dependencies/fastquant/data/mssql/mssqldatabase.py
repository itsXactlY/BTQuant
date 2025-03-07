import time
import datetime as dt
import polars as pl
from fastquant.dontcommit import MSSQLData, connection_string

def get_database_data(ticker, start_date, end_date, time_resolution="1d", pair="USDT"):
    def identify_gaps(df, expected_interval):
        if expected_interval.endswith('h'):
            duration = pl.duration(hours=int(expected_interval[:-1]))
        elif expected_interval.endswith('m'):
            duration = pl.duration(minutes=int(expected_interval[:-1]))
        elif expected_interval.endswith('s'):
            duration = pl.duration(seconds=int(expected_interval[:-1]))
        else:
            duration = pl.duration(days=1)
            
        return (df
            .with_columns([
                pl.col("TimestampStart").diff().alias("time_diff")
            ])
            .filter(pl.col("time_diff") > duration)
        )
    
    resample = True

    if "_1s" in ticker:
        base_ticker = ticker.replace("_1s", "")
        coin_name = base_ticker + pair + "_1s_klines"
    else:
        if pair == "USDT":
            coin_name = ticker + "USDT_klines"
        else:
            coin_name = ticker + pair + "_klines"
        
    start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end = dt.datetime.strptime(end_date, "%Y-%m-%d")
    start_time = time.time()

    db_resolution = "1s" if "_1s" in ticker else "1m"
    pdf = MSSQLData.get_data_from_db(connection_string, coin_name, db_resolution, start, end)
    df = pl.from_pandas(pdf.reset_index())

    elapsed_time = time.time() - start_time

    if df.is_empty():
        print("No data returned from the database - Please check your query and date range")
        return

    print(f"Data extraction completed in {elapsed_time:.20f} seconds")
    print(f"Number of rows retrieved: {len(df)}")

    def convert_time_resolution(time_resolution):
        if time_resolution.endswith('s'):  # seconds
            return time_resolution.lower()
        if time_resolution.endswith('m'):  # minutes
            return time_resolution.lower()
        elif time_resolution.endswith('h'):  # hours
            return time_resolution.lower()
        elif time_resolution.endswith('d'):  # days
            return time_resolution.lower()
        elif time_resolution.endswith('w'):  # weeks
            return time_resolution.lower()
        elif time_resolution.endswith('M'):  # months
            return 'mo'
        elif time_resolution.endswith('Y'):  # years
            return 'y'
        else:
            raise ValueError(f"Unsupported time resolution: {time_resolution}")

    time_resolution = convert_time_resolution(time_resolution)

    if resample:
        print(f'Resampling data into {time_resolution} Candle data')
        
        df_resampled = (df
            .group_by_dynamic("TimestampStart", every=time_resolution)
            .agg([
                pl.col("Open").first(),
                pl.col("High").max(),
                pl.col("Low").min(),
                pl.col("Close").last(),
                pl.col("Volume").sum()
            ])
        )
        
        df_resampled = df_resampled.with_columns([
            pl.col("Close").forward_fill(),
            pl.col("Open").forward_fill().fill_null(pl.col("Close")),
            pl.col("High").forward_fill().fill_null(pl.col("Close")),
            pl.col("Low").forward_fill().fill_null(pl.col("Close")),
            pl.col("Volume").fill_null(0)
        ])

        gaps = identify_gaps(df_resampled, time_resolution)
        if not gaps.is_empty():
            print("Gaps found in the data:")
            print(gaps)
        
        df = df_resampled

    print('Data extraction & manipulation took:', time.time() - start_time, 'seconds')
    
    # Convert back to pandas for compatibility with BTQuant
    return df.to_pandas().set_index("TimestampStart")