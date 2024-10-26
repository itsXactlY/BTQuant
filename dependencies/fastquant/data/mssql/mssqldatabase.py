import time
import datetime as dt
from fastquant.dontcommit import MSSQLData, pd, connection_string
import pandas as pd

def get_database_data(ticker, start_date, end_date, time_resolution="1d"):
    def identify_gaps(df, expected_interval):
        df['timestamp'] = pd.to_datetime(df.index)
        df['time_diff'] = df['timestamp'].diff()
        gaps = df[df['time_diff'] > expected_interval]
        return gaps

    resample = True
    coin_name = ticker + "USDT_klines"
    start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end = dt.datetime.strptime(end_date, "%Y-%m-%d")
    start_time = time.time()

    df = MSSQLData.get_data_from_db(connection_string, coin_name, "1m", start, end)

    elapsed_time = time.time() - start_time

    if df.empty:
        print("No data returned from the database - Please check your query and date range")
        return

    print(f"Data extraction completed in {elapsed_time:.20f} seconds")
    print(f"Number of rows retrieved: {len(df)}")

    def convert_time_resolution(time_resolution):
        if time_resolution.endswith('s'):  # seconds
            return time_resolution.replace('s', 'S')
        if time_resolution.endswith('m'):  # minutes
            return time_resolution.replace('m', 'T')
        elif time_resolution.endswith('h'):  # hours
            return time_resolution.replace('h', 'H')
        elif time_resolution.endswith('d'):  # days
            return time_resolution.replace('d', 'D')
        elif time_resolution.endswith('w'):  # weeks
            return time_resolution.replace('w', 'W')
        elif time_resolution.endswith('M'):  # months
            return time_resolution.replace('M', 'M')
        elif time_resolution.endswith('Y'):  # years
            return time_resolution.replace('Y', 'A')
        else:
            raise ValueError(f"Unsupported time resolution: {time_resolution}")

    time_resolution = convert_time_resolution(time_resolution)

    if resample:
        print(f'Resampling data into {time_resolution} Candle data')
        
        # Create a complete date range
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=time_resolution)
        
        # Resample the data into the specified intervals
        df_resampled = df.resample(time_resolution).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Reindex with the full date range to include missing periods
        df_resampled = df_resampled.reindex(full_date_range)
        
        # Forward fill the missing values
        df_resampled['Close'] = df_resampled['Close'].ffill()
        df_resampled['Open'] = df_resampled['Open'].fillna(df_resampled['Close'])
        df_resampled['High'] = df_resampled['High'].fillna(df_resampled['Close'])
        df_resampled['Low'] = df_resampled['Low'].fillna(df_resampled['Close'])
        df_resampled['Volume'] = df_resampled['Volume'].fillna(0)
        
        # Identify gaps
        gaps = identify_gaps(df_resampled, pd.Timedelta(time_resolution))
        if not gaps.empty:
            print(f"Gaps found in the data:")
            print(gaps)
        
        df = df_resampled

    print('Data extraction & manipulation took: ', time.time() - start_time, 'seconds')
    return df