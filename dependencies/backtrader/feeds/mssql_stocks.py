import time
import datetime as dt
# from backtrader.dontcommit import MSSQLData_Stocks, connection_string

def get_database_stock_data(
    ticker, start_date, end_date, time_resolution="1m"
):
    def identify_gaps(df, expected_interval):
        df['timestamp'] = pd.to_datetime(df.index)
        df['time_diff'] = df['timestamp'].diff()
        gaps = df[df['time_diff'] > expected_interval]
        return gaps
    
    resample = False
    coin_name = ticker
    start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end = dt.datetime.strptime(end_date, "%Y-%m-%d")
    start_time = time.time()

    df = MSSQLData_Stocks.get_data_from_db(connection_string, coin_name, time_resolution, start, end)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    if df.empty:
        print("No data returned from the database - Please check your query and date range")
        return

    print(f"Data extraction completed in {elapsed_time:.20f} seconds")
    print(f"Number of rows retrieved: {len(df)}")

    # Dynamically convert time resolution to Pandas offset alias
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
        # Resample the data into the specified intervals
        df_resampled = df.resample(time_resolution).agg({
            'Open': 'first',   # Take the first open in the window
            'High': 'max',     # Take the highest value in the window
            'Low': 'min',      # Take the lowest value in the window
            'Close': 'last',   # Take the last close in the window
            'Volume': 'sum'    # Sum the volume
        })
        df = df_resampled

        # Identify gaps
        gaps = identify_gaps(df_resampled, pd.Timedelta(time_resolution))
        if not gaps.empty:
            print(f"Gaps found in the data:")
            print(gaps)
        
        df = df_resampled
    print('Data extraction & manipulation took: ', time.time() - start_time, 'seconds')
    return df
