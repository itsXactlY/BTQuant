import time
import datetime as dt
import polars as pl
from backtrader.dontcommit import connection_string, fast_mssql, bt

import pyodbc
import pandas as pd
from datetime import datetime

class MSSQLData(bt.feeds.PandasData):
    @classmethod
    def get_data_from_db(cls, connection_string, coin, timeframe, start_date, end_date):
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        query = f"""
        SELECT 
            TimestampStart, 
            [Open], 
            [High], 
            [Low], 
            [Close], 
            Volume
        FROM {coin}
        WHERE Timeframe = '{timeframe}'
        AND TimestampStart BETWEEN {start_timestamp} AND {end_timestamp}
        ORDER BY TimestampStart
        OPTION(USE HINT('ENABLE_PARALLEL_PLAN_PREFERENCE'))
        """
        
        data = fast_mssql.fetch_data_from_db(connection_string, query)
        
        df = pd.DataFrame(data, columns=['TimestampStart', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['TimestampStart'] = pd.to_datetime(df['TimestampStart'].astype(int), unit='ms')
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df.set_index('TimestampStart', inplace=True)
        return df

    @classmethod
    def get_all_pairs(cls, connection_string):
        query = "SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        data = fast_mssql.fetch_data_from_db(connection_string, query)
        return [row[0] for row in data]

class MSSQLData_Stocks(bt.feeds.PandasData):
    @classmethod
    def get_data_from_db(cls, connection_string, coin, timeframe, start_date, end_date):
        # Convert datetime objects to string format
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
        SELECT 
            TimestampStart, 
            [Open], 
            [High], 
            [Low], 
            [Close], 
            Volume
        FROM {coin}
        WHERE Timeframe = '{timeframe}'
        AND TimestampStart BETWEEN '{start_str}' AND '{end_str}'
        ORDER BY TimestampStart
        OPTION(USE HINT('ENABLE_PARALLEL_PLAN_PREFERENCE'))
        """
        
        data = fast_mssql.fetch_data_from_db(connection_string, query)
        
        df = pd.DataFrame(data, columns=['TimestampStart', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['TimestampStart'] = pd.to_datetime(df['TimestampStart'])
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        df.set_index('TimestampStart', inplace=True)
        return df

    @classmethod
    def get_all_pairs(cls, connection_string):
        query = "SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        data = fast_mssql.fetch_data_from_db(connection_string, query)
        return [row[0] for row in data]


    @classmethod
    def get_all_pairs(cls, connection_string):
        query = "SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        try:
            with pyodbc.connect(connection_string) as conn:
                data = pd.read_sql(query, conn)
            return data['TABLE_NAME'].tolist()
        except pyodbc.Error as e:
            print(f"Database error occurred: {str(e)}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return []

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