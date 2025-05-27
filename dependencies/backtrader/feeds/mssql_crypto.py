import time
import datetime as dt
import polars as pl
from backtrader.dontcommit import connection_string, fast_mssql, bt

class MSSQLData(bt.feeds.PolarsData):
    @classmethod
    def get_data_from_db(cls, connection_string, coin, timeframe, start_date, end_date):
        start_timestamp = int(start_date.timestamp() * 1_000_000)
        end_timestamp = int(end_date.timestamp() * 1_000_000)

        query = f"""
        SELECT
            TimestampStart, 
            [Open], 
            [High], 
            [Low], 
            [Close], 
            Volume
        FROM [{coin}]
        WHERE Timeframe = '{timeframe}'
        AND TimestampStart BETWEEN {start_timestamp} AND {end_timestamp}
        ORDER BY TimestampStart
        OPTION(USE HINT('ENABLE_PARALLEL_PLAN_PREFERENCE'))
        """
        
        data = fast_mssql.fetch_data_from_db(connection_string, query)
        
        df = pl.DataFrame(
            data,
            schema=["TimestampStart", "Open", "High", "Low", "Close", "Volume"],
            orient="row"
        )

        df = df.with_columns([
            pl.col("TimestampStart").cast(pl.Int64).map_elements(
                lambda x: dt.datetime.fromtimestamp(x/1_000_000),
                return_dtype=pl.Datetime
            ).alias("TimestampStart")
        ])

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df.with_columns([
            pl.col(col).cast(pl.Float64) for col in numeric_cols
        ])

        # print("First timestamp in data:", df["TimestampStart"].min())
        # print("Last timestamp in data:", df["TimestampStart"].max())
        
        return df

    @classmethod
    def get_all_pairs(cls, connection_string):
        query = "SELECT DISTINCT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        data = fast_mssql.fetch_data_from_db(connection_string, query)
        return [row[0] for row in data]


def get_database_data(ticker, start_date, end_date, time_resolution="1d", pair="USDT"):
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
    # start_time = time.time()

    db_resolution = "1s" if "_1s" in ticker else "1m"
    df = MSSQLData.get_data_from_db(connection_string, coin_name, db_resolution, start, end)

    # elapsed_time = time.time() - start_time

    if df.is_empty():
        print("No data returned from the database - Please check your query and date range")
        return None

    # print(f"Data extraction completed in {elapsed_time:.4f} seconds")
    # print(f"Number of rows retrieved: {len(df)}")

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
        # print(f'Resampling microseconds from DB into {time_resolution} Candle data')
        df = df.sort("TimestampStart")
        
        df = (df
            .group_by_dynamic("TimestampStart", every=time_resolution)
            .agg([
                pl.col("Open").first().alias("Open"),
                pl.col("High").max().alias("High"),
                pl.col("Low").min().alias("Low"),
                pl.col("Close").last().alias("Close"),
                pl.col("Volume").sum().alias("Volume"),
            ])
        )

    # print('Data extraction & manipulation took:', time.time() - start_time, 'seconds for', pair)
    
    return df