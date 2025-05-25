import time
import datetime as dt
import polars as pl
from backtrader.dontcommit import connection_string, fast_mssql, bt

class PolarsData(bt.feed.DataBase):
    '''
    Uses a Polars DataFrame as the feed source
    '''

    params = (
        ('nocase', True),
        ('datetime', 0),  # Default: first column is datetime
        ('open', 1),      
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),  # -1 means not present
    )

    datafields = [
        'datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest'
    ]

    def __init__(self):
        super(PolarsData, self).__init__()
        self.colnames = self.p.dataname.columns
        self._colmapping = {}

        for datafield in self.getlinealiases():
            param_value = getattr(self.params, datafield)
            
            if isinstance(param_value, int):
                if param_value >= 0:
                    if param_value < len(self.colnames):
                        self._colmapping[datafield] = param_value
                    else:
                        self._colmapping[datafield] = None
                elif param_value == -1:
                    found = False
                    for i, colname in enumerate(self.colnames):
                        if self.p.nocase:
                            found = datafield.lower() == colname.lower()
                        else:
                            found = datafield == colname
                            
                        if found:
                            self._colmapping[datafield] = i
                            break
                    
                    if not found:
                        self._colmapping[datafield] = None
                else:
                    self._colmapping[datafield] = None
            
            elif isinstance(param_value, str):
                try:
                    col_idx = self.colnames.index(param_value)
                    self._colmapping[datafield] = col_idx
                except ValueError:
                    if self.p.nocase:
                        found = False
                        for i, colname in enumerate(self.colnames):
                            if param_value.lower() == colname.lower():
                                self._colmapping[datafield] = i
                                found = True
                                break
                        if not found:
                            self._colmapping[datafield] = None
                    else:
                        self._colmapping[datafield] = None
            else:
                self._colmapping[datafield] = None

    def start(self):
        super(PolarsData, self).start()
        self._idx = -1

    def _load(self):
        self._idx += 1
        
        if self._idx >= len(self.p.dataname):
            return False

        for datafield in self.getlinealiases():
            if datafield == 'datetime':
                continue
                
            col_idx = self._colmapping[datafield]
            if col_idx is None:
                continue
                
            line = getattr(self.lines, datafield)
            try:
                line[0] = self.p.dataname.row(self._idx)[col_idx]
            except Exception as e:
                print(f"Error getting value for {datafield} at index {self._idx}, col_idx {col_idx}: {e}")
                line[0] = float('nan')

        dt_idx = self._colmapping['datetime']
        if dt_idx is not None:
            try:
                dt_value = self.p.dataname.row(self._idx)[dt_idx]

                if isinstance(dt_value, (str, int, float)):
                    if isinstance(dt_value, str):
                        # If string, parse as ISO format
                        from datetime import datetime
                        dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                    else:
                        # If number, assume timestamp
                        from datetime import datetime
                        dt = datetime.fromtimestamp(float(dt_value)/1000 if dt_value > 1e10 else float(dt_value))
                else:
                    dt = dt_value
                    
                from backtrader import date2num
                dtnum = date2num(dt)
                self.lines.datetime[0] = dtnum
            except Exception as e:
                print(f"Error processing datetime at index {self._idx}, col_idx {dt_idx}: {e}")
                self.lines.datetime[0] = float('nan')
            
        return True

class MSSQLData(bt.feeds.PolarsData):
    @classmethod
    def get_data_from_db(cls, connection_string, coin, timeframe, start_date, end_date):
        start_timestamp = int(start_date.timestamp() * 1_000_000)
        end_timestamp = int(end_date.timestamp() * 1_000_000)
        print(start_timestamp, end_timestamp)

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

        print("First timestamp in data:", df["TimestampStart"].min())
        print("Last timestamp in data:", df["TimestampStart"].max())
        
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
    start_time = time.time()

    db_resolution = "1s" if "_1s" in ticker else "1m"
    df = MSSQLData.get_data_from_db(connection_string, coin_name, db_resolution, start, end)

    elapsed_time = time.time() - start_time

    if df.is_empty():
        print("No data returned from the database - Please check your query and date range")
        return None

    print(f"Data extraction completed in {elapsed_time:.4f} seconds")
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
        print(f'Resampling microseconds from DB into {time_resolution} Candle data')
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

    print('Data extraction & manipulation took:', time.time() - start_time, 'seconds for', pair)
    
    return df