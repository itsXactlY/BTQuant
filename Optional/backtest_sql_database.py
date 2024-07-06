from imports import *
from live_strategys.QQE_Hullband_VolumeOsc import QQE_Example
from live_strategys.DCA_QQE_Example_backtesting import QQE_DCA_Example
from live_strategys.SuperTrend_Scalp import SuperSTrend_Scalper

#########
# Quantstats
import quantstats
import datetime as dt

# C++ Microsoft SQL
connection_string = (f'DRIVER={driver};'
                     f'SERVER={server};'
                     f'DATABASE={database};'
                     f'UID={username};'
                     f'PWD={password};'
                     f'TrustServerCertificate=yes;')

import importlib
import sys
import os
import pandas as pd

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './Optional/MsSQL/build/lib.linux-x86_64-cpython-312/fast_mssql.cpython-312-x86_64-linux-gnu.so'))
spec = importlib.util.spec_from_file_location("fast_mssql", module_path)
fast_mssql = importlib.util.module_from_spec(spec)
sys.modules["fast_mssql"] = fast_mssql
spec.loader.exec_module(fast_mssql)


class MSSQLData(bt.feeds.PandasData):
    @classmethod
    def get_data_from_db(cls, connection_string, coin, timeframe, start_date, end_date):
        # Convert datetime to Unix timestamp (milliseconds)
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
        FROM {coin}USDT_klines
        WHERE Timeframe = '{timeframe}'
        AND TimestampStart BETWEEN {start_timestamp} AND {end_timestamp}
        ORDER BY TimestampStart
        """
        
        # Use the C++ function to fetch data
        data = fast_mssql.fetch_data_from_db(connection_string, query)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['TimestampStart', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Convert TimestampStart from bigint (milliseconds) to datetime
        df['TimestampStart'] = pd.to_datetime(df['TimestampStart'].astype(int), unit='ms')
        
        # Convert numeric columns to float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Set TimestampStart as index
        df.set_index('TimestampStart', inplace=True)
        
        return df

def backtest():
    startdate = "2024-01-01"
    enddate = "9999-01-01"
    timeframe = "1m"
    coin_name = "BTC"

    # Convert string dates to datetime objects
    start_date = dt.datetime.strptime(startdate, "%Y-%m-%d")
    end_date = dt.datetime.strptime(enddate, "%Y-%m-%d")
    
    df = MSSQLData.get_data_from_db(connection_string, coin_name, timeframe, start_date, end_date)
    
    if df.empty:
        print("No data returned from the database. Please check your query and date range.")
        return
    data = MSSQLData(dataname=df)
    
    cerebro = bt.Cerebro(oldbuysell=False, stdstats=True)
    cerebro.adddata(data)
    cerebro.addstrategy(SuperSTrend_Scalper, backtest=True)
    
    startcash = 1000
    cerebro.broker.setcash(startcash)

    # Add analyzers
    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    
    start = cerebro.broker.getvalue()
    strategy = cerebro.run()[0]

    # Get analyzer results
    max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()
    sqn = strategy.analyzers.sqn.get_analysis()['sqn']

    # Get returns from PyFolio analyzer
    pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    # Convert returns to a Pandas Series and drop NaN values
    returns = pd.Series(returns)
    returns = returns.dropna()

    # Format the filename with coin name, start date, end date, and current date
    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    filename = f"QuantStat_{startdate}_to_{enddate}_generated_on_{current_date}_{dt.datetime.now()}.html"

    # Generate QuantStats report
    quantstats.reports.html(returns, output=filename, title=f'{current_date}_{startdate}_to_{enddate}_1m')
    
    def print_trade_analyzer_results(trade_analyzer, indent=0):
        for key, value in trade_analyzer.items():
            if isinstance(value, dict):
                print_trade_analyzer_results(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    # Print all analyzer results
    print_trade_analyzer_results(trade_analyzer)
    pprint(f"Max Drawdown: {max_drawdown}")
    pprint(f"SQN: {sqn}")
    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - startcash
    pprint('Final Portfolio Value: ${}'.format(portvalue))
    pprint('P/L: ${}'.format(pnl))

    cerebro.plot(style='candles', numfigs=1, volume=False, block=True)
    
    return start - cerebro.broker.getvalue()

def list_tables(connection_string):
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
    tables = [row.TABLE_NAME for row in cursor.fetchall()]
    print("Tables in the database:", tables)
    conn.close()

if __name__ == '__main__':
    if debug == True:
        list_tables(connection_string)
    backtest()