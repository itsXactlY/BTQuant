import backtrader as bt
import pandas as pd
from live_strategys.SuperTrend_Scalp import SuperSTrend_Scalper
import quantstats
import datetime as dt
import os
import pyodbc

from dontcommit import driver, server, database, username, password
connection_string = (f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;')

class MSSQLData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
    )

    @classmethod
    def get_data_from_db(cls, connection_string, table_name, timeframe, start_date, end_date):
        conn = pyodbc.connect(connection_string)
        try:
            query = f"""
            SELECT 
                TimestampStart,
                [Open], [High], [Low], [Close], Volume
            FROM {table_name}
            WHERE Timeframe = '{timeframe}'
            ORDER BY TimestampStart
            """
            df = pd.read_sql(query, conn)
            df['TimestampStart'] = pd.to_datetime(df['TimestampStart'], unit='ms')
            df = df[(df['TimestampStart'] >= start_date) & (df['TimestampStart'] <= end_date)]
            df.set_index('TimestampStart', inplace=True)
            return df
        finally:
            conn.close()

def backtest(table_name, start_date, end_date, timeframe, output_dir):
    try:
        print(f"Starting backtest for {table_name}")
        df = MSSQLData.get_data_from_db(connection_string, table_name, timeframe, start_date, end_date)
        
        if df.empty:
            print(f"No data returned from the database for table {table_name}. Please check your query and date range.")
            return
        
        data = MSSQLData(dataname=df)
        
        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        cerebro.addstrategy(SuperSTrend_Scalper, backtest=True)
        
        startcash = 10000
        cerebro.broker.setcash(startcash)
        
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        
        strategy = cerebro.run()[0]

        # Analyze results
        max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
        trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()
        sqn = strategy.analyzers.sqn.get_analysis()['sqn']

        pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
        returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
        
        returns = pd.Series(returns).dropna()

        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - startcash

        current_date = dt.datetime.now().strftime("%Y-%m-%d")
        backtest_folder = os.path.join(output_dir, f"{table_name}_{start_date}_to_{end_date}")
        os.makedirs(backtest_folder, exist_ok=True)
    

        html_filename = f"QuantStats_Report_{current_date}.html"
        html_filepath = os.path.join(backtest_folder, html_filename)
        quantstats.reports.html(returns, output=html_filepath, title=f'{table_name} {start_date} to {end_date}')

        results_filename = f"Backtest_Results_{current_date}.txt"
        results_filepath = os.path.join(backtest_folder, results_filename)
        with open(results_filepath, 'w') as f:
            f.write(f"Results for {table_name}:\n")
            f.write(f"Max Drawdown: {max_drawdown}\n")
            f.write(f"SQN: {sqn}\n")
            f.write(f"Final Portfolio Value: ${portvalue}\n")
            f.write(f"P/L: ${pnl}\n")
            f.write(f"Trade Analysis:\n")
            f.write(str(trade_analyzer))

        print(f"Results saved for {table_name} in {backtest_folder}")
        import matplotlib.pyplot as plt
        plt.close('all')
        return True

    except Exception as e:
        print(f"An error occurred during backtest for {table_name}: {e}")
