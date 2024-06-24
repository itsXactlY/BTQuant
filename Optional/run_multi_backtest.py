import os
import pyodbc
import multiprocessing
from backtesting_multi import backtest

# Configuration
base_output_dir = f"/usr/share/nginx/html/QuantStats/"
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Database connection
from dontcommit import driver, server, database, username, password
connection_string = (f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;')

def list_tables(connection_string):
    with pyodbc.connect(connection_string) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
        tables = [row.TABLE_NAME for row in cursor.fetchall()]
        return tables

from time import sleep
def run_backtest_for_table(args):
    table, start_date, end_date, timeframe, base_output_dir = args
    backtest(table, start_date, end_date, timeframe, base_output_dir)

def run_backtests(connection_string):
    tables = list_tables(connection_string)
    start_date = "2023-05-01"
    end_date = "2024-05-31"
    timeframe = "1m"
    
    with multiprocessing.Pool(processes=8) as pool:
        args = [(table, start_date, end_date, timeframe, base_output_dir) for table in tables]
        pool.map(run_backtest_for_table, args)
        sleep(.1)
        pool.close()
        sleep(.1)
        pool.join()

if __name__ == '__main__':
    run_backtests(connection_string)
