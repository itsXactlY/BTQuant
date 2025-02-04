import backtrader as bt
import joblib

identify = ""
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = ""
jrr_order_history_sub1 = ""

bsc_privaccount1 = ""
bsc_privaccountaddress = ""


import importlib
import sys
import os
import pandas as pd
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/mssql/MsSQL/fast_mssql.cpython-313-x86_64-linux-gnu.so'))
spec = importlib.util.spec_from_file_location("fast_mssql", module_path)
fast_mssql = importlib.util.module_from_spec(spec)
sys.modules["fast_mssql"] = fast_mssql
spec.loader.exec_module(fast_mssql)

# SQL Server connection details
server = ''
database = ''
username = ''
password = ''
driver = '{ODBC Driver 18 for SQL Server}'

connection_string = (f'DRIVER={driver};'
                     f'SERVER={server};'
                     f'DATABASE={database};'
                     f'UID={username};'
                     f'PWD={password};'
                     f'TrustServerCertificate=yes;')

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


import pyodbc
import pandas as pd
from datetime import datetime

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


def ptu():
    art = [
        r'''
               ...                            
             ;::::;                           
           ;::::; :;                          
         ;:::::'   :;                         
        ;:::::;     ;.                        
       ,:::::'       ;           OOa\         
       ::::::;       ;          OOOOL\        
       ;:::::;       ;         OOOOOOcO       
      ,;::::::;     ;'         / OOOOOaO      
    ;:::::::::`. ,,,;.        /  / DOOOWOO    
  .';:::::::::::::::::;,     /  /     OOAOO   
 ,::::::;::::::;;;;::::;,   /  /        OSOO  
;`::::::`'::::::;;;::::: ,#/  /          DHOO 
:`:::::::`;::::::;;::: ;::#  /            DEOO
::`:::::::`;:::::::: ;::::# /              ORO
`:`:::::::`;:::::: ;::::::#/               DOE
 :::`:::::::`;; ;:::::::::##                OO
 ::::`:::::::`;::::::::;:::#                OO
 `:::::`::::::::::::;'`:;::#                O 
  `:::::`::::::::;' /  / `:#                  
   ::::::`:::::;'  /  /   `#              
'''

    ]
    for line in art:
        print(line)


