import backtrader as bt
import joblib

identify = ""
# jrr_webhook_url = "http://65.21.116.91:80"
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = ""
jrr_order_history_sub1 = ""



bsc_privaccount1 = ""
bsc_privaccountaddress = ""

import importlib
import sys
import os
import pandas as pd
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/mssql/MsSQL/build/lib.linux-x86_64-cpython-312/fast_mssql.cpython-312-x86_64-linux-gnu.so'))
spec = importlib.util.spec_from_file_location("fast_mssql", module_path)
fast_mssql = importlib.util.module_from_spec(spec)
sys.modules["fast_mssql"] = fast_mssql
spec.loader.exec_module(fast_mssql)

# SQL Server connection details
server = 'localhost'
database = 'BacktraderData'
username = 'sa'
password = 'J9VcSkC8bqA76MpeP2dxKB'
driver = '{ODBC Driver 18 for SQL Server}'  # Adjust the driver version if necessary

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

