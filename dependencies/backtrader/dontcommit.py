import backtrader as bt
import joblib

# JackRabbit Relay
identify = "" # Fill Identify string from JRR Setup
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = "/home/JackrabbitRelay2/Data/Mimic/"

# Web3
bsc_privaccount1 = ""
bsc_privaccountaddress = ""

#Discord
discord_webhook_url = '' #'https://discord.com/api/webhooks/...'

# Telegram
telegram_api_id = 1111111
telegram_api_hash = ""
telegram_session_file = ".base.session"
telegram_channel = -100

# Precompiled C++ MSSQL Adapter
import importlib
import sys
import os
import pandas as pd
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feeds/mssql/fast_mssql.cpython-313-x86_64-linux-gnu.so')) # For Python 3.12 change to cpython-312
spec = importlib.util.spec_from_file_location("fast_mssql", module_path)
fast_mssql = importlib.util.module_from_spec(spec)
sys.modules["fast_mssql"] = fast_mssql
spec.loader.exec_module(fast_mssql)

# SQL Server connection details
server = 'localhost'
database = 'Backtrader'
username = 'username'
password = 'password'
driver = '{ODBC Driver 18 for SQL Server}'  # Adjust the driver version if necessary

connection_string = (f'DRIVER={driver};'
                     f'SERVER={server};'
                     f'DATABASE={database};'
                     f'UID={username};'
                     f'PWD={password};'
                     f'TrustServerCertificate=yes;')


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
