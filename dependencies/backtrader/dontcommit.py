import backtrader as bt
import joblib
import os
import sys

# JackRabbit Relay
identify = "" # Fill Identify string from JRR Setup
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = "/home/JackrabbitRelay2/Data/Mimic/"

# Web3
bsc_privaccount1 = ""
bsc_privaccountaddress = ""

# solana
solana_privkey_base58 = ""
solana_wallet_address = ""

#Discord
discord_webhook_url = '' #'https://discord.com/api/webhooks/...'

# Telegram
telegram_api_id = 1111111
telegram_api_hash = ""
telegram_session_file = ".base.session"
telegram_channel = -100

# fast_mssql shim — drop-in for the C++ fast_mssql module
# The shim is in the mssql/ dir which is on sys.path
import sys
_fast_mssql_path = __file__.rsplit('/', 1)[0] + '/feeds/mssql'
if _fast_mssql_path not in sys.path:
    sys.path.insert(0, _fast_mssql_path)
import fast_mssql

# SQL Server connection details.
#
# This file is git-tracked, so it must stay VALUE-FREE. Real credentials are
# resolved at import time by btq_secrets.py (repo root) from a local, mode-600
# secrets file outside the repo. Resolution order:
#   1. $BTQ_SECRETS                      explicit override
#   2. <venv>/etc/btquant/secrets.py     tpad layout (sys.prefix)
#   3. ~/.btq/etc/btquant/secrets.py     home fallback (no venv)
# If no secrets file is found the public defaults below are used, so importing
# this module never fails — connections just fail auth until you set it up.
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
try:
    import btq_secrets
    _db = btq_secrets.load()
except ImportError:  # repo not present (e.g. installed backtrader wheel)
    btq_secrets = None
    _db = {
        "server": "localhost", "candle_database": "BinanceData",
        "optuna_database": "OptunaBT", "username": "SA", "password": "",
        "driver": "{ODBC Driver 18 for SQL Server}",
    }

SECRETS_FILE = _db.get("_secrets_file")

server = _db["server"]
candle_database = _db["candle_database"]
optuna_database = _db["optuna_database"]
marketdata_database = _db.get("marketdata_database", "BTQ_MarketData")
username = _db["username"]
password = _db["password"]
driver = _db["driver"]  # Adjust the driver version if necessary

# Back-compat alias: older callers import `database`.
database = candle_database


def _conn(db_name):
    """Build an ODBC connection string for the given database."""
    return (f'DRIVER={driver};'
            f'SERVER={server};'
            f'DATABASE={db_name};'
            f'UID={username};'
            f'PWD={password};'
            f'TrustServerCertificate=yes;')


connection_string = _conn(candle_database)
optuna_connection_string = _conn(optuna_database)
marketdata_connection_string = _conn(marketdata_database)


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
