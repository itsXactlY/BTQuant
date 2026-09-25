"""Public config surface for BTQuant. No real values live here.

Secrets are read from the venv: <venv>/etc/btquant/secrets.py (or $BTQ_SECRETS).
Installers/install.sh creates that file (mode 600); edit it there, never here.
"""
import os
import sys

import backtrader as bt

try:
    import fast_mssql
except ImportError:
    fast_mssql = None

# JackRabbit Relay
identify = ""
jrr_webhook_url = ""
jrr_order_history = ""

# Web3
bsc_privaccount1 = ""
bsc_privaccountaddress = ""

# Solana
solana_privkey_base58 = ""
solana_wallet_address = ""

# Discord
discord_webhook_url = ""

# Telegram
telegram_api_id = 1111111
telegram_api_hash = ""
telegram_session_file = ".base.session"
telegram_channel = -100
telegram_channel_debug = -100

# SQL Server
server = "localhost"
candle_database = "BinanceData"
optuna_database = "OptunaBT"
username = "SA"
password = ""
driver = "{ODBC Driver 18 for SQL Server}"

SECRETS_FILE = os.environ.get("BTQ_SECRETS") or os.path.join(sys.prefix, "etc", "btquant", "secrets.py")
if os.path.isfile(SECRETS_FILE):
    _ns = {}
    with open(SECRETS_FILE) as _f:
        exec(compile(_f.read(), SECRETS_FILE, "exec"), _ns)
    globals().update({k: v for k, v in _ns.items() if not k.startswith("_")})
    del _ns, _f

connection_string = (f"DRIVER={driver};SERVER={server};DATABASE={candle_database};"
                     f"UID={username};PWD={password};TrustServerCertificate=yes;")
optuna_connection_string = (f"DRIVER={driver};SERVER={server};DATABASE={optuna_database};"
                            f"UID={username};PWD={password};TrustServerCertificate=yes;")


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
