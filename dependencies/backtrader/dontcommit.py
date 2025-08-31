import backtrader as bt
import joblib
import fast_mssql

### Live trading / paper trading settings

# JackRabbit Relay
identify = "" # Fill Identify string from JRR Setup
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = "/home/JackrabbitRelay2/Data/Mimic/"

# Web3 / BSC TODO :: Move to secrets file, add full documentation
bsc_privaccount1 = ""
bsc_privaccountaddress = ""

### Alerts

# Discord
discord_webhook_url = '' #'https://discord.com/api/webhooks/...'

# Telegram
telegram_api_id = 1111111
telegram_api_hash = ""
telegram_session_file = ".base.session"
telegram_channel = -100

### Backtesting settings
# SQL Server connection details
server = 'localhost'
candle_database = 'BinanceData'
optuna_database = 'OptunaBT'
username = 'SA'
password = 'YourStrong!Passw0rd'
driver = '{ODBC Driver 18 for SQL Server}'  # Adjust the driver version if necessary

connection_string = (f'DRIVER={driver};'
                     f'SERVER={server};'
                     f'DATABASE={candle_database};'
                     f'UID={username};'
                     f'PWD={password};'
                     f'TrustServerCertificate=yes;')

optuna_connection_string = (f'DRIVER={driver};'
                     f'SERVER={server};'
                     f'DATABASE={optuna_database};'
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
