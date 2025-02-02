'''
Ask in BTQuant discord kindly for full Binance MsSQL database (100gb~ RAW - 16gb~ compressed)
'''
from fastquant import get_database_data, backtest
data = get_database_data("BTC", "2018-01-01", "2024-08-08", "1m")
backtest("bbands", data, init_cash=1000, backtest=True, plot=True, verbose=True)