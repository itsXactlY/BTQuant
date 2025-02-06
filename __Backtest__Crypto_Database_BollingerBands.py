'''
Ask in BTQuant discord kindly for full Binance candle-database (100gb uncompressed - 16gb compressed)
'''
from fastquant import get_database_data, backtest
data = get_database_data("BTC", "2024-07-01", "2024-08-08", "1m")
backtest("bbands", data, init_cash=1000, backtest=True, plot=True, verbose=0)