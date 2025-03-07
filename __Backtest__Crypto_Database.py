'''
Ask in BTQuant discord kindly for full Binance candle-database (100gb uncompressed - 16gb compressed)
'''
from fastquant import get_database_data, backtest
from fastquant.strategys.NearestNeighbors_RationalQuadraticKernel import NRK
data = get_database_data("BTC", "2024-07-01", "2024-12-08", "15m")
backtest(NRK, data, init_cash=1000, backtest=True, plot=True, verbose=0)