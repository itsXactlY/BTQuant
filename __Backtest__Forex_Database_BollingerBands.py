'''
5 second data can be found at https://www.patreon.com/RD3277/shop/eur-usd-5-second-ohlcv-data-set-541738
'''
from fastquant import get_database_stock_data, backtest
stocks = get_database_stock_data("EUR_USD", "2005-01-01", "2024-10-10", "5s")
backtest("bbands", stocks, init_cash=1000, backtest=True, plot=True, verbose=True, commission=0)