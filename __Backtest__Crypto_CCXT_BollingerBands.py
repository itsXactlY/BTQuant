from fastquant import backtest, get_database_stock_data
crypto = get_database_stock_data("EUR_USD", "2018-12-01", "2019-12-31", "15m")
backtest('bbands', crypto, period=20, devfactor=2.0)