from fastquant import get_database_stock_data, backtest
crypto = get_database_stock_data("EUR_USD", "2012-01-01", "2024-12-31", "1m")
backtest('bbands', crypto, init_cash=1000, percent_sizer=0.3)