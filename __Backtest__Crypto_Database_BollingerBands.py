from fastquant import get_database_data, backtest
crypto = get_database_data("BTC", "2024-01-01", "2024-12-31", "13m")
backtest('bbands', crypto, init_cash=1000, percent_sizer=0.3)