from fastquant import backtest, get_crypto_data
crypto = get_crypto_data("ETH", "2018-12-01", "2019-12-31", "15m")
backtest('bbands', crypto, period=20, devfactor=2.0)