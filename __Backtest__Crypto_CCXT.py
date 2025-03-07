from fastquant import backtest, get_crypto_data
from fastquant.strategys.NearestNeighbors_RationalQuadraticKernel import NRK
crypto = get_crypto_data("ETH/USDT", "2018-12-01", "2019-01-31", "15m", "kucoin")
backtest(NRK, 
         crypto, 
         period=20, 
         devfactor=2.0, 
         init_cash=1000, 
         backtest=True, 
         plot=True, 
         verbose=3)