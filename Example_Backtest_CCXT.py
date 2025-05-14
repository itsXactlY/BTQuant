from backtrader import backtest, get_crypto_data
from backtrader.strategies.QQE_Hullband_VolumeOsc import QQE_Example

data = get_crypto_data("ETH/USDT", "2018-12-01", "2019-01-31", "15m", "kucoin")
backtest(QQE_Example, 
        data,
        init_cash=1000, 
        backtest=True, 
        plot=True)