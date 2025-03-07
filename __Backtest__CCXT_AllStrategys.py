from fastquant import backtest, get_crypto_data
from fastquant.strategys.Aligator_supertrend import AliG_STrend
from fastquant.strategys.QQE_Hullband_VolumeOsc import QQE_Example
from fastquant.strategys.Order_Chain_Kioseff_Trading import Order_Chain_Kioseff_Trading
from fastquant.strategys.SMA_Cross_MESAdaptive_Prime import SMA_Cross_MESAdaptivePrime
from fastquant.strategys.SuperTrend_Scalp import SuperSTrend_Scalper
from fastquant.strategys.NearestNeighbors_RationalQuadraticKernel import NRK
data_cache = None

def get_data():
    return get_crypto_data("BTC/USDT", "2024-01-01", "2024-02-01", "15m")

def run_backtest():
    global data_cache
    if data_cache is None:
        data_cache = get_data()

    backtest(AliG_STrend, data_cache, init_cash=1000, backtest=True, plot=True, verbose=0)
    backtest(QQE_Example, data_cache, init_cash=1000, backtest=True, plot=True, verbose=0)
    backtest(Order_Chain_Kioseff_Trading, data_cache, init_cash=1000, backtest=True, plot=True, verbose=1)
    backtest(SMA_Cross_MESAdaptivePrime, data_cache, init_cash=1000, backtest=True, plot=True, verbose=0)
    backtest(SuperSTrend_Scalper, data_cache, init_cash=1000, backtest=True, plot=True, verbose=0)
    backtest(NRK, data_cache, init_cash=1000, backtest=True, plot=True, verbose=0)

if __name__ == "__main__":
    run_backtest()