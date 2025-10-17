from backtrader.strategies.ST_RSX_ASI import STrend_RSX_AccumulativeSwingIndex
from backtrader.strategies.StagedConvergenceStrategy import StagedConvergenceStrategy
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK
from backtrader.strategies.QQE_Hullband_VolumeOsc import QQE_Example
from backtrader.utils.backtest import backtest

# BTQ DCA Strategy based on your enhanced indicators


if __name__ == '__main__':
    try:
        backtest(
            STrend_RSX_AccumulativeSwingIndex,
            coin='BTC',
            collateral='USDT',
            start_date="2024-01-01",
            end_date="2024-02-15",
            interval="1m",
            init_cash=1000, 
            plot=True,
            quantstats=False
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
