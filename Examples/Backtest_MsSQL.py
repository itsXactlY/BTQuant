# from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK
from backtrader.strategies.ST_RSX_ASI import STrend_RSX_AccumulativeSwingIndex
from backtrader.utils.backtest import backtest
from backtrader.feeds.mssql_crypto import get_database_data

_coin = "XRP"
_collateral = 'USDT'
_asset = f'{_coin}/{_collateral}'

data = get_database_data(_coin, "2024-06-01", "2025-01-08", "1m")

if __name__ == '__main__':
    try:
        backtest(STrend_RSX_AccumulativeSwingIndex, data, init_cash=1000, backtest=True, quantstats=True, plot=True, asset_name=_asset)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
