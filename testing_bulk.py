from backtrader.utils.backtest import backtest, optimize_backtest, bulk_backtest

# from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK as strategy
# from backtrader.strategies.ST_RSX_ASI import STrend_RSX_AccumulativeSwingIndex as strategy
# from backtrader.strategies.Order_Chain_Kioseff_Trading import Order_Chain_Kioseff_Trading
# from backtrader.strategies.Vumanchu_A import VuManchCipher_A
# from backtrader.strategies.Vumanchu_B import VuManchCipher_B as strategy
from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX2 as strategy
from backtrader.dontcommit import connection_string, fast_mssql, bt
from backtrader.feeds.mssql_crypto import MSSQLData

if __name__ == "__main__":
    try:
        # coins = MSSQLData.get_all_pairs(connection_string)

        results = bulk_backtest(
            strategy=strategy,
            # coins=coins,
            start_date="2025-01-01",
            end_date="2025-01-02",
            interval="4h",
            collateral="USDT",
            init_cash=1000,
            max_workers=8,
            output_file="backtest_results.json"
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
