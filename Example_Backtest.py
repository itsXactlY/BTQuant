'''
############## IMPORTANT NOTE ABOUT IMPORTED STRATEGYS IN THIS FILE - LOAD OR IMPORT ONLY THAT PARTICULAR STRATEGY U USE! ##############
############## BACKTRADER WARMING UP EVERY POSSIBLE STRATEGY WHAT IS DECLARED AS IMPORT HERE! ##############
############## CAUSING ALOT OF WARMUP TIME, MEMORY CONSUMPTION, INDICATORS, AND EVERYTHING BEYONED (TIME IS MONEY!) ##############
'''

from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK
from backtrader.utils.backtest import backtest
from backtrader.feeds.mssql_crypto import get_database_data

data = get_database_data("BTC", "2024-01-01", "2024-04-08", "1m")

def run_backtest():
    backtest(NRK, data, init_cash=1000, backtest=True, plot=True)

if __name__ == '__main__':
    try:
        run_backtest()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
