'''
This is a simple example of how to run a backtest using BTQuant with CCXT data, and some of the custom strategies. There is no limit of what BTQ cant do.
More Strategies can be explored in the backtrader/strategies folder.
This example uses the Order_Chain_Kioseff_Trading strategy, but you can also use the VuManchCipher_A and VuManchCipher_B, etc. strategies.
You can also use the get_crypto_data function to get data from any exchange, and any pair.
'''
from backtrader import backtest, get_crypto_data
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK
from backtrader.strategies.Order_Chain_Kioseff_Trading import Order_Chain_Kioseff_Trading
from backtrader.strategies.Vumanchu_A import VuManchCipher_A
from backtrader.strategies.Vumanchu_B import VuManchCipher_B

_coin = "BTC"
_collateral = 'USDT'
_asset = f'{_coin}/{_collateral}'

data = get_crypto_data(_asset,   # Pair
                       "2025-01-01", # Start date
                       "2025-05-15", # End date
                        "1m",        # Timeframe
                       "binance"     # Exchange
                       )

if __name__ == '__main__':
    try:
        backtest(VuManchCipher_A, # Choose your strategy(s) here
                data,
                init_cash=1000, 
                backtest=True, # Important hint :: Needs always set to "False" for live trading
                quantstats=True,
                plot=True,
                asset_name=_asset)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
