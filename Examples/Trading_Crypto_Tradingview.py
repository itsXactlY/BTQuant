'''
IMPORTANT NOTE ABOUT IMPORTED STRATEGYS IN THIS FILE - LOAD OR IMPORT ONLY THAT PARTICULAR STRATEGY U USE!
BACKTRADER WARMING UP EVERY POSSIBLE STRATEGY WHAT IS DECLARED AS IMPORT HERE!
CAUSING ALOT OF WARMUP TIME, MEMORY CONSUMPTION, INDICATORS, AND EVERYTHING BEYONED (TIME IS MONEY!)
'''
from backtrader.stores.tv_store import TradingViewStore
from backtrader.imports import dt, bt
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK

# Setup your parameters here
_coin = 'BTC'
_collateral = 'USDT'
_exchange = 'mimic'
_asset = f'{_coin}/{_collateral}'
_symbol = f'{_coin}{_collateral}'
print(_symbol)
_amount = '11'
_amount = float(_amount)
_account = '' # Used for JackRabbitRelay

def run():
    cerebro = bt.Cerebro(quicknotify=True)
    store = TradingViewStore(symbol=_symbol)

    from_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=6*15)
    data = store.getdata(start_date=from_date)
    data._dataname = f"{_coin}{_collateral}"
    cerebro.addstrategy(NRK,
                        exchange=_exchange,
                        account=_account,
                        asset=_asset,
                        amount=_amount,
                        coin=_coin,
                        collateral=_collateral,
                        enable_alerts=False,
                        alert_channel=-100,
                        backtest=False)
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()