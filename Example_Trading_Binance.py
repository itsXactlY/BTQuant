'''
IMPORTANT NOTE ABOUT IMPORTED STRATEGYS IN THIS FILE - LOAD OR IMPORT ONLY THAT PARTICULAR STRATEGY U USE!
BACKTRADER WARMING UP EVERY POSSIBLE STRATEGY WHAT IS DECLARED AS IMPORT HERE!
CAUSING ALOT OF WARMUP TIME, MEMORY CONSUMPTION, INDICATORS, AND EVERYTHING BEYONED (TIME IS MONEY!)
'''
from backtrader.stores.binance_store import BinanceStore
from backtrader.imports import dt, bt
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK

# JackRabbitRelay
_coin = 'GAS'
_collateral = 'USDT'
_exchange = 'Binance'
_account = '' #_account = '<JackRabbit_SubaccountName>'
_asset = f'{_coin}/{_collateral}'
_amount = '11'
_amount = float(_amount)

def run():
    cerebro = bt.Cerebro(quicknotify=True)
    store = BinanceStore(
        coin_refer=_coin,
        coin_target=_collateral
        )

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