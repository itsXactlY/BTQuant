from backtrader.stores.binance_store import BinanceStore
from imports import dt, bt
# from live_strategys.QQE_Hullband_VolumeOsc import QQE_Example
# from live_strategys.SuperTrend_Scalp import SuperSTrend_Scalper

from live_strategys.NearestNeighbors_RationalQuadraticKernel import NRK

# JackRabbitRelay WIP
_coin = 'GAS'
_collateral = 'USDT'
_exchange = 'binance'
_account = 'binance_debug1' #_account = '<JackRabbit_SubaccountName>'
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
    cerebro.addstrategy(NRK, exchange=_exchange, account=_account, asset=_asset, amount=_amount, coin=_coin, collateral=_collateral, backtest=False)
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)

if __name__ == '__main__':
    run()



#####
# 322MB on startup 24.4.25 01:14
# 