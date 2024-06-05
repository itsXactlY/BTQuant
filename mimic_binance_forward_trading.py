from imports import *
from live_strategys.QQE_Hullband_VolumeOsc import *


# JackRabbitRelay
_coin = 'PEPE'
_collateral = 'USDT'
_exchange = 'mimic'
_account = 'JackRabbit_SubaccountName' # Change me, im a dummy name :)
_asset = f'{_coin}/{_collateral}'
_amount = '69' # Dollar amount
_amount = float(_amount)



def run():
    cerebro = bt.Cerebro(quicknotify=True)

    store = binance_store.BinanceStore(
        api_key=api,
        api_secret=sec,
        coin_refer=_coin,
        coin_target=_collateral,
        testnet=False)
    
    broker = store.getbroker()
    cerebro.setbroker(broker)

    broker = store.getbroker()
    cerebro.setbroker(broker)

    # Fix datetime usage
    from_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=6*15)
    data = store.getdata(
        timeframe_in_minutes=1,
        start_date=from_date)
    
    data._dataname = f"{_coin}{_collateral}"

    cerebro.addstrategy(QQE_Example, exchange=_exchange, account=_account, asset=_asset, amount=_amount, coin=_coin, collateral=_collateral, backtest=False)
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run()


if __name__ == '__main__':
    run()