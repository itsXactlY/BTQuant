from imports import *
from live_strategys.SuperTrend_Scalp import SuperSTrend_Scalper

# JackRabbitRelay WIP
_coin = 'ARKM'
_collateral = 'USDT'
_exchange = 'mimic'
_account = '<JackRabbit_SubaccountName>'
_asset = f'{_coin}/{_collateral}'
_amount = '11' # we use %Sizer
_amount = float(_amount)


def run():
    cerebro = bt.Cerebro(quicknotify=True)
    store = bybit_store.BybitStore(
        coin_refer=_coin,
        coin_target=_collateral
        )

    from_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=6*15)
    data = store.getdata(start_date=from_date)
    data._dataname = f"{_coin}{_collateral}"
    cerebro.addstrategy(SuperSTrend_Scalper, exchange=_exchange, account=_account, asset=_asset, amount=_amount, coin=_coin, collateral=_collateral, backtest=False)
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)


if __name__ == '__main__':
    run()