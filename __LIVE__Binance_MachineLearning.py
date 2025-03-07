from fastquant import livetrade_crypto_binance_ML as livetrade
from fastquant.strategys.NearestNeighbors_RationalQuadraticKernel import NRK

# Binance.com Exchange
_coin = 'BTC'
_collateral = 'USDT'
_exchange = 'mimic' # JackRabbitRelay Exchange Name
_account = 'binance_machinelearning' # JackRabbitRelay Account name
_asset = f'{_coin}/{_collateral}'
_amount = '40'
_amount = float(_amount)

'''
As most Machine learning Strategys need to pre-compute it indicator(s),
and in this fast living world we have no time to sit tight and wait for
2000 Candles~ on LIVE processing to arrive, i created an sneaky workaround
to use the historical DATA fetched via REST Endpoint, run an warmup,
and than go over into the LIVE Websocket FEED, where indicator(s) store
all its values from before.

'''

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=NRK,
    exchange=_exchange,
    account=_account,
    asset=_asset,
    amount=_amount,
    enable_alerts=False,
    alert_channel=-100123456789 # Telegram channel/group
)
