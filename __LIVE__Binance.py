from fastquant import livetrade_crypto_binance as livetrade
from fastquant.strategies.SuperTrend_Scalp import SuperSTrend_Scalper

# Binance.com Exchange
_coin = 'EUR'
_collateral = 'USDT'
_exchange = 'mimic' # JackRabbitRelay Exchange Name
_account = 'binance_sub2' # JackRabbitRelay Account name
_asset = f'{_coin}/{_collateral}'

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=SuperSTrend_Scalper,
    exchange=_exchange,
    account=_account,
    asset=_asset,
    enable_alerts=False,
    alert_channel=-100123456789 # Telegram channel/group
)
