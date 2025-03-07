from fastquant import livetrade_crypto_mexc as livetrade
from fastquant.strategys.SuperTrend_Scalp import SuperSTrend_Scalper

# Mexc.com Exchange
_coin = 'PI'
_collateral = 'USDT'
_exchange = 'mimic' # JackRabbitRelay Exchange Name
_account = 'mexc_testing' # JackRabbitRelay Account name
_asset = f'{_coin}/{_collateral}'
_amount = '40'
_amount = float(_amount)

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=SuperSTrend_Scalper,
    exchange=_exchange,
    account=_account,
    asset=_asset,
    amount=_amount,
    enable_alerts=True,
    alert_channel=-100123456789 # Telegram channel/group
)
