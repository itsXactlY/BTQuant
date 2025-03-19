from fastquant import livetrade_crypto_bitget as livetrade
from fastquant.strategies.SuperTrend_Scalp import SuperSTrend_Scalp


_coin = 'PI'
_collateral = 'USDT'
_exchange = 'mimic' # JackRabbitRelay Exchange Name
_account = 'bitget' # JackRabbitRelay Account name
_asset = f'{_coin}/{_collateral}'

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=SuperSTrend_Scalp,
    exchange=_exchange,
    account=_account,
    asset=_asset,
    enable_alerts=False,
    alert_channel=-100123456789
)