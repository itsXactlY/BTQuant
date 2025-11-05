from backtrader.livetrading import livetrade_ccxt as livetrade
from backtrader.strategies.Aligator_supertrend import AliG_STrend

# CCXT Supported Exchange

_coin = 'XBT'
_collateral = 'USD'
_exchange = 'Bitmex'
_asset = f'{_coin}/{_collateral}'
_amount = '11'
_amount = float(_amount)

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=AliG_STrend,
    exchange=_exchange,
    asset=_asset
)