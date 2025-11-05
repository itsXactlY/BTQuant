from backtrader.livetrading import livetrade_mexc as livetrade
from backtrader.strategies.Aligator_supertrend import AliG_STrend

# mexc.com Exchange - Raltime TickData - no fixed feed.

_coin = 'BTC'
_collateral = 'USDT'
_exchange = 'mexc' 
_account = ''
_asset = f'{_coin}/{_collateral}'
_amount = '11'
_amount = float(_amount)

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=AliG_STrend,
    exchange=_exchange,
    account=_account,
    asset=_asset
)