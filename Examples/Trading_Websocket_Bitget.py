from backtrader.livetrading import livetrade_bitget as livetrade
from backtrader.strategies.Aligator_supertrend import AliG_STrend

# bitget.com Exchange

_coin = 'BTC'
_collateral = 'USDT'
_exchange = 'mimic' # To Papertrade with REAL Conditions
_account = 'JRR_Subaccount_Name'
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