from backtrader.livetrading import livetrade_ccxt as livetrade
from backtrader.strategies.Aligator_supertrend import AliG_STrend
from backtrader.ccxt_config import load_ccxt_config

_coin = 'BTC'
_collateral = 'USDT'
_exchange = 'binance'      # ccxt id -> lowercase
_account = 'main'
_asset = f'{_coin}/{_collateral}'

# this reads <venv>/ccxt/binance_main.json (+ env overrides)
binance_main = load_ccxt_config(exchange=_exchange, account=_account)

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=AliG_STrend,
    exchange=_exchange,
    account=_account,
    asset=_asset,
    config=binance_main,
)