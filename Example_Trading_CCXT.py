from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK
from backtrader import livetrading

_coin = 'XRP'
_collateral = 'USDT'
_exchange = 'mexc'
_account = '' # Used for JackRabbitRelay
_asset = f'{_coin}/{_collateral}'

ccxt_config = {
    'apiKey': '',
    'secret': '',
    'enableRateLimit': True,
    'rateLimit': 20,
    'options': {
        'defaultType': 'spot'
    }
}

livetrading.livetrade(coin=_coin,collateral=_collateral, strategy=NRK, asset=_asset, exchange=_exchange, account=_account, config=ccxt_config)