from fastquant import livetrade_crypto_binance as livetrade

# Web3 Decentral Exchanges
_coin = 'BTC'
_collateral = 'USDT'
_exchange = 'mimic'
_account = 'binance_sub2'
_asset = f'{_coin}/{_collateral}'
_amount = '11'
_amount = float(_amount)


livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy="STScalp",
    exchange=_exchange,
    account=_account,
    asset=_asset,
    amount=_amount
)
