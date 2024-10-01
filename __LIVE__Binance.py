from fastquant import livetrade_crypto_binance as livetrade

# Web3 Decentral Exchanges
_coin = "BTC" 
_collateral = "USDT"
_exchange = 'Binance'
_account = 'JackRabbit_Binance'
_asset = f'{"$BTC"}/{"USDT"}'
_amount = '0.1'
_amount = float(_amount)


livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy="qqe",
    exchange=_exchange,
    account=_account,
    asset=_asset,
    amount=_amount
)
