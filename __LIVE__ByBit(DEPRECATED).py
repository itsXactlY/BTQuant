from fastquant import livetrade_crypto_bybit as livetrade

# ByBit with their horrendous API, im deprecating support for this Exchange as of 2025-02-04
# It only stays here for historical purposes
_coin = "BTC" 
_collateral = "USDT"
_exchange = 'ByBit'
_account = 'JackRabbit_ByBit'
_asset = f'{"$BTC"}/{"USDT"}'
_amount = '0.1'
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
