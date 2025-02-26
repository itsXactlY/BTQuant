from fastquant import livetrade_crypto_binance as livetrade

# Binance.com Exchange
_coin = 'EUR'
_collateral = 'USDT'
_exchange = 'mimic' # JackRabbitRelay Exchange Name
_account = 'binance_sub2' # JackRabbitRelay Account name
_asset = f'{_coin}/{_collateral}'
_amount = '40'
_amount = float(_amount)

livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy="STScalp",
    exchange=_exchange,
    account=_account,
    asset=_asset,
    amount=_amount,
    enable_alerts=True,
    alert_channel=-100123456789 # Telegram channel/group
)
