from fastquant import livetrade_solana as livetrade
from fastquant.strategies.pancakeswap_dca_marketmaker import Pancakeswap_dca_mm
# I have no dang idea why this is one and only strategy yet what will not work with the strategy_mapping.
# Im on it to figure out.

# Web3 Decentral Exchanges
_coin = "7ZqzGzTNg5tjK1CHTBdGFHyKjBtXdfvAobuGgdt4pump" #Giggle Academy
_exchange = 'raydium'
_account = 'sol'
_asset = f'{"$Giggle"}/{"wBNB"}'
_amount = '0.00042'
_amount = float(_amount)


livetrade(
    coin=_coin,
    strategy=Pancakeswap_dca_mm,
    exchange=_exchange,
    account=_account,
    asset=_asset,
    amount=_amount
)
