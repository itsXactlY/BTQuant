from fastquant import livetrade_web3 as livetrade
from fastquant.strategys.pancakeswap_dca_marketmaker import Pancakeswap_dca_mm
# I have no dang idea why this is one and only strategy yet what will not work with the strategy_mapping.
# Im on it to figure out.

# Web3 Decentral Exchanges
_coin = "0xb265cba9bd6e34fa412bd8b4c1514c902c0e7e7d" #Giggle Academy
_collateral = "0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c" # WRAPPED BNB
_exchange = 'pancakeswap'
_websocket = "wss://bsc-rpc.publicnode.com"
_account = 'web3'
_asset = f'{"$Giggle"}/{"wBNB"}'
_amount = '0.00042'
_amount = float(_amount)


livetrade(
    coin=_coin,
    collateral=_collateral,
    strategy=Pancakeswap_dca_mm,
    exchange=_exchange,
    web3ws=_websocket,
    account=_account,
    asset=_asset,
    amount=_amount,
    enable_alerts = False
)
