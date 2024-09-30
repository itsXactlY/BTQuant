import backtrader as bt
from BTQuant_Exchange_Adapters import pancakeswap_store
import datetime as dt
from fastquant.strategies.pancakeswap_dca_marketmaker import Pancakeswap_dca_mm
import pytz

# Web3 Decentral Exchanges
_coin = "0x6894cde390a3f51155ea41ed24a33a4827d3063d" # Simon the CAT
_collateral = "0xBB4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c" # WRAPPED BNB
_exchange = 'pancakeswap'
_account = 'web3'
_asset = f'{"$CAT"}/{"wBNB"}'
_amount = '0.00042'
_amount = float(_amount)


def run():
    cerebro = bt.Cerebro(quicknotify=True)
    store = pancakeswap_store.PancakeSwapStore(
        coin_refer=_coin,
        coin_target=_collateral)

    # Set your desired timezone
    timezone = pytz.timezone('Europe/Berlin')  # e.g., 'America/New_York' or 'Europe/London'

    # Get current time in UTC
    utc_now = dt.datetime.now(pytz.utc)

    # Convert UTC time to your desired timezone
    local_now = utc_now.astimezone(timezone)

    # Calculate the start date
    from_date = local_now - dt.timedelta(hours=2)

    data = store.getdata(start_date=from_date)
    data._dataname = f"{_coin}{_collateral}"
    cerebro.addstrategy(Pancakeswap_dca_mm, exchange=_exchange, account=_account, asset=_asset, amount=_amount, coin=_coin, collateral=_collateral, backtest=False)
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)

if __name__ == '__main__':
    run()