import backtrader as bt
from backtrader.stores.tv_store import TradingViewStore
from backtrader.strategies.Aligator_supertrend import AliG_STrend
from backtrader.imports import dt, bt

# Setup your parameters here
_coin = '0x2c3a8Ee94dDD97244a93Bc48298f97d2C412F7Db'  # AKE
_collateral = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c' # WBNB
_exchange = 'pancakeswap'
_asset = f'{_coin}/{_collateral}'
_amount = '0.05'
_amount = float(_amount)
_account = 'web3' # Used for JackRabbitRelay's accountname and/or decentral web3 trading

def run():
    cerebro = bt.Cerebro(quicknotify=True, live=True, preload=False)
    store = TradingViewStore(symbol="AKEWBNB_4D3BF2.USD") # Possibilitys: BINANCE:BTCUSDT, eurex:NKEF1!, blackbull:BRENT, comex:GC1!, coinbase:SOLV2025, TAIFEX:FKF1!

    from_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)
    data = store.getdata(start_date=from_date)
    data._dataname = f"{_coin}{_collateral}"
    cerebro.addstrategy(AliG_STrend,
                        exchange=_exchange,
                        account=_account,
                        asset=_asset,
                        amount=_amount,
                        coin=_coin,
                        collateral=_collateral,
                        enable_alerts=False,
                        alert_channel=-100,
                        backtest=False,
                        debug=True)
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        raise e