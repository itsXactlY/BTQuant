import backtrader as bt
from BTQuant_Exchange_Adapters import pancakeswap_store, binance_store, bybit_store, binance_liquidation_store, binance_liquidation_feed
from datetime import datetime, timedelta
import pytz
from fastquant import STRATEGY_MAPPING


# Global variable to store the latest data
latest_data = []

class DataCollectorStrategy(bt.Strategy):
    print("DataCollectorStrategy Init...")
    def next(self):
        global latest_data
        current_time = self.data.datetime.datetime(0)
        data = {
            'time': int(current_time.timestamp()),
            'open': float(self.data.open[0]),
            'high': float(self.data.high[0]),
            'low': float(self.data.low[0]),
            'close': float(self.data.close[0]),
            'volume': float(self.data.volume[0]),
        }
        latest_data.append(data)
        # we keep only the last 1000 data points
        if len(latest_data) > 1000:
            latest_data = latest_data[-1000:]


def livetrade_web3(

    coin: str,
    collateral: str,
    web3ws: str,
    exchange: str,
    account: str,
    asset: str,
    amount: float,
    strategy: str = "",  # Allow passing strategy as a string
    timezone: str = 'Europe/Berlin',
    start_hours_ago: int = 2,
    enable_alerts: bool = False,
) -> None:
    """
    Live trade a strategy on PancakeSwap.

    Args:
    - coin (str): The address of the coin to trade.
    - collateral (str): The address of the collateral coin.
    - web3ws (str): The Web3 WebSocket URL.
    - exchange (str): The exchange to use (e.g., 'pancakeswap').
    - account (str): The account type to use (e.g., 'web3').
    - asset (str): The asset to trade (e.g., '$CAT/wBNB').
    - amount (float): The amount to trade.
    - strategy (bt.Strategy): The strategy to use. Defaults to Pancakeswap_dca_mm.
    - timezone (str): The timezone to use. Defaults to 'Europe/Berlin'.
    - start_hours_ago (int): The number of hours ago to start the data feed. Defaults to 2.
    - enable_alerts (bool): Whether to enable the alert engine (e.g., Telegram/Discord). Defaults to False.
    """

    # Get the strategy class from the mapping if the strategy is passed as a string
    if isinstance(strategy, str):
        strategy_class = STRATEGY_MAPPING.get(strategy)
        if strategy_class is None:
            raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy

    cerebro = bt.Cerebro(quicknotify=True)
    store = pancakeswap_store.PancakeSwapStore(
        coin_refer=coin,
        coin_target=collateral,
        web3ws=web3ws)

    tz = pytz.timezone(timezone)
    utc_now = datetime.now(pytz.utc)
    local_now = utc_now.astimezone(tz)
    from_date = local_now - timedelta(hours=start_hours_ago)

    data = store.getdata(start_date=from_date)
    data._dataname = f"{coin}{collateral}"
    # Add strategy using the resolved strategy class
    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset=asset,
        amount=amount,
        coin=coin,
        collateral=collateral,
        backtest=False,
        enable_alerts=enable_alerts
    )
    
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)


def livetrade_crypto_binance(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    amount: float,
    strategy: str = "",
    start_hours_ago: int = 5,
    enable_alerts: bool = False
) -> None:
    """
    Live trade a strategy on Binance.

    Args:
    - coin (str): The address of the coin to trade.
    - collateral (str): The address of the collateral coin.
    - exchange (str): The exchange to use (e.g., 'binance').
    - account (str): The account type to use (e.g., 'JackRabbit_Binance').
    - asset (str): The asset to trade (e.g., '$BTC/USDT').
    - amount (float): The amount to trade.
    - strategy (str): The strategy name as a string or strategy class.
                    Defaults to "".
    - start_hours_ago (int): The number of hours ago to start the data feed. Defaults to 5.
    - enable_alerts (bool): Whether to enable the alert engine (e.g., Telegram/Discord). Defaults to False.
    """

    # Get the strategy class from the mapping if the strategy is passed as a string
    if isinstance(strategy, str):
        strategy_class = STRATEGY_MAPPING.get(strategy)
        if strategy_class is None:
            raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy

    cerebro = bt.Cerebro(quicknotify=True)
    store = binance_store.BinanceStore(
        coin_refer=coin,
        coin_target=collateral
    )

    # Set the timezone to UTC+2
    tz = pytz.timezone('Europe/Berlin')
    current_time = datetime.now(tz)

    # Add extra buffer time to ensure smooth transition
    buffer_minutes = 2
    from_date = current_time - timedelta(hours=start_hours_ago, minutes=buffer_minutes)

    print(f"Current time (UTC+2): {current_time}")
    print(f"Fetching historical data from (UTC+2): {from_date}")
    
    data = store.getdata(start_date=from_date)
    data._dataname = f"{coin}{collateral}"
    
    cerebro.addstrategy(DataCollectorStrategy)
    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset=asset,
        amount=amount,
        coin=coin,
        collateral=collateral,
        backtest=False,
        enable_alerts=enable_alerts
    )
    
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)


def livetrade_crypto_bybit(

    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    amount: float,
    strategy: str = "",
    timezone: str = 'Europe/Berlin',
    start_hours_ago: int = 2,
    enable_alerts: bool = False,
) -> None:
    """
    Live trade a strategy on PancakeSwap.

    Args:
    - coin (str): The address of the coin to trade.
    - collateral (str): The address of the collateral coin.
    - exchange (str): The exchange to use (e.g., 'pancakeswap').
    - account (str): The account type to use (e.g., 'web3').
    - asset (str): The asset to trade (e.g., '$CAT/wBNB').
    - amount (float): The amount to trade.
    - strategy (bt.Strategy): The strategy to use. Defaults to Pancakeswap_dca_mm.
    - timezone (str): The timezone to use. Defaults to 'Europe/Berlin'.
    - start_hours_ago (int): The number of hours ago to start the data feed. Defaults to 2.
    - enable_alerts (bool): Whether to enable the alert engine (e.g., Telegram/Discord). Defaults to False.
    """

    # Get the strategy class from the mapping if the strategy is passed as a string
    if isinstance(strategy, str):
        strategy_class = STRATEGY_MAPPING.get(strategy)
        if strategy_class is None:
            raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy

    cerebro = bt.Cerebro(quicknotify=True)
    store = bybit_store.BybitStore(
        coin_refer=coin,
        coin_target=collateral)

    tz = pytz.timezone(timezone)
    utc_now = datetime.now(pytz.utc)
    local_now = utc_now.astimezone(tz)
    from_date = local_now - timedelta(hours=start_hours_ago)

    data = store.getdata(start_date=from_date)
    data._dataname = f"{coin}{collateral}"
    
    # Add strategy using the resolved strategy class
    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset=asset,
        amount=amount,
        coin=coin,
        collateral=collateral,
        backtest=False,
        enable_alerts=enable_alerts
    )
    
    cerebro.adddata(data=data, name=data._dataname)
    print(f"Daten vorhanden? {len(cerebro.datas) > 0}")
    cerebro.run(live=True)