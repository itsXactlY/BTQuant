import backtrader as bt
from datetime import datetime, timedelta
import pytz
from typing import Type

def livetrade_ccxt(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: Type[bt.Strategy],
    config: str,
) -> None:
    
    if strategy is None:
        raise ValueError("No strategy class provided.")

    from backtrader.feeds.ccxt import CCXT
    from backtrader.brokers.ccxtbroker import CCXTBroker

    cerebro = bt.Cerebro()
    
    # Create broker with patched class
    broker = CCXTBroker(
        exchange,
        currency='USDT',
        config=config
    )

    broker = CCXTBroker(exchange, collateral, config)
    cerebro.setbroker(broker)    
    data = CCXT(
        exchange=exchange,
        symbol=asset,
        ohlcv_limit=500,
        config=config,
        retries=5
    )

    cerebro.setbroker(broker)
    cerebro.adddata(data, name=data._dataname)
    cerebro.addstrategy(
                        strategy,
                        exchange=exchange,
                        account=account,
                        asset=asset,
                        coin=coin,
                        collateral=collateral,
                        enable_alerts=False,
                        backtest=False)

    try:
        cerebro.run(live=True, runonce=False, exactbars=False, stdstats=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

def livetrade_web3(
    coin: str,
    collateral: str,
    web3ws: str,
    exchange: str,
    account: str,
    asset: str,
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

    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy

    from backtrader.stores import pancakeswap_store

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
        coin=coin,
        collateral=collateral,
        backtest=False,
        enable_alerts=enable_alerts
    )
    
    cerebro.adddata(data=data, name=data._dataname)
    
    try:
        cerebro.run(live=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

def livetrade_binance(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = "",
) -> None:
    """
    Live trade a strategy on Binance.
    Args:
    - coin (str): The address of the coin to trade.
    - collateral (str): The address of the collateral coin.
    - exchange (str): The exchange to use (e.g., 'binance').
    - account (str): The account type to use (e.g., 'JackRabbit_Binance').
    - asset (str): The asset to trade (e.g., '$BTC/USDT').
    - strategy (str): The strategy name as a string or strategy class.
      Defaults to "".
    - start_hours_ago (int): The number of hours ago to start the data feed. Defaults to 1.
    - enable_alerts (bool): Whether to enable the alert engine. Defaults to False.
    - alert_channel (str): Define where to send alerts via alert engine.
    """
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy
    
    from backtrader.stores import binance_store
    
    # Initialize Binance store
    store = binance_store.BinanceStore(
        coin_refer=coin,
        coin_target=collateral
    )
    
    # Set time range for fetching historical data (for indicator warmup)
    tz = pytz.timezone('Europe/Berlin')
    current_time = datetime.now(tz)
    from_date = current_time - timedelta(hours=start_hours_ago, minutes=2)
    print(f"Fetching historical data from: {from_date} (UTC+2)")
    
    data = store.getdata(start_date=from_date)
    data._dataname = f"{coin}{collateral}"
    
    # Create Cerebro instance with memory saving options
    cerebro = bt.Cerebro(quicknotify=True)
    
    # Add the combined (historical + live) data feed
    cerebro.adddata(data, name=data._dataname)
    
    # Add strategy with all necessary parameters
    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset=asset,
        coin=coin,
        collateral=collateral,
        backtest=False,
        enable_alerts=enable_alerts,
        alert_channel=alert_channel
    )
    
    # Run live trading with memory saving
    # print(f"Starting live trading with historical backfill (memory saving level: {memory_saving})...")
    data.live = True
    
    # Apply memory saving settings
    cerebro.run(live=True, exactbars=100)

def livetrade_mexc(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = ""
) -> None:
    """
    Live trade a strategy on Mexc.

    Args:
    - coin (str): The address of the coin to trade.
    - collateral (str): The address of the collateral coin.
    - exchange (str): The exchange to use (e.g., 'mexc').
    - account (str): The account type to use (e.g., 'JackRabbit_Mexc').
    - asset (str): The asset to trade (e.g., '$BTC/USDT').
    - amount (float): The amount to trade.
    - strategy (str): The strategy name as a string or strategy class.
                    Defaults to "".
    - start_hours_ago (int): The number of hours ago to start the data feed. Defaults to 5.
    - enable_alerts (bool): Whether to enable the alert engine (e.g., Telegram/Discord). Defaults to False.
    - alert_channel (str): Define where to send alerts via alert engine to corresponding channels.
    """

    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy

    from backtrader.stores import mexc_store

    cerebro = bt.Cerebro(quicknotify=True)
    store = mexc_store.MexcStore(
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
    
    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset=asset,
        coin=coin,
        collateral=collateral,
        backtest=False,
        enable_alerts=enable_alerts,
        alert_channel=alert_channel
    )
    
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)

def livetrade_bitget(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = ""
) -> None:
    """
    Live trade a strategy on Bitget.

    Args:
    - coin (str): The address of the coin to trade.
    - collateral (str): The address of the collateral coin.
    - exchange (str): The exchange to use (e.g., 'bitget').
    - account (str): The account type to use (e.g., 'JackRabbit_bitget').
    - asset (str): The asset to trade (e.g., 'BTCUSDT').
    - amount (float): The amount to trade.
    - strategy (str): The strategy name as a string or strategy class.
                    Defaults to "".
    - start_hours_ago (int): The number of hours ago to start the data feed. Defaults to 5.
    - enable_alerts (bool): Whether to enable the alert engine (e.g., Telegram/Discord). Defaults to False.
    - alert_channel (str): Define where to send alerts via alert engine to corresponding channels.
    """

    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy

    from backtrader.stores import bitget_store

    cerebro = bt.Cerebro(quicknotify=True)
    store = bitget_store.BitgetStore(
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
    
    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset=asset,
        coin=coin,
        collateral=collateral,
        backtest=False,
        enable_alerts=enable_alerts,
        alert_channel=alert_channel
    )
    
    cerebro.adddata(data=data, name=data._dataname)
    cerebro.run(live=True)

def livetrade_tv(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: Type[bt.Strategy],
    tv_symbol: str,
    amount: float = 0.05,
    lookback_minutes: int = 1,
) -> None:
    """
    Live trading using TradingView as data source.
    """
    
    if strategy is None:
        raise ValueError("No strategy class provided.")
    
    from backtrader.stores.tv_store import TradingViewStore
    import datetime as dt
    
    # Create Cerebro instance with live trading parameters
    cerebro = bt.Cerebro(quicknotify=True, preload=False)
    
    # Initialize TradingView store
    store = TradingViewStore(symbol=tv_symbol)
    
    # Calculate start date for historical data
    from_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=lookback_minutes)
    
    # Get data feed from TradingView
    data = store.getdata(start_date=from_date)
    data._dataname = f"{coin}{collateral}"
    
    # Add symbol attribute for strategy compatibility
    data.symbol = asset
    data.params.symbol = asset
    
    # Add strategy with parameters
    cerebro.addstrategy(
        strategy,
        exchange=exchange,
        account=account,
        asset=asset,
        amount=amount,
        coin=coin,
        collateral=collateral,
        enable_alerts=False,
        backtest=False
    )
    
    # Add data feed
    cerebro.adddata(data=data, name=data._dataname)
    
    # Run live trading
    try:
        cerebro.run(live=True, runonce=False, exactbars=False, stdstats=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()


''' Experimental WIP '''
''' Strategy example still WIP in debugging state '''
def livetrade_multiple_pairs(
    pairs: list,
    exchange: str,
    account: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = ""
) -> None:
    cerebro = bt.Cerebro(quicknotify=True)
    
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy
    
    from backtrader.stores import bitget_store

    asset_mapping = {}
    
    for pair in pairs:
        store = bitget_store.BitgetStore(
            coin_refer=pair['coin'],
            coin_target=pair['collateral']
        )

        tz = pytz.timezone('Europe/Berlin')
        current_time = datetime.now(tz)
        buffer_minutes = 2
        from_date = current_time - timedelta(hours=start_hours_ago, minutes=buffer_minutes)
        data = store.getdata(start_date=from_date)

        data_name = f"{pair['coin']}{pair['collateral']}"
        data._dataname = data_name
        cerebro.adddata(data=data, name=data._dataname)
        asset_mapping[data_name] = pair["asset"]

    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset_mapping=asset_mapping,
        backtest=False,
        enable_alerts=enable_alerts,
        alert_channel=alert_channel
    )
    
    cerebro.run(live=True)
