import backtrader as bt
from datetime import datetime, timedelta
import pytz
<<<<<<< HEAD

from typing import Type, Optional, Dict, Any
import backtrader as bt

from .ccxt_config import load_ccxt_config
from backtrader.feeds.ccxt import CCXT
from backtrader.brokers.ccxtbroker import CCXTBroker


def livetrade_ccxt(
=======
from typing import Type


def livetrade(
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
<<<<<<< HEAD
    strategy_class: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
        strategy_class = strategy

    if config is None:
        config = load_ccxt_config(exchange=exchange, account=account, default_type="spot")

    cerebro = bt.Cerebro()

    broker = CCXTBroker(exchange, collateral, config)
    cerebro.setbroker(broker)

=======
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    data = CCXT(
        exchange=exchange,
        symbol=asset,
        ohlcv_limit=500,
        config=config,
<<<<<<< HEAD
        retries=5,
    )

    cerebro.adddata(data, name=data._dataname)
    cerebro.addstrategy(
        strategy_class,
        exchange=exchange,
        account=account,
        asset=asset,
        coin=coin,
        collateral=collateral,
        enable_alerts=False,
        backtest=False,
    )
=======
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

    try:
        cerebro.run(live=True, runonce=False, exactbars=False, stdstats=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
<<<<<<< HEAD
        traceback.print_exc()


=======
        print("Full traceback:")
        traceback.print_exc()

>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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

<<<<<<< HEAD
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
=======
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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
<<<<<<< HEAD


def livetrade_hotspine(
    symbol_id: int,
    strategy_class,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001,
    **strategy_params
) -> None:
    """
    Live trade a strategy using HotSpine as the data source.
    
    This function provides live trading capabilities using HotSpine shared memory
    for ultra-low latency trade data while maintaining compatibility with
    Backtrader's trading engine.
    
    Args:
        symbol_id: Symbol ID to filter trades from HotSpine
        strategy_class: Strategy class or instance
        shm_name: Shared memory segment name (default: "/btquant_hotspine")
        batch_mode: Whether to use batch reading for higher throughput
        poll_interval: Polling interval in seconds
        **strategy_params: Additional parameters to pass to the strategy
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Resolve strategy class
    if isinstance(strategy_class, str):
        strategy_class = strategy_class.lower()
    elif not callable(strategy_class):
        raise ValueError(f"Invalid strategy: {strategy_class}")
    
    # Create Cerebro instance for live trading
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)  # Enable cheat-on-close for live trading
    
    # Create HotSpine data feed
    from backtrader.feeds.hotspine_feed import HotSpineData
    
    data = HotSpineData(
        symbol_id=symbol_id,
        shm_name=shm_name,
        batch_mode=batch_mode,
        poll_interval=poll_interval
    )
    
    # Set data name for identification
    data._dataname = f"hotspine_{symbol_id}"
    data._name = f"HotSpine_{symbol_id}"
    
    # Add data feed to Cerebro
    cerebro.adddata(data, name=data._dataname)
    
    # Add strategy with parameters
    cerebro.addstrategy(
        strategy_class,
        backtest=False,
        live=True,
        **strategy_params
    )
    
    # Configure Cerebro for live trading
    cerebro.broker.set_cash(1000.0)  # Default starting cash
    
    try:
        logger.info(f"Starting HotSpine live trading for symbol_id={symbol_id}")
        logger.info(f"Shared memory: {shm_name}")
        logger.info(f"Batch mode: {batch_mode}")
        logger.info(f"Poll interval: {poll_interval}s")
        
        # Run live trading
        cerebro.run(
            live=True,
            runonce=False,  # Live trading requires event-based processing
            exactbars=False,  # Don't require exact bar counts for live data
            stdstats=False,  # Disable standard stats for live trading
            preload=False,  # Don't preload live data
            quicknotify=True  # Enable quick notifications
        )
        
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    except Exception as e:
        logger.error(f"Live trading failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def livetrade_hotspine_multi_symbol(
    symbol_ids: list,
    strategy,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001,
    **strategy_params
) -> None:
    """
    Live trade a strategy using multiple HotSpine symbols.
    
    Args:
        symbol_ids: List of symbol IDs to trade
        strategy: Strategy class or instance
        shm_name: Shared memory segment name
        batch_mode: Whether to use batch reading
        poll_interval: Polling interval in seconds
        **strategy_params: Additional parameters to pass to the strategy
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Resolve strategy class
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
        strategy_class = strategy
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Create Cerebro instance
    cerebro = bt.Cerebro()
    cerebro.broker.set_coc(True)
    
    # Add multiple HotSpine data feeds
    from backtrader.feeds.hotspine_feed import HotSpineData
    
    for symbol_id in symbol_ids:
        data = HotSpineData(
            symbol_id=symbol_id,
            shm_name=shm_name,
            batch_mode=batch_mode,
            poll_interval=poll_interval
        )
        
        data._dataname = f"hotspine_{symbol_id}"
        data._name = f"HotSpine_{symbol_id}"
        
        cerebro.adddata(data, name=data._dataname)
    
    # Add strategy
    cerebro.addstrategy(
        strategy_class,
        backtest=False,
        live=True,
        **strategy_params
    )
    
    try:
        logger.info(f"Starting HotSpine live trading for symbols: {symbol_ids}")
        
        cerebro.run(
            live=True,
            runonce=False,
            exactbars=False,
            stdstats=False,
            preload=False,
            quicknotify=True
        )
        
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    except Exception as e:
        logger.error(f"Live trading failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def livetrade_binance(
=======
    
def livetrade_crypto_binance(
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
    - alert_channel (str): Define where to send alerts via alert engine to corresponding channels.
    """

    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy

    from backtrader.stores import binance_store

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

def livetrade_crypto_binance_ML(
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = "",
<<<<<<< HEAD
=======
    # memory_saving: int = -1  # Add this parameter with default value
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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
<<<<<<< HEAD
    """
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
        strategy_class = strategy
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
=======
    - memory_saving (int): Memory saving level:
        0: No memory saving (default behavior)
        1: Maximum memory savings, disables plotting
       -1: Save memory for subindicators only
       -2: Save memory for non-strategy attributes
    """
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    
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

<<<<<<< HEAD
def livetrade_mexc(
=======
def livetrade_crypto_mexc(
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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

<<<<<<< HEAD
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
=======
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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

<<<<<<< HEAD
def livetrade_bitget(
=======

# 
def livetrade_crypto_bitget(
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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

<<<<<<< HEAD
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
        strategy_class = strategy

    from backtrader.stores.bitget_store import BitgetStore
    from backtrader.feeds.bitget_feed import BitgetData

    cerebro = bt.Cerebro(quicknotify=True)
    store = BitgetStore(symbol=coin, product="spot", debug=True)
=======
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

    # Set the timezone to UTC+2
    tz = pytz.timezone('Europe/Berlin')
    current_time = datetime.now(tz)

    # Add extra buffer time to ensure smooth transition
    buffer_minutes = 2
    from_date = current_time - timedelta(hours=start_hours_ago, minutes=buffer_minutes)

    print(f"Current time (UTC+2): {current_time}")
    print(f"Fetching historical data from (UTC+2): {from_date}")
    
<<<<<<< HEAD
    data = BitgetData(store=store, start_date=from_date)
=======
    data = store.getdata(start_date=from_date)
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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

<<<<<<< HEAD
def livetrade_tv(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str,
    tv_symbol: str,
    amount: float = 0.05,
    lookback_minutes: int = 1,
) -> None:
    """
    Live trading using TradingView as data source.
    """
    
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
        strategy_class = strategy
    
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


=======
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
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
    
<<<<<<< HEAD
    if isinstance(strategy, str):
        strategy_class = strategy.lower()
    elif callable(strategy):
        strategy_class = strategy
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
=======
    if strategy_class is None:
        raise ValueError(f"Strategy '{strategy}' not found in STRATEGY_MAPPING.")
    else:
        strategy_class = strategy
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    
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
