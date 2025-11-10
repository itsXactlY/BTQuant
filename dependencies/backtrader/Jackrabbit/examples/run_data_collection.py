"""
examples/run_data_collection.py

Production example: Real-time market data collection from Binance and OKX
"""

import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_collection.log'),
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """
    Main data collection loop.

    Collects OHLCV, trades, and orderbook data from multiple exchanges.
    """
    from backtrader.Jackrabbit.manager import ManagerBuilder
    from backtrader.Jackrabbit.storages import StorageConfig

    # Configure storage
    storage_config = StorageConfig(
        host='localhost',
        port=5432,
        database='jrr_marketdata',
        user='postgres',
        password='',  # Set via environment variable
    )

    # Build manager with Binance and OKX
    manager = await (
        ManagerBuilder(storage_config)
        .add_binance({
            'max_reconnect_attempts': 10,
            'initial_backoff': 1,
            'heartbeat_interval': 30,
        })
        .add_okx({
            'max_reconnect_attempts': 10,
            'initial_backoff': 1,
            'heartbeat_interval': 30,
        })
        .build()
    )

    # Example subscriber: Print trades
    async def on_trade(message):
        """Handle trade message"""
        data = message.data
        logger.info(
            f"TRADE: {message.exchange} {message.symbol} "
            f"Price={data['price']} Qty={data['quantity']} "
            f"Side={data.get('side', 'unknown')}"
        )

    manager.add_subscriber('trade', on_trade)

    # Example subscriber: Print candles
    async def on_ohlcv(message):
        """Handle OHLCV message"""
        data = message.data
        logger.info(
            f"CANDLE: {message.exchange} {message.symbol} {data['timeframe']} "
            f"O={data['open']} H={data['high']} L={data['low']} C={data['close']} V={data['volume']}"
        )

    manager.add_subscriber('ohlcv', on_ohlcv)

    # Subscribe to Binance channels
    await manager.subscribe_exchange('binance', [
        'btcusdt@trade',         # Real-time trades
        'btcusdt@depth5@100ms',  # L2 depth snapshots
        'btcusdt@kline_1m',      # 1-minute candles
        'btcusdt@kline_5m',      # 5-minute candles
        'ethusdt@trade',
        'ethusdt@kline_1m',
    ])

    # Subscribe to OKX channels
    await manager.subscribe_exchange('okx', [
        'trades:BTC-USDT',       # Spot trades
        'candle1m:BTC-USDT',     # 1m candles
        'candle5m:BTC-USDT',     # 5m candles
        'trades:ETH-USDT',
        'candle1m:ETH-USDT',
    ])

    # Start statistics printer
    stats_task = asyncio.create_task(manager.print_stats(interval=60))

    # Start all listeners
    listeners_task = asyncio.create_task(manager.start_all_listeners())

    try:
        logger.info("Data collection started")
        await listeners_task
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        stats_task.cancel()
        await manager.shutdown()
        logger.info("Data collection stopped")


if __name__ == '__main__':
    asyncio.run(main())
