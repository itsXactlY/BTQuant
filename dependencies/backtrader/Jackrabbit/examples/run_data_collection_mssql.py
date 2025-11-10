"""
examples/run_data_collection_mssql.py

Production example: Real-time market data collection with SQL Server backend.
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
    Main data collection loop with SQL Server backend.

    Collects OHLCV, trades, and orderbook data from Binance and OKX.
    """
    from backtrader.Jackrabbit.manager_mssq import ManagerBuilder, MSSQLConfig

    # SQL Server configuration
    storage_config = MSSQLConfig(
        server='localhost',
        database='BinanceData',
        username='SA',
        password='q?}33YIToo:H%xue$Kr*',
        driver='{ODBC Driver 18 for SQL Server}',
        trust_server_certificate=True,
        min_size=5,
        max_size=20,
    )

    logger.info(f"Connecting to SQL Server: {storage_config.server}/{storage_config.database}")

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
            f"Price={data['price']:.2f} Qty={data['quantity']:.4f} "
            f"Side={data.get('side', 'unknown')}"
        )

    manager.add_subscriber('trade', on_trade)

    # Example subscriber: Print candles
    async def on_ohlcv(message):
        """Handle OHLCV message"""
        data = message.data
        if data.get('is_closed', True):  # Only log closed candles
            logger.info(
                f"CANDLE: {message.exchange} {message.symbol} {data['timeframe']} "
                f"O={data['open']:.2f} H={data['high']:.2f} "
                f"L={data['low']:.2f} C={data['close']:.2f} V={data['volume']:.2f}"
            )

    manager.add_subscriber('ohlcv', on_ohlcv)

    # Subscribe to Binance channels
    logger.info("Subscribing to Binance channels...")
    await manager.subscribe_exchange('binance', [
        'btcusdt@trade',         # Real-time trades
        'btcusdt@depth5@100ms',  # L2 depth snapshots
        'btcusdt@kline_1m',      # 1-minute candles
        'btcusdt@kline_5m',      # 5-minute candles
        'btcusdt@kline_15m',     # 15-minute candles
        'ethusdt@trade',
        'ethusdt@kline_1m',
        'ethusdt@kline_5m',
    ])

    # Subscribe to OKX channels
    logger.info("Subscribing to OKX channels...")
    await manager.subscribe_exchange('okx', [
        'trades:BTC-USDT',       # Spot trades
        'candle1m:BTC-USDT',     # 1m candles
        'candle5m:BTC-USDT',     # 5m candles
        'candle15m:BTC-USDT',    # 15m candles
        'trades:ETH-USDT',
        'candle1m:ETH-USDT',
        'candle5m:ETH-USDT',
    ])

    # Start statistics printer (every 60 seconds)
    stats_task = asyncio.create_task(manager.print_stats(interval=60))

    # Start all listeners
    listeners_task = asyncio.create_task(manager.start_all_listeners())

    try:
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info("║  Data collection started - Press Ctrl+C to stop             ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")
        await listeners_task
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        stats_task.cancel()
        await manager.shutdown()
        logger.info("Data collection stopped")


if __name__ == '__main__':
    asyncio.run(main())
