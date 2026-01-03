#!/usr/bin/env python3
"""
Live Trading Example with HotSpine and SMA Cross Strategy

This example demonstrates how to use HotSpine for ultra-low latency live trading
with the SMA_Cross_MESAdaptive_Prime strategy using CCAPI/JRR for order execution.

Requirements:
- HotSpine shared memory segment must be running
- HotSpine reader library must be available
- Backtrader with HotSpine integration
- CCAPI for order execution

Usage:
    python Live_Trading_HotSpine_SMA.py

The script will:
1. Connect to HotSpine shared memory for ultra-low latency data
2. Initialize the SMA Cross strategy
3. Use CCAPI/JRR for order execution (no CCXT)
4. Display trading statistics and performance

HotSpine provides sub-microsecond latency for live trading while maintaining
architectural separation from storage operations. CCAPI handles all order execution.
"""

import sys
import os
import time
import signal
import logging
from datetime import datetime

# Add the dependencies to Python path
sys.path.insert(0, 'dependencies')
sys.path.insert(0, 'dependencies/backtrader')

import backtrader as bt
from backtrader.strategies.SMA_Cross_MESAdaptive_Prime import SMA_Cross_MESAdaptivePrime
from backtrader.feeds.hotspine_feed import HotSpineData, create_hotspine_data_feed
from backtrader.brokers.jrrbroker import JrrOrderBase
from backtrader import analyzers

# Import HotSpine configuration management
from backtrader.hotspine.config import HotSpineConfig, configure_logging

# Configure logging using HotSpine configuration
configure_logging()
logger = logging.getLogger(__name__)


class HotSpineSMALiveStrategy(SMA_Cross_MESAdaptivePrime):
    """
    Enhanced SMA Cross strategy for HotSpine live trading

    This strategy extends the base SMA_Cross_MESAdaptive_Prime strategy
    with HotSpine-specific optimizations and monitoring.
    """

    params = (
        ('fast', 13),
        ('slow', 37),
        ('dca_deviation', 1.5),
        ('take_profit', 2.0),
        ('percent_sizer', 0.01),
        ('debug', True),  # Enable debug output for live trading
        ('symbol_id', 123),  # Symbol ID for HotSpine filtering
    )

    def __init__(self):
        super().__init__()

        # HotSpine-specific initialization
        self.hotspine_trade_count = 0
        self.hotspine_start_time = time.time()
        self.last_status_time = self.hotspine_start_time

        logger.info("HotSpine SMA Strategy initialized")
        logger.info(f"Symbol ID: {self.p.symbol_id}")
        logger.info(f"Fast MA: {self.p.fast}, Slow MA: {self.p.slow}")

    def next(self):
        """Main strategy logic with HotSpine monitoring"""
        # Call parent strategy logic
        super().next()

        # HotSpine-specific monitoring
        self.hotspine_trade_count += 1
        current_time = time.time()

        # Log status every 10 seconds
        if current_time - self.last_status_time >= 10.0:
            elapsed = current_time - self.hotspine_start_time
            rate = self.hotspine_trade_count / elapsed if elapsed > 0 else 0

            logger.info(f"HotSpine Status: {self.hotspine_trade_count} trades processed "
                       f"({rate:.1f} trades/sec) | "
                       f"Position: {self.position.size if self.position else 0} | "
                       f"Price: {self.data.close[0]:.4f}")

            self.last_status_time = current_time

    def notify_trade(self, trade):
        """Enhanced trade notification with HotSpine details"""
        super().notify_trade(trade)

        if trade.isclosed:
            logger.info(f"HotSpine Trade Closed: PnL={trade.pnl:.2f}, "
                       f"PnLComm={trade.pnlcomm:.2f}")


def setup_broker(exchange_config):
    """
    Setup CCAPI/JRR broker for live trading

    Args:
        exchange_config: Dictionary with exchange configuration

    Returns:
        Configured JrrOrderBase instance
    """
    # Initialize CCAPI/JRR broker for order execution
    broker = JrrOrderBase()
    
    logger.info(f"CCAPI/JRR broker configured for {exchange_config.get('exchange', 'mimic')}")
    return broker


def run_hotspine_live_trading(strategy_class=SMA_Cross_MESAdaptivePrime,
                               symbol_id=123,
                               shm_name="/btquant_hotspine",
                               batch_mode=False,
                               exchange_config=None):
    """
    Run live trading with HotSpine data feed

    Args:
        strategy_class: Trading strategy class
        symbol_id: Symbol ID for HotSpine filtering
        shm_name: HotSpine shared memory name
        batch_mode: Whether to use batch reading mode
        exchange_config: Exchange configuration for broker
    """

    print("=" * 80)
    print("HOTSPINE LIVE TRADING EXAMPLE")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Strategy: {strategy_class.__name__}")
    print(f"  Symbol ID: {symbol_id}")
    print(f"  Shared Memory: {shm_name}")
    print(f"  Batch Mode: {batch_mode}")
    print()

    # Create Cerebro engine
    cerebro = bt.Cerebro()

    # Add analyzers for performance tracking
    cerebro.addanalyzer(analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(analyzers.Returns, _name='returns')

    # Setup CCAPI/JRR broker for order execution
    # HotSpine handles data, CCAPI handles orders - complete separation
    broker = setup_broker(exchange_config)
    cerebro.setbroker(broker)
    logger.info("CCAPI/JRR broker configured for order execution")

    # Create HotSpine configuration
    hotspine_config = HotSpineConfig()
    hotspine_config.shm_name = shm_name
    hotspine_config.batch_mode = batch_mode
    hotspine_config.poll_interval = 0.0001  # 100 microseconds polling
    hotspine_config.enable_monitoring = True
    
    # Create HotSpine data feed with enhanced configuration
    try:
        logger.info(f"Connecting to HotSpine shared memory: {shm_name}")
        logger.info(f"Using configuration: batch_mode={batch_mode}, monitoring_enabled=True")
        
        data = create_hotspine_data_feed(
            symbol_id=symbol_id,
            shm_name=shm_name,
            batch_mode=batch_mode,
            poll_interval=0.0001  # 100 microseconds polling
        )

        cerebro.adddata(data)
        logger.info("HotSpine data feed added successfully")
        
        # Log initial health status
        if hasattr(data, 'get_health_status'):
            health_status = data.get_health_status()
            logger.info(f"Initial health status: {health_status}")

    except Exception as e:
        logger.error(f"Failed to create HotSpine data feed: {e}")
        print("\n❌ HotSpine Connection Failed!")
        print("Make sure:")
        print("  1. HotSpine writer is running")
        print("  2. Shared memory segment exists")
        print("  3. HotSpine library is properly installed")
        print(f"\nError: {e}")
        return

    # Add strategy
    cerebro.addstrategy(strategy_class, symbol_id=symbol_id)

    # Set initial cash
    # cerebro.broker.setcash(1000.0)
    # logger.info("Initial cash: $1000.00")

    print("\n🚀 Starting HotSpine Live Trading...")
    print("Press Ctrl+C to stop")
    print()

    # Global variables for signal handling
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        print("\n⏹️  Shutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Run live trading
        start_time = time.time()
        results = cerebro.run(live=True, runonce=False)

        # Calculate runtime
        runtime = time.time() - start_time

        print("\n" + "=" * 80)
        print("TRADING SESSION COMPLETED")
        print("=" * 80)

        # Display results
        if results:
            strat = results[0]

            print(f"\n📊 Performance Summary:")
            print(f"Runtime: {runtime:.1f} seconds")
            print(f"Final Cash: ${cerebro.broker.getcash():.2f}")
            print(f"Total Return: ${cerebro.broker.getcash() - 1000:.2f}")

            # Display analyzer results if available
            if hasattr(strat, 'analyzers'):
                try:
                    sharpe = strat.analyzers.sharpe.get_analysis()
                    drawdown = strat.analyzers.drawdown.get_analysis()
                    returns = strat.analyzers.returns.get_analysis()

                    print(f"\nSharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
                    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")
                    print(f"Total Return: {returns.get('rtot', 'N/A'):.2f}%")

                except Exception as e:
                    logger.warning(f"Could not retrieve analyzer results: {e}")

            # Display HotSpine metrics if available
            if hasattr(data, 'get_feed_metrics'):
                try:
                    feed_metrics = data.get_feed_metrics()
                    print(f"\n📈 HotSpine Feed Metrics:")
                    print(f"Trades Processed: {feed_metrics.get('trades_processed', 0)}")
                    print(f"Trades/Sec: {feed_metrics.get('trades_per_second', 0):.1f}")
                    print(f"Processing Errors: {feed_metrics.get('processing_errors', 0)}")
                    
                    if 'reader_metrics' in feed_metrics:
                        reader_metrics = feed_metrics['reader_metrics']
                        print(f"Reader Healthy: {reader_metrics.get('healthy', False)}")
                        print(f"Reader Trades: {reader_metrics.get('trades_read', 0)}")
                        print(f"Avg Read Latency: {reader_metrics.get('avg_read_latency', 0):.6f}s")
                        
                except Exception as e:
                    logger.warning(f"Could not retrieve HotSpine metrics: {e}")

        print("\n✅ HotSpine live trading session completed successfully!")
        
        # Log final health status
        if hasattr(data, 'get_health_status'):
            try:
                final_health = data.get_health_status()
                logger.info(f"Final health status: {final_health}")
            except Exception as e:
                logger.warning(f"Could not get final health status: {e}")

    except KeyboardInterrupt:
        print("\n⏹️  Live trading stopped by user")
    except Exception as e:
        logger.error(f"Live trading error: {e}")
        print(f"\n❌ Live trading failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function with configuration options"""

    # CCAPI/JRR configuration for order execution
    # HotSpine handles all data, CCAPI handles all orders
    exchange_config = {
        'exchange': 'mimic',  # Using JackRabbitRelay for order execution
        'account': '',       # Your JackRabbit account
        'asset': 'BTC/USDT',  # Trading pair
        'coin': 'BTC',       # Base currency
        'collateral': 'USDT' # Quote currency
    }

    logger.info("CCAPI/JRR configuration loaded for order execution")

    # HotSpine configuration
    symbol_id = 123  # BTC/USDT typically
    shm_name = "/btquant_hotspine"
    batch_mode = False  # Use single trade mode for lowest latency

    # Run live trading
    run_hotspine_live_trading(
        strategy_class=HotSpineSMALiveStrategy,
        symbol_id=symbol_id,
        shm_name=shm_name,
        batch_mode=batch_mode,
        exchange_config=exchange_config
    )


if __name__ == "__main__":
    main()