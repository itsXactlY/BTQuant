"""
Market Data Collector - Python equivalent of market_data_collector.cpp

This is the main orchestrator that coordinates all components for market data collection.
"""

import threading
import time
import signal
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime

from .config_loader import load_config, Config
from .exchange_connection_manager import ExchangeConnectionManager
from .market_data_processor import MarketDataProcessor
from .candle_aggregator import CandleAggregator
from .hotspine_writer import HotSpineWriter
from .utilities import get_current_timestamp, now_micros


class MarketDataCollector:
    """
    Main market data collector that orchestrates all components.

    This class manages the lifecycle of market data collection, coordinates
    between exchanges, processors, and storage systems.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the market data collector.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config.json"
        self.config: Optional[Config] = None

        # Core components
        self.exchange_manager: Optional[ExchangeConnectionManager] = None
        self.processor: Optional[MarketDataProcessor] = None
        self.candle_agg: Optional[CandleAggregator] = None
        self.hotspine_writer: Optional[HotSpineWriter] = None

        # Threading
        self.running = False
        self.main_thread: Optional[threading.Thread] = None
        self.stats_thread: Optional[threading.Thread] = None

        # Shutdown handling
        self.shutdown_event = threading.Event()

        print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Initialized")

    def load_configuration(self) -> bool:
        """Load configuration from file"""
        try:
            self.config = load_config(self.config_path)
            print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Configuration loaded from {self.config_path}")
            return True
        except Exception as e:
            print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: Failed to load configuration: {e}")
            return False

    def initialize_components(self) -> bool:
        """Initialize all collector components"""
        if not self.config:
            print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: No configuration loaded")
            return False

        try:
            # Initialize HotSpine writer if enabled
            if self.config.enable_exclusive_hotspine or self.config.enable_hotspine:
                self.hotspine_writer = HotSpineWriter(
                    batch_size=self.config.hotspine_batch_size,
                    max_buffer_size=self.config.hotspine_max_buffer_size,
                    debug_config=self.config.debug
                )
                print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: HotSpine writer initialized")

            # Initialize candle aggregator
            if self.config.timeframes:
                self.candle_agg = CandleAggregator(self.config.timeframes)
                print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Candle aggregator initialized with timeframes: {self.config.timeframes}")

            # Initialize market data processor
            self.processor = MarketDataProcessor(
                candle_agg=self.candle_agg,
                hotspine_writer=self.hotspine_writer,
                enable_exclusive_hotspine=self.config.enable_exclusive_hotspine,
                debug_config=self.config.debug
            )

            # Set buffer limits
            self.processor.set_buffer_limits(
                self.config.max_trade_buffer_size,
                self.config.max_candle_buffer_size,
                self.config.max_orderbook_buffer_size
            )

            # Initialize exchange connection manager
            self.exchange_manager = ExchangeConnectionManager(
                exchanges=self.config.exchanges,
                processor=self.processor,
                debug_config=self.config.debug
            )

            print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: All components initialized successfully")
            return True

        except Exception as e:
            print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: Failed to initialize components: {e}")
            return False

    def start_collection(self) -> bool:
        """Start market data collection"""
        if not self.config or not self.processor or not self.exchange_manager:
            print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: Components not initialized")
            return False

        try:
            self.running = True

            # Start exchange connections
            if not self.exchange_manager.start_connections():
                print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: Failed to start exchange connections")
                return False

            # Start processor
            self.processor.start()

            # Start statistics thread
            self.stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
            self.stats_thread.start()

            print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Market data collection started")
            return True

        except Exception as e:
            print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: Failed to start collection: {e}")
            return False

    def stop_collection(self) -> None:
        """Stop market data collection"""
        print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Stopping market data collection...")

        self.running = False
        self.shutdown_event.set()

        # Stop exchange connections
        if self.exchange_manager:
            self.exchange_manager.stop_connections()

        # Stop processor
        if self.processor:
            self.processor.stop()

        # Wait for threads to finish
        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=5.0)

        print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Market data collection stopped")

    def _stats_loop(self) -> None:
        """Background thread for periodic statistics logging"""
        while self.running and not self.shutdown_event.is_set():
            try:
                time.sleep(self.config.stats_interval_seconds if self.config else 30)

                if self.processor:
                    self.processor.log_web_socket_data_flow_stats()

                    # Additional debugging if enabled
                    if self.config and self.config.debug.enabled:
                        if self.config.debug.websocket_debug:
                            self.processor.add_web_socket_debugging()

                        if self.config.debug.candle_debug:
                            self.processor.validate_candle_aggregation()

            except Exception as e:
                print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: Stats loop error: {e}")

    def run(self) -> int:
        """Main run loop"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Load configuration
            if not self.load_configuration():
                return 1

            # Initialize components
            if not self.initialize_components():
                return 1

            # Start collection
            if not self.start_collection():
                return 1

            # Main loop
            print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Running... Press Ctrl+C to stop")

            while self.running and not self.shutdown_event.is_set():
                time.sleep(1.0)

                # Periodic validation
                if self.config and self.config.debug.enabled and self.config.debug.websocket_debug:
                    self.processor.validate_web_socket_data_flow()

            return 0

        except KeyboardInterrupt:
            print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Received keyboard interrupt")
            return 0
        except Exception as e:
            print(f"[{get_current_timestamp()}][ERROR] MarketDataCollector: Fatal error: {e}")
            return 1
        finally:
            self.stop_collection()

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals"""
        print(f"[{get_current_timestamp()}][INFO] MarketDataCollector: Received signal {signum}, initiating shutdown...")
        self.running = False
        self.shutdown_event.set()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the collector"""
        status = {
            "running": self.running,
            "config_loaded": self.config is not None,
            "components_initialized": all([
                self.exchange_manager is not None,
                self.processor is not None
            ]),
            "exchanges_connected": 0,
            "processor_stats": {},
            "timestamp": get_current_timestamp()
        }

        if self.exchange_manager:
            status["exchanges_connected"] = self.exchange_manager.get_connected_count()

        if self.processor:
            status["processor_stats"] = self.processor.get_stats().__dict__

        return status

    def get_status_json(self) -> str:
        """Get status as JSON string"""
        import json
        return json.dumps(self.get_status(), indent=2)


def main() -> int:
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Market Data Collector")
    parser.add_argument("--config", "-c", help="Path to configuration file", default="config.json")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration and exit")

    args = parser.parse_args()

    collector = MarketDataCollector(args.config)

    if args.validate_only:
        # Just load and validate configuration
        if collector.load_configuration():
            print(f"[{get_current_timestamp()}][INFO] Configuration validation successful")
            return 0
        else:
            print(f"[{get_current_timestamp()}][ERROR] Configuration validation failed")
            return 1

    # Run the collector
    return collector.run()


if __name__ == "__main__":
    sys.exit(main())