"""
jrr_websocket/manager_sync.py

Synchronous WebSocket manager with threading and fast_mssql backend.
No asyncio - pure threading model for compatibility with custom C++ driver.
"""

import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json
import queue
import time

from backtrader.Jackrabbit.connectors.base import BaseWebSocketConnector, Message
from .storage_mssql import MarketDataStorage, MSSQLConfig


class WebSocketManagerSync:
    """
    Synchronous coordinator for multiple websocket connections.

    Uses threading model instead of asyncio for compatibility with
    synchronous fast_mssql driver.

    Responsibilities:
    - Manage connectors for multiple exchanges
    - Handle subscription requests
    - Distribute data to multiple subscribers
    - Persist data to SQL Server (synchronous)
    - Provide health monitoring and statistics
    """

    def __init__(
        self,
        storage_config: MSSQLConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.storage_config = storage_config
        self.logger = logger or logging.getLogger(__name__)

        # Connection management
        self.connectors: Dict[str, BaseWebSocketConnector] = {}
        self.connector_threads: Dict[str, threading.Thread] = {}

        # Data persistence
        self.storage: Optional[MarketDataStorage] = None

        # Message queue for thread-safe processing
        self.message_queue = queue.Queue(maxsize=10000)
        self.processing_thread = None
        self.running = False

        # Subscriber callbacks
        self.subscribers: Dict[str, List[Callable[[Message], Any]]] = {
            'trade': [],
            'orderbook': [],
            'ohlcv': [],
            'funding': [],
            'mark_price': [],
        }

        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_errors': 0,
            'data_stored': 0,
            'connectors_active': 0,
            'queue_size': 0,
        }

        self._stats_lock = threading.Lock()

    def initialize(self) -> bool:
        """Initialize storage and schema"""
        try:
            self.storage = MarketDataStorage(self.storage_config, self.logger)
            self.storage.connect()
            self.logger.info("Manager initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def shutdown(self) -> None:
        """Graceful shutdown of all connections"""
        self.logger.info("Shutting down manager...")

        self.running = False

        # Stop all connector threads
        for name, thread in self.connector_threads.items():
            if thread.is_alive():
                self.logger.info(f"Stopping {name} thread...")
                # Signal connector to stop (implementation dependent)

        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        # Close storage
        if self.storage:
            self.storage.disconnect()

        self.logger.info("Manager shutdown complete")

    def register_connector(self, connector: BaseWebSocketConnector) -> None:
        """Register a new exchange connector"""
        self.connectors[connector.exchange] = connector

        # Add internal queue subscriber
        def queue_handler(message: Message):
            try:
                self.message_queue.put_nowait(message)
            except queue.Full:
                self.logger.warning(f"Message queue full, dropping message from {message.exchange}")

        connector.add_subscriber(queue_handler)

        self.logger.info(f"Registered connector: {connector.exchange}")

    def subscribe_exchange(
        self,
        exchange: str,
        channels: List[str]
    ) -> bool:
        """
        Subscribe to channels on an exchange.

        Args:
            exchange: Exchange name (e.g., 'binance', 'okx')
            channels: List of channel names

        Returns:
            True if subscription successful
        """
        if exchange not in self.connectors:
            self.logger.error(f"Exchange not registered: {exchange}")
            return False

        connector = self.connectors[exchange]

        try:
            # This needs to be adapted for sync operation
            # In async version, this would be: await connector.subscribe(channels)
            # For now, store subscriptions and apply when thread starts
            if not hasattr(connector, '_pending_subscriptions'):
                connector._pending_subscriptions = []
            connector._pending_subscriptions.extend(channels)

            self.logger.info(f"Queued subscription for {exchange}: {len(channels)} channels")
            return True
        except Exception as e:
            self.logger.error(f"Subscription failed: {e}")
            return False

    def _process_messages(self) -> None:
        """Background thread for processing messages from queue"""
        self.logger.info("Message processing thread started")

        while self.running:
            try:
                # Get message from queue with timeout
                message = self.message_queue.get(timeout=1.0)

                # Store to database
                self._store_message(message)

                # Dispatch to subscribers
                self._dispatch_message(message)

                with self._stats_lock:
                    self.stats['messages_processed'] += 1
                    self.stats['queue_size'] = self.message_queue.qsize()

                self.message_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
                with self._stats_lock:
                    self.stats['messages_errors'] += 1

        self.logger.info("Message processing thread stopped")

    def _store_message(self, message: Message) -> None:
        """Store message to database (synchronous)"""
        try:
            data = {
                'exchange': message.exchange,
                'symbol': message.symbol,
                'timestamp': message.timestamp,
                **message.data
            }

            if message.type == 'ohlcv':
                self.storage.store_ohlcv(data)
            elif message.type == 'trade':
                self.storage.store_trade(data)
            elif message.type == 'orderbook':
                self.storage.store_orderbook(data)

            with self._stats_lock:
                self.stats['data_stored'] += 1

        except Exception as e:
            self.logger.error(f"Storage error: {e}")

    def _dispatch_message(self, message: Message) -> None:
        """Dispatch message to subscribers"""
        if message.type in self.subscribers:
            for callback in self.subscribers[message.type]:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Subscriber callback error: {e}")

    def add_subscriber(
        self,
        message_type: str,
        callback: Callable[[Message], Any]
    ) -> None:
        """
        Register a callback for a message type.

        Args:
            message_type: 'trade', 'orderbook', 'ohlcv', 'funding', 'mark_price'
            callback: Synchronous function(message: Message)
        """
        if message_type not in self.subscribers:
            self.logger.warning(f"Unknown message type: {message_type}")
            return

        self.subscribers[message_type].append(callback)
        self.logger.debug(f"Added subscriber for {message_type}")

    def remove_subscriber(
        self,
        message_type: str,
        callback: Callable[[Message], Any]
    ) -> None:
        """Unregister a callback"""
        if message_type in self.subscribers and callback in self.subscribers[message_type]:
            self.subscribers[message_type].remove(callback)

    def start(self) -> None:
        """
        Start manager and all listeners.
        Launches background threads for message processing.
        """
        if self.running:
            self.logger.warning("Manager already running")
            return

        self.running = True

        # Start message processing thread
        self.processing_thread = threading.Thread(
            target=self._process_messages,
            name="MessageProcessor",
            daemon=True
        )
        self.processing_thread.start()

        self.logger.info("Manager started")

    def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        if not self.storage:
            return None
        return self.storage.get_latest_price(exchange, symbol)

    def get_ohlcv_history(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get recent OHLCV data"""
        if not self.storage:
            return []

        from datetime import datetime, timedelta
        end = datetime.utcnow()

        # Estimate start time based on timeframe and limit
        minutes_per_bar = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        minutes = minutes_per_bar.get(timeframe, 1) * limit
        start = end - timedelta(minutes=minutes)

        return self.storage.get_ohlcv(
            exchange, symbol, timeframe, start, end, limit=limit
        )

    def get_connector_stats(self, exchange: str) -> Dict[str, Any]:
        """Get statistics for a connector"""
        if exchange not in self.connectors:
            return {}
        return self.connectors[exchange].get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall manager statistics"""
        with self._stats_lock:
            stats_copy = self.stats.copy()

        connector_stats = {
            exchange: self.get_connector_stats(exchange)
            for exchange in self.connectors.keys()
        }

        # Get database stats
        db_stats = {}
        if self.storage:
            db_stats = self.storage.get_stats()

        return {
            'manager': stats_copy,
            'connectors': connector_stats,
            'database': db_stats,
            'timestamp': datetime.utcnow().isoformat(),
        }

    def print_stats(self, interval: int = 60) -> None:
        """
        Print statistics periodically.
        Should be run in a separate thread.
        """
        while self.running:
            time.sleep(interval)
            stats = self.get_stats()
            self.logger.info(f"Stats: {json.dumps(stats, indent=2, default=str)}")


class ManagerBuilder:
    """Builder for easy manager setup with SQL Server"""

    def __init__(self, storage_config: Optional[MSSQLConfig] = None):
        self.storage_config = storage_config or MSSQLConfig()
        self.manager = WebSocketManagerSync(self.storage_config)
        self.connectors_to_add = []

    def add_binance(self, config: Optional[Dict] = None):
        """Add Binance connector"""
        # Note: This assumes you have a synchronous version of connectors
        # or adapt async connectors to work with threading
        from .connectors.binance import BinanceWebSocketConnector

        connector = BinanceWebSocketConnector(
            'binance',
            config or {},
            self.manager.logger
        )
        self.connectors_to_add.append(connector)
        return self

    def add_okx(self, config: Optional[Dict] = None):
        """Add OKX connector"""
        from .connectors.okx import OKXWebSocketConnector

        connector = OKXWebSocketConnector(
            'okx',
            config or {},
            self.manager.logger
        )
        self.connectors_to_add.append(connector)
        return self

    def build(self) -> WebSocketManagerSync:
        """Build and initialize manager"""
        if not self.manager.initialize():
            raise RuntimeError("Failed to initialize manager")

        for connector in self.connectors_to_add:
            self.manager.register_connector(connector)

        return self.manager
