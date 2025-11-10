"""
jrr_websocket/manager_mssql.py

WebSocket manager with SQL Server storage backend.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json

from backtrader.Jackrabbit.connectors.base import BaseWebSocketConnector, Message
from .storage_mssql import MarketDataStorage, MSSQLConfig


class WebSocketManager:
    """
    Central coordinator for multiple websocket connections with SQL Server backend.

    Responsibilities:
    - Manage connectors for multiple exchanges
    - Handle subscription requests
    - Distribute data to multiple subscribers
    - Persist data to SQL Server
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
        self.connector_tasks: Dict[str, asyncio.Task] = {}

        # Data persistence
        self.storage: Optional[MarketDataStorage] = None

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
        }

    async def initialize(self) -> bool:
        """Initialize storage and schema"""
        try:
            self.storage = MarketDataStorage(self.storage_config, self.logger)
            await self.storage.connect()
            self.logger.info("Manager initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Graceful shutdown of all connections"""
        self.logger.info("Shutting down manager...")

        # Cancel all tasks
        for task in self.connector_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Disconnect all connectors
        for connector in self.connectors.values():
            await connector.disconnect()

        # Close storage
        if self.storage:
            await self.storage.disconnect()

        self.logger.info("Manager shutdown complete")

    def register_connector(self, connector: BaseWebSocketConnector) -> None:
        """Register a new exchange connector"""
        self.connectors[connector.exchange] = connector

        # Add internal storage subscriber
        connector.add_subscriber(self._handle_message)

        self.logger.info(f"Registered connector: {connector.exchange}")

    async def subscribe_exchange(
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
            await connector.subscribe(channels)
            self.logger.info(f"Subscribed {exchange} to {len(channels)} channels")
            return True
        except Exception as e:
            self.logger.error(f"Subscription failed: {e}")
            return False

    async def start_listener(self, exchange: str) -> None:
        """
        Start listening for messages from an exchange.
        Runs in background task.
        """
        if exchange not in self.connectors:
            self.logger.error(f"Exchange not registered: {exchange}")
            return

        connector = self.connectors[exchange]

        task = asyncio.create_task(connector.listen())
        self.connector_tasks[exchange] = task

        self.logger.info(f"Listener started for {exchange}")

        try:
            await task
        except asyncio.CancelledError:
            self.logger.info(f"Listener cancelled for {exchange}")

    async def start_all_listeners(self) -> None:
        """Start listeners for all registered connectors"""
        tasks = [
            self.start_listener(exchange)
            for exchange in self.connectors.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    def add_subscriber(
        self,
        message_type: str,
        callback: Callable[[Message], Any]
    ) -> None:
        """
        Register a callback for a message type.

        Args:
            message_type: 'trade', 'orderbook', 'ohlcv', 'funding', 'mark_price'
            callback: Async or sync function(message: Message)
        """
        if message_type not in self.subscribers:
            self.logger.warning(f"Unknown message type: {message_type}")
            return

        self.subscribers[message_type].append(callback)
        self.logger.debug(f"Added subscriber for {message_type} (total: {len(self.subscribers[message_type])})")

    def remove_subscriber(
        self,
        message_type: str,
        callback: Callable[[Message], Any]
    ) -> None:
        """Unregister a callback"""
        if message_type in self.subscribers and callback in self.subscribers[message_type]:
            self.subscribers[message_type].remove(callback)

    async def _handle_message(self, message: Message) -> None:
        """
        Internal message handler - persists to storage and dispatches to subscribers.
        """
        try:
            # Store message
            if message.type == 'ohlcv':
                await self.storage.store_ohlcv({
                    'exchange': message.exchange,
                    'symbol': message.symbol,
                    'timestamp': message.timestamp,
                    **message.data
                })
            elif message.type == 'trade':
                await self.storage.store_trade({
                    'exchange': message.exchange,
                    'symbol': message.symbol,
                    'timestamp': message.timestamp,
                    **message.data
                })
            elif message.type == 'orderbook':
                await self.storage.store_orderbook({
                    'exchange': message.exchange,
                    'symbol': message.symbol,
                    'timestamp': message.timestamp,
                    **message.data
                })

            self.stats['data_stored'] += 1

            # Dispatch to subscribers
            if message.type in self.subscribers:
                for callback in self.subscribers[message.type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        self.logger.error(f"Subscriber callback error: {e}")
                        self.stats['messages_errors'] += 1

            self.stats['messages_processed'] += 1

        except Exception as e:
            self.logger.error(f"Message handling error: {e}")
            self.stats['messages_errors'] += 1

    async def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        if not self.storage:
            return None
        return await self.storage.get_latest_price(exchange, symbol)

    async def get_ohlcv_history(
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

        return await self.storage.get_ohlcv(
            exchange, symbol, timeframe, start, end, limit=limit
        )

    def get_connector_stats(self, exchange: str) -> Dict[str, Any]:
        """Get statistics for a connector"""
        if exchange not in self.connectors:
            return {}
        return self.connectors[exchange].get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall manager statistics"""
        connector_stats = {
            exchange: self.get_connector_stats(exchange)
            for exchange in self.connectors.keys()
        }

        return {
            'manager': self.stats,
            'connectors': connector_stats,
            'timestamp': datetime.utcnow().isoformat(),
        }

    async def print_stats(self, interval: int = 60) -> None:
        """Periodically print statistics"""
        while True:
            await asyncio.sleep(interval)
            stats = self.get_stats()

            # Also get database stats
            if self.storage:
                db_stats = await self.storage.get_stats()
                stats['database'] = db_stats

            self.logger.info(f"Stats: {json.dumps(stats, indent=2, default=str)}")


# Example usage helper
class ManagerBuilder:
    """Builder for easy manager setup with SQL Server"""

    def __init__(self, storage_config: Optional[MSSQLConfig] = None):
        self.storage_config = storage_config or MSSQLConfig()
        self.manager = WebSocketManager(self.storage_config)
        self.connectors_to_add = []

    def add_binance(self, config: Optional[Dict] = None):
        """Add Binance connector"""
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

    async def build(self) -> WebSocketManager:
        """Build and initialize manager"""
        if not await self.manager.initialize():
            raise RuntimeError("Failed to initialize manager")

        for connector in self.connectors_to_add:
            self.manager.register_connector(connector)

        return self.manager
