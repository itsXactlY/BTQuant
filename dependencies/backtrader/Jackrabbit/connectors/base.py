"""
jrr_websocket/connectors/base.py

Base WebSocket connector class with reconnection, heartbeat, and error handling.
"""

from abc import ABC, abstractmethod
import asyncio
import websockets
import logging
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class Message:
    """Normalized message format"""
    type: str  # 'trade', 'orderbook', 'ohlcv'
    exchange: str
    symbol: str
    timestamp: int
    data: Dict[str, Any]


class BaseWebSocketConnector(ABC):
    """
    Abstract base class for exchange-specific websocket implementations.

    Handles:
    - Connection lifecycle management
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring
    - Message parsing and normalization
    - Subscriber callback dispatch
    """

    def __init__(
        self,
        exchange: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        self.exchange = exchange
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.ws: Optional[websockets.WebSocketClientProtocol] = None

        # Configuration
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self.initial_backoff = config.get('initial_backoff', 1)
        self.max_backoff = config.get('max_backoff', 60)
        self.heartbeat_interval = config.get('heartbeat_interval', 30)

        # Subscriptions and callbacks
        self.subscriptions: List[str] = []
        self.callbacks: List[Callable[[Message], Any]] = []

        # Metrics
        self.messages_received = 0
        self.messages_failed = 0
        self.last_heartbeat = None

    @abstractmethod
    async def get_ws_url(self) -> str:
        """Return websocket URL for this exchange"""
        pass

    @abstractmethod
    async def subscribe(self, channels: List[str]) -> None:
        """Send subscription message for channels"""
        pass

    @abstractmethod
    def parse_message(self, raw_message: str) -> Optional[Message]:
        """Parse and normalize exchange-specific message format"""
        pass

    @abstractmethod
    async def send_heartbeat(self) -> None:
        """Send exchange-specific heartbeat message"""
        pass

    async def connect(self) -> bool:
        """
        Establish websocket connection.

        Returns:
            True if connection successful, False otherwise
        """
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return self.state == ConnectionState.CONNECTED

        self.state = ConnectionState.CONNECTING

        try:
            url = await self.get_ws_url()
            self.logger.info(f"[{self.exchange}] Connecting to {url}")

            self.ws = await asyncio.wait_for(
                websockets.connect(url, ping_interval=None),
                timeout=10
            )

            self.state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0
            self.logger.info(f"[{self.exchange}] Connected successfully")

            return True

        except asyncio.TimeoutError:
            self.logger.error(f"[{self.exchange}] Connection timeout")
            self.state = ConnectionState.FAILED
            return False

        except Exception as e:
            self.logger.error(f"[{self.exchange}] Connection failed: {e}")
            self.state = ConnectionState.FAILED
            return False

    async def disconnect(self) -> None:
        """Gracefully close websocket connection"""
        if self.ws:
            await self.ws.close()
        self.state = ConnectionState.DISCONNECTED
        self.logger.info(f"[{self.exchange}] Disconnected")

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection successful, False if max attempts exceeded
        """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(
                f"[{self.exchange}] Max reconnection attempts ({self.max_reconnect_attempts}) exceeded"
            )
            return False

        backoff = min(
            self.initial_backoff * (2 ** self.reconnect_attempts),
            self.max_backoff
        )

        self.state = ConnectionState.RECONNECTING
        self.reconnect_attempts += 1

        self.logger.warning(
            f"[{self.exchange}] Reconnecting in {backoff}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})"
        )

        await asyncio.sleep(backoff)

        if await self.connect():
            # Re-subscribe to channels after reconnect
            if self.subscriptions:
                try:
                    await self.subscribe(self.subscriptions)
                    self.logger.info(f"[{self.exchange}] Re-subscribed to {len(self.subscriptions)} channels")
                except Exception as e:
                    self.logger.error(f"[{self.exchange}] Re-subscription failed: {e}")
                    return False
            return True

        return False

    def add_subscriber(self, callback: Callable[[Message], Any]) -> None:
        """Register a callback for message dispatch"""
        self.callbacks.append(callback)
        self.logger.debug(f"[{self.exchange}] Subscriber added (total: {len(self.callbacks)})")

    def remove_subscriber(self, callback: Callable[[Message], Any]) -> None:
        """Unregister a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self.logger.debug(f"[{self.exchange}] Subscriber removed (total: {len(self.callbacks)})")

    async def dispatch(self, message: Message) -> None:
        """Dispatch parsed message to all subscribers"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self.logger.error(f"[{self.exchange}] Callback error: {e}")
                self.messages_failed += 1

    async def _heartbeat_loop(self) -> None:
        """Monitor connection health and send heartbeats"""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                if self.state == ConnectionState.CONNECTED:
                    await self.send_heartbeat()
                    self.last_heartbeat = datetime.utcnow()

            except Exception as e:
                self.logger.warning(f"[{self.exchange}] Heartbeat error: {e}")

    async def listen(self) -> None:
        """
        Main event loop for receiving and processing messages.

        Handles reconnection logic automatically.
        """
        while True:
            try:
                # Establish connection
                if not await self.connect():
                    self.logger.error(f"[{self.exchange}] Initial connection failed, retrying...")
                    await asyncio.sleep(5)
                    continue

                # Start heartbeat task
                heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                # Main message loop
                async for raw_message in self.ws:
                    try:
                        message = self.parse_message(raw_message)
                        if message:
                            self.messages_received += 1
                            await self.dispatch(message)
                    except Exception as e:
                        self.logger.error(f"[{self.exchange}] Message parsing error: {e}")
                        self.messages_failed += 1
                        continue

            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"[{self.exchange}] Connection closed")
                heartbeat_task.cancel()

                if not await self.reconnect():
                    self.logger.error(f"[{self.exchange}] Reconnection failed, stopping listener")
                    break

            except Exception as e:
                self.logger.error(f"[{self.exchange}] Listener error: {e}")
                heartbeat_task.cancel()

                if not await self.reconnect():
                    break

    def get_stats(self) -> Dict[str, Any]:
        """Return connection statistics"""
        return {
            'exchange': self.exchange,
            'state': self.state.value,
            'messages_received': self.messages_received,
            'messages_failed': self.messages_failed,
            'reconnect_attempts': self.reconnect_attempts,
            'last_heartbeat': self.last_heartbeat,
        }