"""
Exchange Connection Manager - Python equivalent of exchange_connection_manager.h/.cpp

This module manages WebSocket connections to cryptocurrency exchanges
for real-time market data streaming.
"""

import asyncio
import json
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

from .config_types import ExchangeConfig
from .market_data_types import Trade, OrderbookSnapshot
from .utilities import get_current_timestamp

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration"""
    url: str
    ping_interval: float = 30.0  # seconds
    pong_timeout: float = 10.0   # seconds
    connection_timeout: float = 30.0  # seconds
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 5.0  # seconds


class ExchangeConnectionManager:
    """
    Manages WebSocket connections to cryptocurrency exchanges.

    This class handles connection establishment, subscription management,
    and data reception from multiple exchanges concurrently.
    """

    # Exchange WebSocket URLs
    EXCHANGE_URLS = {
        "binance": "wss://stream.binance.com:9443/ws/",
        "okx": "wss://ws.okx.com:8443/ws/v5/public",
        "bybit": "wss://stream.bybit.com/v5/public/spot",
        "mexc": "wss://wbs.mexc.com/ws",
        "gate": "wss://api.gate.io/ws/v4/"
    }

    def __init__(self, trade_callback: Optional[Callable[[Trade], None]] = None,
                 orderbook_callback: Optional[Callable[[OrderbookSnapshot], None]] = None):
        """
        Initialize the exchange connection manager.

        Args:
            trade_callback: Callback function for trade data
            orderbook_callback: Callback function for orderbook data
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library is required for ExchangeConnectionManager")

        self.trade_callback = trade_callback
        self.orderbook_callback = orderbook_callback

        # Connection management
        self.connections: Dict[str, Dict[str, Any]] = {}  # exchange -> connection info
        self.running = False
        self.event_loop = None
        self.loop_thread = None

        # Statistics
        self.messages_received = 0
        self.trades_received = 0
        self.orderbooks_received = 0
        self.connection_errors = 0

        print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Initialized")

    def subscribe(self, configs: List[ExchangeConfig]) -> None:
        """
        Subscribe to market data streams for the given exchange configurations.

        Args:
            configs: List of exchange configurations
        """
        print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Starting subscription process...")
        print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Number of exchange configs: {len(configs)}")

        # Start event loop in background thread
        self._start_event_loop()

        # Create subscriptions for each exchange
        for config in configs:
            print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Processing exchange: {config.exchange_name}")
            print(f"[{get_current_timestamp()}][INFO]   Market type: {config.market_type}")
            print(f"[{get_current_timestamp()}][INFO]   Number of symbols: {len(config.symbols)}")
            print(f"[{get_current_timestamp()}][INFO]   Number of channels: {len(config.channels)}")

            # Create connection for this exchange
            self._create_exchange_connection(config)

        print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Subscription setup completed")

    def _start_event_loop(self) -> None:
        """Start the asyncio event loop in a background thread"""
        if self.event_loop is not None:
            return

        def run_loop():
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_forever()

        self.loop_thread = threading.Thread(target=run_loop, daemon=True, name="WebSocketEventLoop")
        self.loop_thread.start()

        # Wait for loop to start
        time.sleep(0.1)

    def _create_exchange_connection(self, config: ExchangeConfig) -> None:
        """Create WebSocket connection for an exchange"""
        exchange = config.exchange_name

        if exchange not in self.EXCHANGE_URLS:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Unsupported exchange: {exchange}")
            return

        base_url = self.EXCHANGE_URLS[exchange]
        ws_config = WebSocketConfig(url=base_url)

        # Store connection info
        self.connections[exchange] = {
            'config': config,
            'ws_config': ws_config,
            'connected': False,
            'task': None,
            'last_message': 0,
            'reconnect_count': 0
        }

        # Start connection task
        if self.event_loop:
            asyncio.run_coroutine_threadsafe(
                self._connect_exchange(exchange),
                self.event_loop
            )

    async def _connect_exchange(self, exchange: str) -> None:
        """Connect to an exchange's WebSocket and handle data"""
        conn_info = self.connections[exchange]
        config = conn_info['config']
        ws_config = conn_info['ws_config']

        while self.running:
            try:
                print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Connecting to {exchange}...")

                async with websockets.connect(
                    ws_config.url,
                    ping_interval=ws_config.ping_interval,
                    ping_timeout=ws_config.pong_timeout,
                    close_timeout=ws_config.connection_timeout
                ) as websocket:
                    print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Connected to {exchange}")
                    conn_info['connected'] = True
                    conn_info['reconnect_count'] = 0

                    # Subscribe to streams
                    await self._subscribe_streams(websocket, exchange, config)

                    # Handle messages
                    async for message in websocket:
                        try:
                            await self._handle_message(exchange, message)
                            conn_info['last_message'] = time.time()
                            self.messages_received += 1
                        except Exception as e:
                            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error handling message from {exchange}: {e}")

            except Exception as e:
                conn_info['connected'] = False
                self.connection_errors += 1
                print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Connection error for {exchange}: {e}")

                # Reconnection logic
                if conn_info['reconnect_count'] < ws_config.max_reconnect_attempts:
                    conn_info['reconnect_count'] += 1
                    delay = ws_config.reconnect_delay * conn_info['reconnect_count']  # Exponential backoff
                    print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Reconnecting to {exchange} in {delay}s (attempt {conn_info['reconnect_count']})")
                    await asyncio.sleep(delay)
                else:
                    print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Max reconnection attempts reached for {exchange}")
                    break

    async def _subscribe_streams(self, websocket, exchange: str, config: ExchangeConfig) -> None:
        """Subscribe to data streams for an exchange"""
        if exchange == "binance":
            await self._subscribe_binance(websocket, config)
        elif exchange == "okx":
            await self._subscribe_okx(websocket, config)
        elif exchange == "bybit":
            await self._subscribe_bybit(websocket, config)
        else:
            print(f"[{get_current_timestamp()}][WARNING] ExchangeConnectionManager: Subscription not implemented for {exchange}")

    async def _subscribe_binance(self, websocket, config: ExchangeConfig) -> None:
        """Subscribe to Binance streams"""
        streams = []

        for symbol in config.symbols:
            # Convert to Binance format (e.g., BTCUSDT -> btcusdt)
            binance_symbol = symbol.lower().replace('-', '')

            if "TRADE" in config.channels:
                streams.append(f"{binance_symbol}@trade")
            if "MARKET_DEPTH" in config.channels:
                streams.append(f"{binance_symbol}@depth5")  # Top 5 levels

        if streams:
            subscription = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            await websocket.send(json.dumps(subscription))
            print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Subscribed to Binance streams: {streams}")

    async def _subscribe_okx(self, websocket, config: ExchangeConfig) -> None:
        """Subscribe to OKX streams"""
        args = []

        for symbol in config.symbols:
            # OKX uses format like "BTC-USDT"
            if "TRADE" in config.channels:
                args.append({"channel": "trades", "instId": symbol})
            if "MARKET_DEPTH" in config.channels:
                args.append({"channel": "books5", "instId": symbol})

        if args:
            subscription = {
                "op": "subscribe",
                "args": args
            }
            await websocket.send(json.dumps(subscription))
            print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Subscribed to OKX streams: {len(args)} channels")

    async def _subscribe_bybit(self, websocket, config: ExchangeConfig) -> None:
        """Subscribe to Bybit streams"""
        topics = []

        for symbol in config.symbols:
            if "TRADE" in config.channels:
                topics.append(f"publicTrade.{symbol}")
            if "MARKET_DEPTH" in config.channels:
                topics.append(f"orderbook.5.{symbol}")

        if topics:
            subscription = {
                "op": "subscribe",
                "args": topics
            }
            await websocket.send(json.dumps(subscription))
            print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Subscribed to Bybit streams: {topics}")

    async def _handle_message(self, exchange: str, message: str) -> None:
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)

            if exchange == "binance":
                await self._handle_binance_message(data)
            elif exchange == "okx":
                await self._handle_okx_message(data)
            elif exchange == "bybit":
                await self._handle_bybit_message(data)

        except json.JSONDecodeError:
            # Skip non-JSON messages (like ping/pong)
            pass
        except Exception as e:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error processing message from {exchange}: {e}")

    async def _handle_binance_message(self, data: Dict[str, Any]) -> None:
        """Handle Binance WebSocket message"""
        if "stream" in data:
            stream = data["stream"]
            payload = data["data"]

            if "@trade" in stream:
                # Trade message
                trade = self._parse_binance_trade(payload)
                if trade and self.trade_callback:
                    self.trade_callback(trade)
                    self.trades_received += 1

            elif "@depth" in stream:
                # Orderbook message
                orderbook = self._parse_binance_orderbook(payload)
                if orderbook and self.orderbook_callback:
                    self.orderbook_callback(orderbook)
                    self.orderbooks_received += 1

    async def _handle_okx_message(self, data: Dict[str, Any]) -> None:
        """Handle OKX WebSocket message"""
        if data.get("event") == "subscribe":
            print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: OKX subscription confirmed")
            return

        if "arg" in data and "data" in data:
            arg = data["arg"]
            channel = arg.get("channel")

            if channel == "trades":
                for trade_data in data["data"]:
                    trade = self._parse_okx_trade(trade_data, arg)
                    if trade and self.trade_callback:
                        self.trade_callback(trade)
                        self.trades_received += 1

            elif channel.startswith("books"):
                for ob_data in data["data"]:
                    orderbook = self._parse_okx_orderbook(ob_data, arg)
                    if orderbook and self.orderbook_callback:
                        self.orderbook_callback(orderbook)
                        self.orderbooks_received += 1

    async def _handle_bybit_message(self, data: Dict[str, Any]) -> None:
        """Handle Bybit WebSocket message"""
        if data.get("op") == "subscribe":
            print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Bybit subscription confirmed")
            return

        topic = data.get("topic", "")
        payload = data.get("data", {})

        if "publicTrade" in topic:
            for trade_data in payload:
                trade = self._parse_bybit_trade(trade_data, topic)
                if trade and self.trade_callback:
                    self.trade_callback(trade)
                    self.trades_received += 1

        elif "orderbook" in topic:
            orderbook = self._parse_bybit_orderbook(payload, topic)
            if orderbook and self.orderbook_callback:
                self.orderbook_callback(orderbook)
                self.orderbooks_received += 1

    def _parse_binance_trade(self, data: Dict[str, Any]) -> Optional[Trade]:
        """Parse Binance trade message"""
        try:
            return Trade(
                timestamp_us=int(data["T"]) * 1000,  # Convert ms to µs
                exchange="binance",
                symbol=data["s"],
                market_type="spot",
                trade_id=data["t"],
                price=float(data["p"]),
                quantity=float(data["q"]),
                side="buy" if data["m"] else "sell",  # m=true means buyer is market maker
                is_buyer_maker=data["m"]
            )
        except (KeyError, ValueError) as e:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error parsing Binance trade: {e}")
            return None

    def _parse_okx_trade(self, data: Dict[str, Any], arg: Dict[str, Any]) -> Optional[Trade]:
        """Parse OKX trade message"""
        try:
            return Trade(
                timestamp_us=int(data["ts"]) * 1000,  # Convert ms to µs
                exchange="okx",
                symbol=arg["instId"],
                market_type="spot",
                trade_id=data["tradeId"],
                price=float(data["px"]),
                quantity=float(data["sz"]),
                side=data["side"]
            )
        except (KeyError, ValueError) as e:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error parsing OKX trade: {e}")
            return None

    def _parse_bybit_trade(self, data: Dict[str, Any], topic: str) -> Optional[Trade]:
        """Parse Bybit trade message"""
        try:
            # Extract symbol from topic (format: publicTrade.BTCUSDT)
            symbol = topic.split(".", 2)[1]

            return Trade(
                timestamp_us=int(data["T"]) * 1000,  # Convert ms to µs
                exchange="bybit",
                symbol=symbol,
                market_type="spot",
                trade_id=data["i"],
                price=float(data["p"]),
                quantity=float(data["v"]),
                side=data["S"].lower()
            )
        except (KeyError, ValueError, IndexError) as e:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error parsing Bybit trade: {e}")
            return None

    def _parse_binance_orderbook(self, data: Dict[str, Any]) -> Optional[OrderbookSnapshot]:
        """Parse Binance orderbook message"""
        try:
            bids = [[float(price), float(qty)] for price, qty in data.get("bids", [])]
            asks = [[float(price), float(qty)] for price, qty in data.get("asks", [])]

            return OrderbookSnapshot(
                timestamp_us=int(data["E"]) * 1000,  # Convert ms to µs
                exchange="binance",
                symbol=data["s"],
                market_type="spot",
                bids_json=json.dumps(bids),
                asks_json=json.dumps(asks)
            )
        except (KeyError, ValueError) as e:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error parsing Binance orderbook: {e}")
            return None

    def _parse_okx_orderbook(self, data: Dict[str, Any], arg: Dict[str, Any]) -> Optional[OrderbookSnapshot]:
        """Parse OKX orderbook message"""
        try:
            bids = [[float(price), float(qty)] for price, qty in data.get("bids", [])]
            asks = [[float(price), float(qty)] for price, qty in data.get("asks", [])]

            return OrderbookSnapshot(
                timestamp_us=int(data["ts"]) * 1000,  # Convert ms to µs
                exchange="okx",
                symbol=arg["instId"],
                market_type="spot",
                bids_json=json.dumps(bids),
                asks_json=json.dumps(asks)
            )
        except (KeyError, ValueError) as e:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error parsing OKX orderbook: {e}")
            return None

    def _parse_bybit_orderbook(self, data: Dict[str, Any], topic: str) -> Optional[OrderbookSnapshot]:
        """Parse Bybit orderbook message"""
        try:
            # Extract symbol from topic (format: orderbook.5.BTCUSDT)
            symbol = topic.split(".", 2)[2]

            bids = [[float(level[0]), float(level[1])] for level in data.get("b", [])]
            asks = [[float(level[0]), float(level[1])] for level in data.get("a", [])]

            return OrderbookSnapshot(
                timestamp_us=int(data["ts"]) * 1000,  # Convert ms to µs
                exchange="bybit",
                symbol=symbol,
                market_type="spot",
                bids_json=json.dumps(bids),
                asks_json=json.dumps(asks)
            )
        except (KeyError, ValueError, IndexError) as e:
            print(f"[{get_current_timestamp()}][ERROR] ExchangeConnectionManager: Error parsing Bybit orderbook: {e}")
            return None

    def start(self) -> None:
        """Start the connection manager"""
        print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Starting...")
        self.running = True

    def stop(self) -> None:
        """Stop the connection manager and close all connections"""
        print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Stopping...")
        self.running = False

        # Stop event loop
        if self.event_loop:
            self.event_loop.call_soon_threadsafe(self.event_loop.stop)

        if self.loop_thread:
            self.loop_thread.join(timeout=5.0)

        print(f"[{get_current_timestamp()}][INFO] ExchangeConnectionManager: Stopped")

    def is_running(self) -> bool:
        """Check if the connection manager is running"""
        return self.running

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'messages_received': self.messages_received,
            'trades_received': self.trades_received,
            'orderbooks_received': self.orderbooks_received,
            'connection_errors': self.connection_errors,
            'active_connections': sum(1 for conn in self.connections.values() if conn['connected'])
        }