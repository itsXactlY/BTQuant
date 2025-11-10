"""
jrr_websocket/connectors/binance.py

Binance-specific WebSocket implementation with full market data support.
"""

import json
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .base import BaseWebSocketConnector, Message, ConnectionState

logger = logging.getLogger(__name__)


class BinanceWebSocketConnector(BaseWebSocketConnector):
    """
    Binance WebSocket connector supporting:
    - Real-time trades (24/7 order flow)
    - Partial book depth snapshots and deltas (L2 orderbook)
    - Kline (candlestick) data for multiple timeframes
    - Aggregated trade streams for high-frequency analysis
    """

    BASE_URL = "wss://stream.binance.com:9443/ws"
    COMBO_URL = "wss://stream.binance.com:9443/stream"

    # Supported kline intervals
    KLINE_INTERVALS = {
        '1s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    }

    def __init__(self, exchange: str, config: Dict[str, Any], logger=None):
        super().__init__(exchange, config, logger)
        self.session_id = None
        self.listen_key = None

    async def get_ws_url(self) -> str:
        """Build websocket URL with subscribed streams"""
        return self.BASE_URL

    async def subscribe(self, channels: List[str]) -> None:
        """
        Subscribe to Binance channels.

        Channel format examples:
        - "btcusdt@trade"              # Real-time trades
        - "btcusdt@depth20@100ms"      # L2 depth snapshots
        - "btcusdt@depth"              # Full depth deltas
        - "btcusdt@kline_1m"           # 1-minute candles
        - "btcusdt@aggTrade"           # Aggregate trades
        """
        if not self.subscriptions:
            self.subscriptions = channels
        else:
            self.subscriptions.extend([c for c in channels if c not in self.subscriptions])

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": channels,
            "id": 1
        }

        self.logger.info(f"[binance] Subscribing to {len(channels)} channels")

        if self.ws and self.state == ConnectionState.CONNECTED:
            await self.ws.send(json.dumps(subscribe_msg))

    def parse_message(self, raw_message: str) -> Optional[Message]:
        """
        Parse Binance WebSocket message and normalize to standard format.

        Handles:
        - Trade execution events (e='trade')
        - Depth updates (e='depthUpdate')
        - Kline (candlestick) events (e='kline')
        - Aggregate trade events (e='aggTrade')
        """
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            self.logger.warning(f"[binance] Invalid JSON: {raw_message[:100]}")
            return None

        # Result messages (subscription confirmations, etc.)
        if 'result' in data or 'id' in data:
            return None

        if 'e' not in data:
            return None

        event_type = data.get('e')

        # Route to appropriate parser with error handling
        try:
            if event_type == 'trade':
                return self._parse_trade(data)
            elif event_type == 'depthUpdate':
                return self._parse_depth(data)
            elif event_type == 'kline':
                return self._parse_kline(data)
            elif event_type == 'aggTrade':
                return self._parse_agg_trade(data)
        except KeyError as e:
            self.logger.warning(f"[binance] Missing required field {e} in {event_type} message")
            return None
        except (ValueError, TypeError) as e:
            self.logger.warning(f"[binance] Invalid data format in {event_type} message: {e}")
            return None

        return None

    def _parse_trade(self, data: Dict[str, Any]) -> Message:
        """Parse 'trade' event from Binance stream"""
        # Validate required fields
        required_fields = ['s', 'T', 't', 'p', 'q', 'b', 'a', 'm']
        for field in required_fields:
            if field not in data:
                raise KeyError(field)

        return Message(
            type='trade',
            exchange='binance',
            symbol=data['s'].lower(),
            timestamp=int(data['T']),
            data={
                'trade_id': data['t'],
                'price': float(data['p']),
                'quantity': float(data['q']),
                'buyer_order_id': data['b'],
                'seller_order_id': data['a'],
                'trade_time': data['T'],
                'is_buyer_maker': data['m'],  # True if buyer is market maker
                'is_best_match': data.get('M', True),
            }
        )

    def _parse_depth(self, data: Dict[str, Any]) -> Message:
        """Parse 'depthUpdate' event from Binance stream"""
        # Validate required fields
        required_fields = ['s', 'E', 'U', 'u']
        for field in required_fields:
            if field not in data:
                raise KeyError(field)

        return Message(
            type='orderbook',
            exchange='binance',
            symbol=data['s'].lower(),
            timestamp=int(data['E']),
            data={
                'bids': [[float(p), float(q)] for p, q in data.get('b', [])],
                'asks': [[float(p), float(q)] for p, q in data.get('a', [])],
                'first_update_id': data['U'],
                'final_update_id': data['u'],
                'previous_final_id': data.get('pu'),
            }
        )

    def _parse_kline(self, data: Dict[str, Any]) -> Message:
        """Parse 'kline' event from Binance stream"""
        if 'k' not in data:
            raise KeyError('k')

        kline = data['k']

        # Validate required kline fields
        required_fields = ['t', 'T', 'i', 'o', 'h', 'l', 'c', 'v', 'q', 'n', 'V', 'Q', 'x']
        for field in required_fields:
            if field not in kline:
                raise KeyError(f"k.{field}")

        return Message(
            type='ohlcv',
            exchange='binance',
            symbol=data['s'].lower(),
            timestamp=int(kline['t']),
            data={
                'timeframe': kline['i'],
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'quote_asset_volume': float(kline['q']),
                'number_of_trades': kline['n'],
                'taker_buy_volume': float(kline['V']),
                'taker_buy_quote_volume': float(kline['Q']),
                'is_closed': kline['x'],
                'open_time': kline['t'],
                'close_time': kline['T'],
            }
        )

    def _parse_agg_trade(self, data: Dict[str, Any]) -> Message:
        """Parse 'aggTrade' event from Binance stream"""
        # Validate required fields
        required_fields = ['s', 'T', 'a', 'p', 'q', 'f', 'l', 'm']
        for field in required_fields:
            if field not in data:
                raise KeyError(field)

        return Message(
            type='trade',
            exchange='binance',
            symbol=data['s'].lower(),
            timestamp=int(data['T']),
            data={
                'trade_id': data['a'],
                'price': float(data['p']),
                'quantity': float(data['q']),
                'first_trade_id': data['f'],
                'last_trade_id': data['l'],
                'trade_time': data['T'],
                'is_buyer_maker': data['m'],
            }
        )

    async def send_heartbeat(self) -> None:
        """Binance WebSocket uses automatic ping/pong, minimal action needed"""
        # Binance automatically sends pings and expects pongs
        # The websockets library handles this automatically
        pass


class BinanceUSDMFuturesConnector(BinanceWebSocketConnector):
    """
    Binance USDT-M Futures WebSocket connector.

    Inherits from spot connector but uses futures endpoints.
    """

    BASE_URL = "wss://fstream.binance.com:443/ws"
    COMBO_URL = "wss://fstream.binance.com:443/stream"

    # Additional channels for futures
    FUTURES_ONLY_CHANNELS = {
        'markPrice',      # Mark price updates
        'fundingRate',    # Funding rate
        'klines_long',    # Long liquidations
        'klines_short',   # Short liquidations
    }


class BinanceUSDCFuturesConnector(BinanceWebSocketConnector):
    """
    Binance USDC-M Futures (Quarterly/Perpetual) WebSocket connector.
    """

    BASE_URL = "wss://dstream.binance.com:443/ws"
    COMBO_URL = "wss://dstream.binance.com:443/stream"
