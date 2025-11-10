"""
jrr_websocket/connectors/okx.py

OKX (formerly OKEx) WebSocket implementation with spot, margin, and derivatives support.
"""

import json
import asyncio
import hmac
import hashlib
import base64
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from .base import BaseWebSocketConnector, Message, ConnectionState

logger = logging.getLogger(__name__)


class OKXWebSocketConnector(BaseWebSocketConnector):
    """
    OKX public WebSocket connector supporting:
    - Spot trading (SPOT)
    - Margin trading (MARGIN)
    - Perpetual futures (SWAP)
    - Options (OPTION)
    - Real-time market data (trades, depth, candles)
    """

    BASE_URL = "wss://ws.okx.com:8443/ws/v5/public"
    PRIVATE_URL = "wss://ws.okx.com:8443/ws/v5/private"

    # Market types mapping
    INST_TYPE_MAP = {
        'SPOT': 'spot',
        'MARGIN': 'margin',
        'SWAP': 'perpetual',
        'FUTURES': 'futures',
        'OPTION': 'option',
    }

    def __init__(self, exchange: str, config: Dict[str, Any], logger=None):
        super().__init__(exchange, config, logger)
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.passphrase = config.get('passphrase')
        self.use_private = bool(self.api_key and self.api_secret)

    async def get_ws_url(self) -> str:
        """Return appropriate WebSocket URL"""
        if self.use_private:
            return self.PRIVATE_URL
        return self.BASE_URL

    async def subscribe(self, channels: List[str]) -> None:
        """
        Subscribe to OKX channels.

        Channel format: "{channel}:{instrument_id}"
        Examples:
        - "trades:BTC-USDT"              # Spot trades
        - "candle1m:BTC-USDT"            # 1m candles
        - "bbo-tbt:BTC-USDT"             # Best bid/offer real-time
        - "books5:BTC-USDT"              # Top 5 depth
        - "books50-l2-tbt:BTC-USDT"      # L2 depth updates
        """
        if not self.subscriptions:
            self.subscriptions = channels
        else:
            self.subscriptions.extend([c for c in channels if c not in self.subscriptions])

        # Parse channels to construct subscription message
        sub_list = []
        for channel in channels:
            parts = channel.split(':')
            if len(parts) == 2:
                sub_list.append({
                    'channel': parts[0],
                    'instId': parts[1],
                })

        subscribe_msg = {
            "op": "subscribe",
            "args": sub_list
        }

        self.logger.info(f"[okx] Subscribing to {len(channels)} channels")

        if self.ws and self.state == ConnectionState.CONNECTED:
            await self.ws.send(json.dumps(subscribe_msg))

    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate OKX WebSocket authentication headers"""
        timestamp = datetime.utcnow().isoformat() + 'Z'

        message = timestamp + 'GET' + '/users/self/verify'

        mac = hmac.new(
            bytes(self.api_secret, encoding='utf8'),
            bytes(message, encoding='utf8'),
            digestmod=hashlib.sha256
        )

        d = mac.digest()
        sign = base64.b64encode(d).decode()

        return {
            'op': 'login',
            'args': [{
                'apiKey': self.api_key,
                'passphrase': self.passphrase,
                'timestamp': timestamp,
                'sign': sign,
            }]
        }

    async def _authenticate(self) -> bool:
        """Authenticate private channel connection"""
        if not self.use_private:
            return True

        try:
            auth_msg = self._get_auth_headers()
            await self.ws.send(json.dumps(auth_msg))

            # Wait for authentication response
            auth_response = await asyncio.wait_for(self.ws.recv(), timeout=5)
            response = json.loads(auth_response)

            if response.get('code') == '0':
                self.logger.info("[okx] Authentication successful")
                return True
            else:
                self.logger.error(f"[okx] Authentication failed: {response}")
                return False

        except Exception as e:
            self.logger.error(f"[okx] Authentication error: {e}")
            return False

    async def connect(self) -> bool:
        """Establish connection with optional authentication"""
        if not await super().connect():
            return False

        # Authenticate if using private channels
        if self.use_private:
            if not await self._authenticate():
                await self.disconnect()
                return False

        return True

    def parse_message(self, raw_message: str) -> Optional[Message]:
        """
        Parse OKX WebSocket message.

        OKX uses a wrapper format: {"op": "...", "data": [...]}
        """
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            self.logger.warning(f"[okx] Invalid JSON: {raw_message[:100]}")
            return None

        # Handle subscription confirmations
        if data.get('op') in ['subscribe', 'login']:
            return None

        # Handle error messages
        if data.get('event') == 'error':
            self.logger.error(f"[okx] Error: {data}")
            return None

        # Extract data payload
        if 'data' not in data:
            return None

        payload = data.get('data', [])
        if not payload or not isinstance(payload, list):
            return None

        # Get first data item
        msg = payload[0]
        arg = data.get('arg', {})
        channel = arg.get('channel', '')

        # Route to appropriate parser with error handling
        try:
            if channel.startswith('trades'):
                return self._parse_trades(msg, arg)
            elif channel.startswith('candle'):
                return self._parse_candle(msg, arg, channel)
            elif 'books' in channel or 'bbo' in channel:
                return self._parse_depth(msg, arg)
            elif channel.startswith('funding'):
                return self._parse_funding(msg, arg)
            elif channel.startswith('mark-price'):
                return self._parse_mark_price(msg, arg)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"[okx] Error parsing {channel} message: {e}")
            return None

        return None

    def _parse_trades(self, msg: Dict[str, Any], arg: Dict[str, Any]) -> Message:
        """Parse trades channel"""
        return Message(
            type='trade',
            exchange='okx',
            symbol=arg.get('instId', '').lower().replace('-', '/'),
            timestamp=int(msg['ts']),
            data={
                'trade_id': msg.get('tradeId'),
                'price': float(msg['px']),
                'quantity': float(msg['sz']),
                'side': msg['side'],  # buy or sell
                'instrument_type': self.INST_TYPE_MAP.get(arg.get('instType', 'SPOT')),
            }
        )

    def _parse_candle(self, msg: Dict[str, Any], arg: Dict[str, Any], 
                     channel: str) -> Message:
        """Parse candle (kline) data"""
        # Extract timeframe from channel name: candle1m, candle5m, candle1h, etc.
        match = re.search(r'candle(\d+[mhd])', channel)
        timeframe = match.group(1) if match else '1m'

        return Message(
            type='ohlcv',
            exchange='okx',
            symbol=arg.get('instId', '').lower().replace('-', '/'),
            timestamp=int(msg[0]),  # Open time
            data={
                'timeframe': timeframe,
                'open': float(msg[1]),
                'high': float(msg[2]),
                'low': float(msg[3]),
                'close': float(msg[4]),
                'volume': float(msg[5]),
                'volume_usdt': float(msg[6]),
                'number_of_trades': msg[7] if len(msg) > 7 else 0,
                'instrument_type': self.INST_TYPE_MAP.get(arg.get('instType', 'SPOT')),
            }
        )

    def _parse_depth(self, msg: Dict[str, Any], arg: Dict[str, Any]) -> Message:
        """Parse depth/orderbook data"""
        return Message(
            type='orderbook',
            exchange='okx',
            symbol=arg.get('instId', '').lower().replace('-', '/'),
            timestamp=int(msg.get('ts', 0)),
            data={
                'bids': [[float(p), float(q)] for p, q in msg.get('bids', [])],
                'asks': [[float(p), float(q)] for p, q in msg.get('asks', [])],
                'checksum': msg.get('checksum'),
                'sequence': msg.get('seqId'),
                'instrument_type': self.INST_TYPE_MAP.get(arg.get('instType', 'SPOT')),
            }
        )

    def _parse_funding(self, msg: Dict[str, Any], arg: Dict[str, Any]) -> Message:
        """Parse funding rate data (perpetuals only)"""
        return Message(
            type='funding',
            exchange='okx',
            symbol=arg.get('instId', '').lower().replace('-', '/'),
            timestamp=int(msg.get('ts', 0)),
            data={
                'funding_rate': float(msg.get('fundingRate', 0)),
                'funding_time': int(msg.get('fundingTime', 0)),
                'next_funding_time': int(msg.get('nextFundingTime', 0)),
                'instrument_type': self.INST_TYPE_MAP.get(arg.get('instType', 'SWAP')),
            }
        )

    def _parse_mark_price(self, msg: Dict[str, Any], arg: Dict[str, Any]) -> Message:
        """Parse mark price data (perpetuals/futures only)"""
        return Message(
            type='mark_price',
            exchange='okx',
            symbol=arg.get('instId', '').lower().replace('-', '/'),
            timestamp=int(msg.get('ts', 0)),
            data={
                'mark_price': float(msg['markPx']),
                'index_price': float(msg.get('idxPx', 0)),
                'last_price': float(msg.get('lastPx', 0)),
                'instrument_type': self.INST_TYPE_MAP.get(arg.get('instType', 'SWAP')),
            }
        )

    async def send_heartbeat(self) -> None:
        """Send ping message to maintain connection"""
        if self.ws and self.state == ConnectionState.CONNECTED:
            try:
                await self.ws.send('ping')
            except Exception as e:
                self.logger.warning(f"[okx] Heartbeat error: {e}")
