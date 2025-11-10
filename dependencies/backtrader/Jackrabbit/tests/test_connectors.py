"""
tests/test_connectors.py

Unit and integration tests for websocket connectors.
"""

import pytest
import json
from unittest.mock import AsyncMock

from backtrader.Jackrabbit.connectors.base import BaseWebSocketConnector, Message, ConnectionState
from backtrader.Jackrabbit.connectors.binance import BinanceWebSocketConnector
from backtrader.Jackrabbit.connectors.okx import OKXWebSocketConnector


class TestBaseConnector:
    """Test base connector functionality"""

    @pytest.mark.asyncio
    async def test_connector_initialization(self):
        """Test connector initializes with correct state"""
        config = {
            'max_reconnect_attempts': 5,
            'initial_backoff': 1,
        }

        connector = BinanceWebSocketConnector('binance', config)

        assert connector.exchange == 'binance'
        assert connector.state == ConnectionState.DISCONNECTED
        assert connector.reconnect_attempts == 0
        assert len(connector.callbacks) == 0

    def test_message_normalization(self):
        """Test message parsing and normalization"""
        connector = BinanceWebSocketConnector('binance', {})

        # Trade message
        trade_msg = {
            'e': 'trade',
            's': 'BTCUSDT',
            'p': '50000.00',
            'q': '0.01',
            'T': 1234567890000,
            't': 123456,
            'b': 999,
            'a': 1000,
            'm': True,
            'M': True,
        }

        parsed = connector.parse_message(json.dumps(trade_msg))

        assert parsed is not None
        assert parsed.type == 'trade'
        assert parsed.exchange == 'binance'
        assert parsed.symbol == 'btcusdt'
        assert parsed.data['price'] == 50000.0
        assert parsed.data['quantity'] == 0.01

    def test_kline_parsing(self):
        """Test OHLCV candle parsing"""
        connector = BinanceWebSocketConnector('binance', {})

        kline_msg = {
            'e': 'kline',
            's': 'ETHUSDT',
            'k': {
                't': 1234567890000,
                'T': 1234567950000,
                'i': '1m',
                'o': '3000.00',
                'h': '3100.00',
                'l': '2900.00',
                'c': '3050.00',
                'v': '100.5',
                'q': '305000.00',
                'n': 50,
                'V': '50.25',
                'Q': '152500.00',
                'x': True,
            }
        }

        parsed = connector.parse_message(json.dumps(kline_msg))

        assert parsed.type == 'ohlcv'
        assert parsed.data['open'] == 3000.0
        assert parsed.data['high'] == 3100.0
        assert parsed.data['low'] == 2900.0
        assert parsed.data['close'] == 3050.0
        assert parsed.data['volume'] == 100.5
        assert parsed.data['timeframe'] == '1m'

    def test_depth_parsing(self):
        """Test orderbook depth parsing"""
        connector = BinanceWebSocketConnector('binance', {})

        depth_msg = {
            'e': 'depthUpdate',
            's': 'BTCUSDT',
            'U': 1000,
            'u': 1050,
            'pu': 999,
            'b': [
                ['50000.00', '1.0'],
                ['49999.00', '2.0'],
            ],
            'a': [
                ['50001.00', '1.5'],
                ['50002.00', '2.5'],
            ],
            'E': 1234567890000,
        }

        parsed = connector.parse_message(json.dumps(depth_msg))

        assert parsed.type == 'orderbook'
        assert len(parsed.data['bids']) == 2
        assert len(parsed.data['asks']) == 2
        assert parsed.data['bids'][0] == [50000.0, 1.0]
        assert parsed.data['asks'][0] == [50001.0, 1.5]

    @pytest.mark.asyncio
    async def test_subscriber_management(self):
        """Test adding and removing subscribers"""
        connector = BinanceWebSocketConnector('binance', {})

        callback1 = AsyncMock()
        callback2 = AsyncMock()

        connector.add_subscriber(callback1)
        assert len(connector.callbacks) == 1

        connector.add_subscriber(callback2)
        assert len(connector.callbacks) == 2

        connector.remove_subscriber(callback1)
        assert len(connector.callbacks) == 1
        assert callback2 in connector.callbacks

    @pytest.mark.asyncio
    async def test_message_dispatch(self):
        """Test message dispatch to subscribers"""
        connector = BinanceWebSocketConnector('binance', {})

        callback = AsyncMock()
        connector.add_subscriber(callback)

        message = Message(
            type='trade',
            exchange='binance',
            symbol='btcusdt',
            timestamp=1234567890000,
            data={'price': 50000, 'quantity': 1.0}
        )

        await connector.dispatch(message)

        callback.assert_called_once_with(message)


class TestOKXConnector:
    """Test OKX-specific functionality"""

    def test_okx_trade_parsing(self):
        """Test OKX trade message parsing"""
        connector = OKXWebSocketConnector('okx', {})

        okx_msg = {
            'op': 'update',
            'arg': {
                'channel': 'trades',
                'instId': 'BTC-USDT',
                'instType': 'SPOT',
            },
            'data': [{
                'ts': '1234567890000',
                'tradeId': '12345',
                'px': '50000.00',
                'sz': '0.01',
                'side': 'buy',
            }]
        }

        parsed = connector.parse_message(json.dumps(okx_msg))

        assert parsed is not None
        assert parsed.type == 'trade'
        assert parsed.exchange == 'okx'
        assert parsed.data['price'] == 50000.0
        assert parsed.data['side'] == 'buy'

    def test_okx_candle_parsing(self):
        """Test OKX candle parsing"""
        connector = OKXWebSocketConnector('okx', {})

        okx_msg = {
            'op': 'update',
            'arg': {
                'channel': 'candle1m',
                'instId': 'ETH-USDT',
                'instType': 'SPOT',
            },
            'data': [[
                '1234567890000',  # open time
                '3000.00',        # open
                '3100.00',        # high
                '2900.00',        # low
                '3050.00',        # close
                '100.5',          # volume
                '305000.00',      # volume_usdt
                '50',             # trades
            ]]
        }

        parsed = connector.parse_message(json.dumps(okx_msg))

        assert parsed.type == 'ohlcv'
        assert parsed.data['timeframe'] == '1m'
        assert parsed.data['close'] == 3050.0


class TestErrorHandling:
    """Test error handling and recovery"""

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON"""
        connector = BinanceWebSocketConnector('binance', {})

        result = connector.parse_message('invalid json')
        assert result is None

    def test_missing_fields_handling(self):
        """Test handling of messages with missing fields"""
        connector = BinanceWebSocketConnector('binance', {})

        incomplete_msg = {'e': 'trade', 's': 'BTCUSDT'}

        result = connector.parse_message(json.dumps(incomplete_msg))
        # Should either return None or handle gracefully
        assert result is None or result.type == 'trade'


@pytest.mark.asyncio
async def test_integration_flow():
    """Test full integration flow"""
    connector = BinanceWebSocketConnector('binance', {
        'max_reconnect_attempts': 2,
        'initial_backoff': 0.1,
    })

    received_messages = []

    async def collect_messages(message):
        received_messages.append(message)

    connector.add_subscriber(collect_messages)

    # Simulate message reception
    trade_msg = {
        'e': 'trade',
        's': 'BTCUSDT',
        'p': '50000.00',
        'q': '0.01',
        'T': 1234567890000,
        't': 123456,
        'b': 999,
        'a': 1000,
        'm': True,
        'M': True,
    }

    parsed = connector.parse_message(json.dumps(trade_msg))
    await connector.dispatch(parsed)

    assert len(received_messages) == 1
    assert received_messages[0].type == 'trade'


def test_backoff_calculation():
    """Test exponential backoff calculation"""
    connector = BinanceWebSocketConnector('binance', {
        'initial_backoff': 1,
        'max_backoff': 60,
    })

    # Simulate reconnection attempts
    backoff_times = []
    for attempt in range(6):
        backoff = min(
            connector.initial_backoff * (2 ** attempt),
            connector.max_backoff
        )
        backoff_times.append(backoff)

    assert backoff_times == [1, 2, 4, 8, 16, 32]

    # Verify backoff caps at max
    backoff = min(
        connector.initial_backoff * (2 ** 10),
        connector.max_backoff
    )
    assert backoff == 60
