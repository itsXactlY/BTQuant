#!/usr/bin/env python3
"""
Test module for HotSpineReader based on actual implementation facts.

This test verifies the functionality of the HotSpineReader class based on the
actual code implementation found in hotspine_reader.py, without making
assumptions beyond what the code actually provides.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from python_market_data_collector.hotspine_reader import (
    HotSpineReader, SymbolMapper, HotTrade, HotOrderbookSnapshot, 
    HotOrderbookLevel, SharedMemoryHeader, TradeData, OrderbookData, Side
)


class TestSymbolMapper(unittest.TestCase):
    """Test SymbolMapper class based on actual implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mapper = SymbolMapper()

    def test_default_mappings_loaded(self):
        """Test that default mappings are loaded correctly."""
        exchange, symbol = self.mapper.get_symbol_info(1)
        self.assertEqual(exchange, "binance")
        self.assertEqual(symbol, "BTC-USDT")

    def test_register_new_symbol(self):
        """Test registering a new symbol."""
        symbol_id = self.mapper.register_symbol("test_exchange", "TEST-SYMBOL")
        self.assertGreaterEqual(symbol_id, 1000)  # User-defined IDs start from 1000
        
        exchange, symbol = self.mapper.get_symbol_info(symbol_id)
        self.assertEqual(exchange, "test_exchange")
        self.assertEqual(symbol, "TEST-SYMBOL")

    def test_resolve_trade_with_known_symbol(self):
        """Test resolving a trade with a known symbol."""
        # Create a mock HotTrade with known symbol_id
        trade = HotTrade(
            ts_exchange=1234567890,
            ts_local=1234567891,
            price=42000.0,
            size=1.0,
            symbol_id=1,  # BTC-USDT on binance
            side=0,  # BUY
            padding=(0, 0, 0)
        )
        
        resolved = self.mapper.resolve_trade(trade)
        self.assertEqual(resolved.exchange, "binance")
        self.assertEqual(resolved.symbol, "BTC-USDT")
        self.assertEqual(resolved.price, 42000.0)
        self.assertEqual(resolved.side, Side.BUY)

    def test_resolve_trade_with_unknown_symbol(self):
        """Test resolving a trade with an unknown symbol."""
        trade = HotTrade(
            ts_exchange=1234567890,
            ts_local=1234567891,
            price=42000.0,
            size=1.0,
            symbol_id=999999,  # Unknown symbol_id
            side=0,  # BUY
            padding=(0, 0, 0)
        )
        
        resolved = self.mapper.resolve_trade(trade)
        self.assertEqual(resolved.exchange, "unknown")
        self.assertEqual(resolved.symbol, "symbol_999999")


class TestHotSpineReader(unittest.TestCase):
    """Test HotSpineReader class based on actual implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the shared memory attachment to avoid actual system calls
        with patch.object(HotSpineReader, '_attach'):
            self.reader = HotSpineReader()

    def test_initialization(self):
        """Test that reader initializes with correct defaults."""
        self.assertIsNotNone(self.reader.symbol_mapper)
        self.assertTrue(self.reader.enable_stats)
        self.assertEqual(self.reader.poll_interval, 0.0001)

    def test_get_buffer_utilization_structure(self):
        """Test that get_buffer_utilization returns expected structure."""
        result = self.reader.get_buffer_utilization()
        
        expected_keys = [
            "trade_count", "trade_capacity", "trade_utilization_pct",
            "orderbook_count", "orderbook_capacity", "orderbook_utilization_pct"
        ]
        
        for key in expected_keys:
            self.assertIn(key, result)

    def test_get_statistics_structure(self):
        """Test that get_statistics returns expected structure."""
        stats = self.reader.get_statistics()
        
        expected_keys = [
            'trades_read', 'orderbooks_read', 'poll_calls', 'empty_polls',
            'bytes_read', 'lost_trades', 'lost_orderbooks', 'read_errors'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_list_available_data_types(self):
        """Test that list_available_data_types returns expected types."""
        types = self.reader.list_available_data_types()
        self.assertIn("trades", types)
        # Orderbooks may or may not be available depending on initialization


class TestDataClasses(unittest.TestCase):
    """Test data classes based on actual implementation."""

    def test_trade_data_properties(self):
        """Test TradeData properties."""
        trade = TradeData(
            timestamp_us=1634567890123456,
            local_timestamp_us=1634567890123457,
            price=42000.0,
            size=1.0,
            symbol_id=1,
            side=Side.BUY,
            exchange="binance",
            symbol="BTC-USDT"
        )
        
        # Test timestamp property
        self.assertEqual(trade.timestamp.year, 2021)
        self.assertEqual(trade.timestamp.month, 10)
        self.assertEqual(trade.timestamp.day, 18)

    def test_orderbook_data_methods(self):
        """Test OrderbookData methods."""
        orderbook = OrderbookData(
            timestamp_us=1634567890123456,
            local_timestamp_us=1634567890123457,
            symbol_id=1,
            bids=[(41999.0, 1.0), (41998.0, 2.0)],
            asks=[(42001.0, 1.0), (42002.0, 2.0)],
            exchange="binance",
            symbol="BTC-USDT"
        )
        
        # Test mid price calculation
        mid_price = orderbook.get_mid_price()
        self.assertEqual(mid_price, 42000.0)  # (41999.0 + 42001.0) / 2.0
        
        # Test spread calculation
        spread = orderbook.get_spread()
        self.assertEqual(spread, 2.0)  # 42001.0 - 41999.0


class TestHotTradeStructure(unittest.TestCase):
    """Test HotTrade structure based on actual implementation."""

    def test_hottrade_creation(self):
        """Test creating and using HotTrade structure."""
        trade = HotTrade(
            ts_exchange=1234567890,
            ts_local=1234567891,
            price=42000.0,
            size=1.0,
            symbol_id=1,
            side=0,  # BUY
            padding=(0, 0, 0)
        )
        
        self.assertEqual(trade.ts_exchange, 1234567890)
        self.assertEqual(trade.price, 42000.0)
        self.assertEqual(trade.side, 0)
        self.assertEqual(trade.side_enum, Side.BUY)
        
        # Test to_dict method
        trade_dict = trade.to_dict()
        self.assertEqual(trade_dict["price"], 42000.0)
        self.assertEqual(trade_dict["side"], "BUY")

    def test_hotorderbook_snapshot_creation(self):
        """Test creating and using HotOrderbookSnapshot structure."""
        # Create a snapshot with some bids and asks
        bids = [(41999.0, 1.0), (41998.0, 2.0)]
        asks = [(42001.0, 1.0), (42002.0, 2.0)]
        
        # Create the structure with ctypes arrays
        snapshot = HotOrderbookSnapshot()
        snapshot.ts_exchange = 1234567890
        snapshot.symbol_id = 1
        snapshot.bids_count = 2
        snapshot.asks_count = 2
        
        # Fill the bid levels
        for i, (price, size) in enumerate(bids):
            snapshot.bids[i].price = price
            snapshot.bids[i].size = size
            
        # Fill the ask levels
        for i, (price, size) in enumerate(asks):
            snapshot.asks[i].price = price
            snapshot.asks[i].size = size
        
        self.assertEqual(snapshot.ts_exchange, 1234567890)
        self.assertEqual(snapshot.symbol_id, 1)
        self.assertEqual(snapshot.bids_count, 2)
        self.assertEqual(snapshot.asks_count, 2)
        
        # Test getting bids and asks
        retrieved_bids = snapshot.get_bids()
        retrieved_asks = snapshot.get_asks()
        
        self.assertEqual(retrieved_bids, bids)
        self.assertEqual(retrieved_asks, asks)


class TestIntegration(unittest.TestCase):
    """Integration tests based on actual implementation."""

    def test_symbol_mapper_with_hottrade(self):
        """Test integration between SymbolMapper and HotTrade."""
        mapper = SymbolMapper()
        
        # Create a trade with known symbol
        trade_struct = HotTrade(
            ts_exchange=1234567890,
            ts_local=1234567891,
            price=42000.0,
            size=1.0,
            symbol_id=1,  # binance BTC-USDT
            side=0,  # BUY
            padding=(0, 0, 0)
        )
        
        # Resolve the trade
        resolved_trade = mapper.resolve_trade(trade_struct)
        
        # Verify the resolution worked correctly
        self.assertEqual(resolved_trade.exchange, "binance")
        self.assertEqual(resolved_trade.symbol, "BTC-USDT")
        self.assertEqual(resolved_trade.price, 42000.0)
        self.assertEqual(resolved_trade.side, Side.BUY)


if __name__ == '__main__':
    unittest.main()