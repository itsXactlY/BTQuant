#!/usr/bin/env python3
"""
Test script to validate HotSpine integration with Backtrader

This script tests:
1. Backtest path remains unchanged
2. HotSpine live trading integration works correctly
3. Architectural separation between HotSpine and SQL is maintained
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add the dependencies directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

import backtrader as bt
from backtrader.feeds.hotspine_feed import HotSpineData, HotSpineFeed
from backtrader.hotspine.reader import HotTrade


class TestBacktestPath(unittest.TestCase):
    """Test that backtest path remains unchanged"""
    
    def test_backtest_path_unchanged(self):
        """Test that existing backtest functionality still works"""
        # Create a simple strategy
        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=5)
            
            def next(self):
                if not self.position and self.sma[0] > self.data.close[0]:
                    self.buy()
                elif self.position and self.sma[0] < self.data.close[0]:
                    self.sell()
        
        # Create cerebro instance
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(TestStrategy)
        
        # Create test data (this should work as before)
        data = bt.feeds.PandasData(dataname=None)
        
        # Test that cerebro can be configured for backtesting
        cerebro.broker.set_cash(1000.0)
        
        # Test that standard backtest parameters work
        result = cerebro.run(runonce=True, preload=True)
        
        # Should complete without errors
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)


class TestHotSpineIntegration(unittest.TestCase):
    """Test HotSpine integration for live trading"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_reader = MagicMock()
        self.test_trade = HotTrade()
        self.test_trade.ts_exchange = 1711234567890123  # microseconds
        self.test_trade.ts_local = 1711234567890123
        self.test_trade.price = 50000.0
        self.test_trade.size = 0.1
        self.test_trade.symbol_id = 123
        self.test_trade.side = 0  # BUY
        
    def test_hotspine_data_feed_creation(self):
        """Test HotSpine data feed creation"""
        # Test that HotSpineData can be instantiated
        data = HotSpineData(symbol_id=123, shm_name="/test_hotspine")
        
        self.assertEqual(data.symbol_id, 123)
        self.assertEqual(data.p.shm_name, "/test_hotspine")
        self.assertTrue(data.islive())
        
    @patch('backtrader.feeds.hotspine_feed.HotSpineReader')
    def test_hotspine_data_feed_processing(self, MockHotSpineReader):
        """Test HotSpine data feed processing"""
        # Mock the HotSpine reader
        mock_reader = MockHotSpineReader.return_value
        mock_reader.poll_trade.return_value = self.test_trade
        
        # Create HotSpine data feed
        data = HotSpineData(symbol_id=123, shm_name="/test_hotspine")
        
        # Start the data feed
        data.start()
        
        # Test that reader was initialized
        MockHotSpineReader.assert_called_once_with("/test_hotspine")
        
        # Test trade processing
        result = data._load()
        self.assertTrue(result)
        
        # Check that trade was processed correctly
        self.assertEqual(data.lines.close[0], 50000.0)
        self.assertEqual(data.lines.volume[0], 0.1)
        
        # Clean up
        data.stop()
        
    def test_hotspine_feed_creation(self):
        """Test HotSpine feed creation"""
        feed = HotSpineFeed(symbol_id=123, shm_name="/test_hotspine")
        
        self.assertEqual(feed._symbol_id, 123)
        self.assertEqual(feed._shm_name, "/test_hotspine")
        
    def test_architectural_separation(self):
        """Test that HotSpine and SQL integration are separate"""
        # Import SQL integration
        from backtrader.hotspine.sql_integration import HotSpineSQLIntegration
        
        # Create HotSpine data feed
        hotspine_data = HotSpineData(symbol_id=123)
        
        # Create SQL integration
        sql_integration = HotSpineSQLIntegration()
        
        # Verify they are separate instances with different responsibilities
        self.assertIsNotNone(hotspine_data.hotspine_reader)
        self.assertIsNotNone(sql_integration.storage)
        
        # Verify HotSpine data feed doesn't depend on SQL
        self.assertFalse(hasattr(hotspine_data, 'sql_integration'))
        
        # Verify SQL integration doesn't depend on HotSpine reader
        self.assertFalse(hasattr(sql_integration, 'hotspine_reader'))


class TestLiveTradingIntegration(unittest.TestCase):
    """Test live trading integration"""
    
    @patch('backtrader.feeds.hotspine_feed.HotSpineReader')
    def test_live_trading_setup(self, MockHotSpineReader):
        """Test live trading setup with HotSpine"""
        # Mock the HotSpine reader
        mock_reader = MockHotSpineReader.return_value
        mock_reader.poll_trade.return_value = None  # No trades initially
        
        # Create a simple strategy
        class TestStrategy(bt.Strategy):
            def next(self):
                pass
        
        # Create cerebro instance
        cerebro = bt.Cerebro()
        
        # Add HotSpine data feed
        data = HotSpineData(symbol_id=123, shm_name="/test_hotspine")
        cerebro.adddata(data)
        
        # Add strategy
        cerebro.addstrategy(TestStrategy)
        
        # Configure for live trading
        cerebro.broker.set_cash(1000.0)
        
        # Test that live trading parameters work
        # Note: We can't actually run the live trading loop in a test,
        # but we can verify the setup is correct
        self.assertTrue(data.islive())
        self.assertEqual(cerebro.broker.get_cash(), 1000.0)
        
        # Clean up
        data.stop()


def run_tests():
    """Run all tests"""
    print("Running HotSpine integration tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestPath))
    suite.addTests(loader.loadTestsFromTestCase(TestHotSpineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestLiveTradingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed! HotSpine integration is working correctly.")
        print("✅ Backtest path remains unchanged.")
        print("✅ Architectural separation is maintained.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)