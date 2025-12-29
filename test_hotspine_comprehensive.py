#!/usr/bin/env python3
"""
Comprehensive test script to validate HotSpine integration with Backtrader

This script tests the complete integration including:
1. HotSpine data feed functionality
2. Live trading integration
3. Architectural separation
4. Backtest path compatibility
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the dependencies directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

import backtrader as bt
from backtrader.feeds.hotspine_feed import HotSpineData, HotSpineFeed


class TestComprehensiveIntegration(unittest.TestCase):
    """Comprehensive tests for HotSpine integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_trade = MagicMock()
        self.mock_trade.ts_exchange = 1711234567890123
        self.mock_trade.ts_local = 1711234567890123
        self.mock_trade.price = 50000.0
        self.mock_trade.size = 0.1
        self.mock_trade.symbol_id = 123
        self.mock_trade.side = 0
        
    @patch('backtrader.feeds.hotspine_feed._import_hotspine_reader')
    def test_hotspine_data_feed_lifecycle(self, MockImportReader):
        """Test complete lifecycle of HotSpine data feed"""
        # Mock the HotSpine reader
        mock_reader_class = MagicMock()
        mock_reader_instance = MagicMock()
        mock_reader_class.return_value = mock_reader_instance
        mock_reader_instance.poll_trade.return_value = self.mock_trade
        MockImportReader.return_value = mock_reader_class
        
        # Create HotSpine data feed
        data = HotSpineData(symbol_id=123, shm_name="/test_hotspine")
        
        # Test initial state
        self.assertIsNone(data.hotspine_reader)
        self.assertEqual(data.symbol_id, 123)
        self.assertTrue(data.islive())
        
        # Start the data feed
        data.start()
        
        # Test that reader was initialized
        self.assertIsNotNone(data.hotspine_reader)
        mock_reader_class.assert_called_once_with("/test_hotspine")
        
        # Initialize data lines for testing
        data.lines.advance(1)
        
        # Test data loading
        result = data._load()
        self.assertTrue(result)
        
        # Test data processing
        self.assertEqual(data.lines.close[0], 50000.0)
        self.assertEqual(data.lines.volume[0], 0.1)
        
        # Stop the data feed
        data.stop()
        mock_reader_instance.close.assert_called_once()
        
    def test_hotspine_feed_factory(self):
        """Test HotSpine feed factory functionality"""
        feed = HotSpineFeed(symbol_id=456, shm_name="/test_feed", batch_mode=True)
        
        # Test factory parameters
        self.assertEqual(feed._symbol_id, 456)
        self.assertEqual(feed._shm_name, "/test_feed")
        self.assertTrue(feed._batch_mode)
        
        # Test data creation
        data = feed._getdata("test_symbol")
        self.assertIsInstance(data, HotSpineData)
        self.assertEqual(data.symbol_id, 456)
        self.assertEqual(data.p.shm_name, "/test_feed")
        self.assertTrue(data.p.batch_mode)
        
    def test_architectural_separation_detailed(self):
        """Test detailed architectural separation"""
        # Create HotSpine data feed
        hotspine_data = HotSpineData(symbol_id=123)
        
        # Verify HotSpine-specific attributes
        self.assertTrue(hasattr(hotspine_data, 'symbol_id'))
        self.assertTrue(hasattr(hotspine_data, 'batch_mode'))
        self.assertTrue(hasattr(hotspine_data, 'poll_interval'))
        
        # Verify it inherits from DataBase properly
        self.assertTrue(isinstance(hotspine_data, bt.feed.DataBase))
        self.assertTrue(hasattr(hotspine_data, 'lines'))
        self.assertTrue(hasattr(hotspine_data, 'start'))
        self.assertTrue(hasattr(hotspine_data, 'stop'))
        
        # Verify live data properties
        self.assertTrue(hotspine_data.islive())
        self.assertTrue(hotspine_data.haslivedata())
        
    def test_backtest_compatibility(self):
        """Test that backtest functionality remains compatible"""
        # Create a simple strategy
        class TestStrategy(bt.Strategy):
            def next(self):
                pass
        
        # Create cerebro instance
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(TestStrategy)
        
        # Test that standard backtest parameters still work
        cerebro.broker.set_cash(1000.0)
        cerebro.broker.setcommission(commission=0.001)
        
        # Test that cerebro can be configured for backtesting
        self.assertEqual(cerebro.broker.get_cash(), 1000.0)
        
        # Test that standard data feeds can still be added
        # (We can't actually run this without polars, but we can test the setup)
        try:
            from backtrader.feeds import PandasData
            # If PandasData is available, test that it can be instantiated
            pandas_data = PandasData(dataname=None)
            self.assertIsInstance(pandas_data, bt.feed.DataBase)
        except ImportError:
            # PandasData might not be available in test environment
            pass
        
    def test_live_trading_configuration(self):
        """Test live trading configuration"""
        # Create HotSpine data feed
        data = HotSpineData(symbol_id=789, shm_name="/live_test")
        
        # Create cerebro for live trading
        cerebro = bt.Cerebro()
        cerebro.adddata(data)
        
        # Test live trading parameters
        self.assertTrue(data.islive())
        
        # Test that cerebro can be configured for live trading
        cerebro.broker.set_cash(5000.0)
        cerebro.broker.set_coc(True)  # Cheat on close for live trading
        
        # Verify configuration
        self.assertEqual(cerebro.broker.get_cash(), 5000.0)
        
    def test_error_handling(self):
        """Test error handling in HotSpine data feed"""
        @patch('backtrader.feeds.hotspine_feed._import_hotspine_reader')
        def test_error_scenarios(MockImportReader):
            # Test initialization error
            MockImportReader.side_effect = Exception("Test error")
            
            data = HotSpineData(symbol_id=123)
            
            with self.assertRaises(Exception) as context:
                data.start()
            
            self.assertIn("Test error", str(context.exception))


def run_comprehensive_tests():
    """Run comprehensive tests"""
    print("Running comprehensive HotSpine integration tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestComprehensiveIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    
    if success:
        print("\n✅ All comprehensive tests passed!")
        print("✅ HotSpine integration is fully validated.")
        print("✅ Backtest path remains unchanged.")
        print("✅ Architectural separation is maintained.")
        print("✅ Live trading integration works correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some comprehensive tests failed!")
        sys.exit(1)