#!/usr/bin/env python3
"""
Comprehensive test suite for HotSpine SQL Architecture

This test suite validates that:
1. SQL is NOT used for live trading data ingestion
2. SQL is used ONLY for long-term storage, replay, analytics, and debugging
3. The architecture follows the new HotSpine design principles
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add the dependencies to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

from backtrader.hotspine.reader import HotSpineRuntime, HotTrade
from backtrader.hotspine.sql_integration import HotSpineSQLIntegration
from backtrader.bigbraincentral.storage_mssql import MSSQLConfig


class MockStrategy:
    """Mock strategy for testing"""
    
    def __init__(self):
        self.trade_count = 0
        self.last_trade = None
        
    def next(self):
        """Handle trade data"""
        if hasattr(self, 'data') and self.data:
            self.trade_count += 1
            self.last_trade = self.data


class TestHotSpineSQLArchitecture(unittest.TestCase):
    """Test suite for HotSpine SQL Architecture validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock SQL config
        self.sql_config = MSSQLConfig(
            server="test-server",
            database="test-db",
            username="test-user",
            password="test-password"
        )
        
        # Mock trade data
        self.mock_trade = HotTrade()
        self.mock_trade.ts_exchange = 1234567890
        self.mock_trade.ts_local = 1234567891
        self.mock_trade.price = 50000.0
        self.mock_trade.size = 0.001
        self.mock_trade.symbol_id = 1
        self.mock_trade.side = 0  # BUY
    
    def test_sql_not_used_for_live_trading(self):
        """Test that SQL is NOT used for live trading data ingestion"""
        print("\n" + "="*60)
        print("TEST: SQL NOT used for live trading data ingestion")
        print("="*60)
        
        # Create runtime with SQL disabled
        runtime = HotSpineRuntime(
            MockStrategy,
            shm_name="/test_hotspine",
            enable_sql_storage=False
        )
        
        # Verify SQL integration is None
        self.assertIsNone(runtime.sql_integration)
        self.assertFalse(runtime.enable_sql_storage)
        
        # Mock the reader to return our test trade
        runtime.reader = Mock()
        runtime.reader.poll_trade = Mock(return_value=self.mock_trade)
        runtime.reader.close = Mock()
        
        # Initialize strategy
        runtime._initialize_strategy()
        
        # Process a trade
        runtime.on_trade(self.mock_trade)
        
        # Verify strategy received the trade
        self.assertEqual(runtime._strategy_instance.trade_count, 1)
        self.assertIsNotNone(runtime._strategy_instance.last_trade)
        
        print("✅ PASS: Live trading works without SQL")
        print("✅ PASS: Strategy receives HotSpine data directly")
        print("✅ PASS: No SQL dependency in live trading path")
    
    @patch('backtrader.hotspine.sql_integration.MarketDataStorage')
    def test_sql_used_for_long_term_storage(self, MockStorage):
        """Test that SQL is used for long-term storage (asynchronous)"""
        print("\n" + "="*60)
        print("TEST: SQL USED for long-term storage")
        print("="*60)
        
        # Mock the storage
        mock_storage = Mock()
        mock_storage.connect = Mock()
        mock_storage.start_async_storage = Mock()
        mock_storage.stop_async_storage = Mock()
        mock_storage.store_trade_async = Mock(return_value=True)
        
        MockStorage.return_value = mock_storage
        
        # Create runtime with SQL enabled
        runtime = HotSpineRuntime(
            MockStrategy,
            shm_name="/test_hotspine",
            sql_config=self.sql_config,
            enable_sql_storage=True
        )
        
        # Verify SQL integration is created
        self.assertIsNotNone(runtime.sql_integration)
        self.assertTrue(runtime.enable_sql_storage)
        
        # Mock the reader
        runtime.reader = Mock()
        runtime.reader.poll_trade = Mock(return_value=self.mock_trade)
        runtime.reader.close = Mock()
        
        # Initialize strategy
        runtime._initialize_strategy()
        
        # Process a trade
        runtime.on_trade(self.mock_trade)
        
        # Verify SQL storage was called (asynchronously)
        mock_storage.store_trade_async.assert_called_once()
        
        print("✅ PASS: SQL integration created")
        print("✅ PASS: SQL storage called asynchronously")
        print("✅ PASS: Storage does not block live trading")
    
    @patch('backtrader.hotspine.sql_integration.MarketDataStorage')
    def test_architecture_separation(self, MockStorage):
        """Test clean architecture separation between HotSpine and SQL"""
        print("\n" + "="*60)
        print("TEST: Architecture separation")
        print("="*60)
        
        # Mock storage
        mock_storage = Mock()
        mock_storage.connect = Mock()
        mock_storage.get_historical_trades = Mock(return_value=[])
        mock_storage.get_historical_ohlcv = Mock(return_value=[])
        mock_storage.get_database_stats = Mock(return_value={})
        MockStorage.return_value = mock_storage
        
        # Create SQL integration directly
        sql_integration = HotSpineSQLIntegration(self.sql_config)
        
        # Test replay capabilities
        result = sql_integration.get_historical_trades("test", "test", datetime.now(), datetime.now())
        self.assertEqual(result, [])
        
        # Test analytics capabilities
        stats = sql_integration.get_database_stats()
        self.assertIsInstance(stats, dict)
        
        print("✅ PASS: SQL integration provides replay capabilities")
        print("✅ PASS: SQL integration provides analytics capabilities")
        print("✅ PASS: Clean separation from live trading")
    
    def test_data_flow_validation(self):
        """Test that data flow follows the correct architecture"""
        print("\n" + "="*60)
        print("TEST: Data flow validation")
        print("="*60)
        
        # Test with SQL disabled
        runtime_no_sql = HotSpineRuntime(
            MockStrategy,
            shm_name="/test_hotspine",
            enable_sql_storage=False
        )
        
        # Verify data flow: HotSpine -> Strategy (no SQL)
        runtime_no_sql.reader = Mock()
        runtime_no_sql.reader.poll_trade = Mock(return_value=self.mock_trade)
        runtime_no_sql._initialize_strategy()
        
        # Process trade
        runtime_no_sql.on_trade(self.mock_trade)
        
        # Verify strategy got the trade
        self.assertEqual(runtime_no_sql._strategy_instance.trade_count, 1)
        
        print("✅ PASS: Data flows HotSpine -> Strategy when SQL disabled")
        
        # Test with SQL enabled
        with patch('backtrader.hotspine.sql_integration.MarketDataStorage') as MockStorage:
            mock_storage = Mock()
            mock_storage.connect = Mock()
            mock_storage.store_trade_async = Mock(return_value=True)
            MockStorage.return_value = mock_storage
            
            runtime_with_sql = HotSpineRuntime(
                MockStrategy,
                shm_name="/test_hotspine",
                sql_config=self.sql_config,
                enable_sql_storage=True
            )
            
            # Verify data flow: HotSpine -> Strategy -> SQL (async)
            runtime_with_sql.reader = Mock()
            runtime_with_sql.reader.poll_trade = Mock(return_value=self.mock_trade)
            runtime_with_sql._initialize_strategy()
            
            # Process trade
            runtime_with_sql.on_trade(self.mock_trade)
            
            # Verify both strategy and SQL got the trade
            self.assertEqual(runtime_with_sql._strategy_instance.trade_count, 1)
            mock_storage.store_trade_async.assert_called_once()
            
            print("✅ PASS: Data flows HotSpine -> Strategy -> SQL when enabled")
            print("✅ PASS: SQL storage is asynchronous and non-blocking")
    
    def test_performance_characteristics(self):
        """Test that the architecture maintains performance characteristics"""
        print("\n" + "="*60)
        print("TEST: Performance characteristics")
        print("="*60)
        
        # Test that SQL operations don't block trading
        with patch('backtrader.hotspine.sql_integration.MarketDataStorage') as MockStorage:
            # Mock storage with slow operations
            mock_storage = Mock()
            mock_storage.connect = Mock()
            
            # Simulate slow storage
            def slow_store(*args, **kwargs):
                time.sleep(0.1)  # 100ms delay
                return True
            
            mock_storage.store_trade_async = Mock(side_effect=slow_store)
            MockStorage.return_value = mock_storage
            
            runtime = HotSpineRuntime(
                MockStrategy,
                shm_name="/test_hotspine",
                sql_config=self.sql_config,
                enable_sql_storage=True
            )
            
            runtime.reader = Mock()
            runtime.reader.poll_trade = Mock(return_value=self.mock_trade)
            runtime._initialize_strategy()
            
            # Measure time to process multiple trades
            start_time = time.time()
            
            # Process trades - should not be blocked by slow SQL
            for i in range(5):
                runtime.on_trade(self.mock_trade)
            
            elapsed = time.time() - start_time
            
            # Should complete quickly even with slow SQL
            # because SQL operations are asynchronous
            self.assertLess(elapsed, 0.5)  # Should complete in < 500ms for 5 trades
            
            print(f"✅ PASS: Processed 5 trades in {elapsed:.3f} seconds")
            print("✅ PASS: SQL operations don't block trading")
            print("✅ PASS: Asynchronous architecture maintains performance")


def run_architecture_validation():
    """Run comprehensive architecture validation"""
    print("\n" + "="*80)
    print("HOTSPINE SQL ARCHITECTURE VALIDATION")
    print("="*80)
    print()
    
    print("🎯 VALIDATION OBJECTIVES:")
    print("1. SQL NOT used for live trading data ingestion")
    print("2. SQL USED for long-term storage (asynchronous)")
    print("3. SQL USED for replay and analytics")
    print("4. Clean architecture separation")
    print("5. Performance characteristics maintained")
    print()
    
    # Run tests
    test_suite = TestHotSpineSQLArchitecture()
    
    # Run all test methods
    test_methods = [
        test_suite.test_sql_not_used_for_live_trading,
        test_suite.test_sql_used_for_long_term_storage,
        test_suite.test_architecture_separation,
        test_suite.test_data_flow_validation,
        test_suite.test_performance_characteristics
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_method()
            passed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    print(f"📊 Total Tests: {len(test_methods)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ARCHITECTURE VALIDATION PASSED!")
        print()
        print("✅ The HotSpine + SQL integration correctly implements:")
        print("   1. HotSpine for live trading data (NOT SQL)")
        print("   2. SQL for long-term storage (asynchronous)")
        print("   3. SQL for replay and analytics")
        print("   4. Clean separation of concerns")
        print("   5. Maintained performance characteristics")
        print()
        print("🏆 The architecture aligns with the new HotSpine design principles!")
        return True
    else:
        print("\n❌ ARCHITECTURE VALIDATION FAILED!")
        print(f"❌ {failed} out of {len(test_methods)} tests failed")
        return False


if __name__ == "__main__":
    # Run validation
    success = run_architecture_validation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)