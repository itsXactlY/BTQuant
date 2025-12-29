#!/usr/bin/env python3
"""
Test script for HotSpine reader implementation

This script validates that the HotSpine reader works correctly with btq_live_runtime.
"""

import sys
import os
import time
import threading
from typing import Optional

# Add the dependencies to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

try:
    from backtrader.hotspine.reader import HotSpineReader, HotTrade, HotSpineRuntime
    print("✓ Successfully imported HotSpine reader")
except ImportError as e:
    print(f"✗ Failed to import HotSpine reader: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic HotSpine reader functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Test with default shared memory name
        reader = HotSpineReader("/btquant_hotspine")
        print("✓ HotSpineReader created successfully")
        
        # Test basic properties
        assert reader.is_healthy(), "Reader should be healthy"
        print("✓ Reader is healthy")
        
        # Test poll_trade (should return None when no data available)
        trade = reader.poll_trade()
        assert trade is None, "Should return None when no trades available"
        print("✓ poll_trade() returns None when no data available")
        
        # Test read_all_trades (should return empty list when no data available)
        trades = reader.read_all_trades()
        assert len(trades) == 0, "Should return empty list when no trades available"
        print("✓ read_all_trades() returns empty list when no data available")
        
        # Test lost count
        lost_count = reader.get_lost_count()
        assert isinstance(lost_count, int), "Lost count should be an integer"
        print(f"✓ Lost count: {lost_count}")
        
        # Test buffer utilization
        utilization = reader.get_buffer_utilization()
        assert "current_size" in utilization, "Utilization should have current_size"
        assert "capacity" in utilization, "Utilization should have capacity"
        print(f"✓ Buffer utilization: {utilization}")
        
        reader.close()
        print("✓ Reader closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_hotspine_runtime():
    """Test HotSpineRuntime integration"""
    print("\n=== Testing HotSpineRuntime Integration ===")
    
    # Create a simple test strategy
    class TestStrategy:
        def __init__(self):
            self.trade_count = 0
            self.last_trade = None
            
        def next(self):
            """Called for each trade"""
            if hasattr(self, 'data') and self.data:
                self.trade_count += 1
                self.last_trade = self.data
                
                # Print trade info periodically
                if self.trade_count % 100 == 0:
                    print(f"Strategy processed {self.trade_count} trades")
    
    try:
        # Create runtime with test strategy
        runtime = HotSpineRuntime(TestStrategy)
        print("✓ HotSpineRuntime created successfully")
        
        # Test that runtime has expected attributes
        assert hasattr(runtime, 'reader'), "Runtime should have reader"
        assert hasattr(runtime, 'run'), "Runtime should have run method"
        print("✓ Runtime has expected attributes")
        
        # Test that reader is healthy
        assert runtime.reader.is_healthy(), "Runtime reader should be healthy"
        print("✓ Runtime reader is healthy")
        
        return True
        
    except Exception as e:
        print(f"✗ HotSpineRuntime test failed: {e}")
        return False


def test_performance():
    """Test performance characteristics"""
    print("\n=== Testing Performance Characteristics ===")
    
    try:
        reader = HotSpineReader("/btquant_hotspine")
        
        # Test polling speed
        start_time = time.time()
        poll_count = 0
        
        # Poll rapidly for 1 second to measure speed
        end_time = start_time + 1.0
        while time.time() < end_time:
            reader.poll_trade()
            poll_count += 1
        
        elapsed = time.time() - start_time
        polls_per_second = poll_count / elapsed if elapsed > 0 else 0
        
        print(f"✓ Polling speed: {polls_per_second:.0f} polls/second")
        
        # Test batch reading speed
        start_time = time.time()
        batch_count = 0
        
        # Read batches for 1 second
        end_time = start_time + 1.0
        while time.time() < end_time:
            reader.read_all_trades()
            batch_count += 1
        
        elapsed = time.time() - start_time
        batches_per_second = batch_count / elapsed if elapsed > 0 else 0
        
        print(f"✓ Batch reading speed: {batches_per_second:.0f} batches/second")
        
        reader.close()
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def test_error_handling():
    """Test error handling"""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with non-existent shared memory
        try:
            reader = HotSpineReader("/nonexistent_hotspine")
            print("✗ Should have failed with non-existent shared memory")
            reader.close()
            return False
        except Exception:
            print("✓ Correctly handles non-existent shared memory")
        
        # Test double close
        reader = HotSpineReader("/btquant_hotspine")
        reader.close()
        reader.close()  # Should not crash
        print("✓ Handles double close gracefully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("HotSpine Reader Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_hotspine_runtime,
        test_performance,
        test_error_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! HotSpine reader is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)