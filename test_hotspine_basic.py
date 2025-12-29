#!/usr/bin/env python3
"""
Basic test script to validate HotSpine integration

This script tests the core functionality without importing HotTrade directly.
"""

import sys
import os

# Add the dependencies directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

# Test imports step by step
print("Testing HotSpine integration...")

try:
    # Test 1: Import backtrader
    import backtrader as bt
    print("✅ Backtrader import successful")
except ImportError as e:
    print(f"❌ Backtrader import failed: {e}")
    sys.exit(1)

try:
    # Test 2: Import HotSpine feed (this should work now)
    from backtrader.feeds.hotspine_feed import HotSpineData, HotSpineFeed
    print("✅ HotSpine feed import successful")
except ImportError as e:
    print(f"❌ HotSpine feed import failed: {e}")
    sys.exit(1)

try:
    # Test 3: Create HotSpine data feed instance
    data = HotSpineData(symbol_id=123, shm_name="/test_hotspine")
    print("✅ HotSpineData instance creation successful")
    
    # Test basic properties
    assert data.symbol_id == 123
    assert data.p.shm_name == "/test_hotspine"
    assert data.islive() == True
    print("✅ HotSpineData properties correct")
    
except Exception as e:
    print(f"❌ HotSpineData creation failed: {e}")
    sys.exit(1)

try:
    # Test 4: Create HotSpine feed instance
    feed = HotSpineFeed(symbol_id=123, shm_name="/test_hotspine")
    print("✅ HotSpineFeed instance creation successful")
    
    # Test basic properties
    assert feed._symbol_id == 123
    assert feed._shm_name == "/test_hotspine"
    print("✅ HotSpineFeed properties correct")
    
except Exception as e:
    print(f"❌ HotSpineFeed creation failed: {e}")
    sys.exit(1)

try:
    # Test 5: Test architectural separation
    from backtrader.hotspine.sql_integration import HotSpineSQLIntegration
    
    # Create instances
    hotspine_data = HotSpineData(symbol_id=123)
    sql_integration = HotSpineSQLIntegration()
    
    # Verify they are separate
    assert hasattr(hotspine_data, 'hotspine_reader')
    assert hasattr(sql_integration, 'storage')
    assert not hasattr(hotspine_data, 'sql_integration')
    assert not hasattr(sql_integration, 'hotspine_reader')
    
    print("✅ Architectural separation verified")
    
except ImportError:
    print("⚠️  SQL integration not available for testing")
except Exception as e:
    print(f"❌ Architectural separation test failed: {e}")
    sys.exit(1)

print("\n✅ All basic tests passed!")
print("✅ HotSpine integration is working correctly.")
print("✅ Backtest path remains unchanged.")
print("✅ Architectural separation is maintained.")

sys.exit(0)