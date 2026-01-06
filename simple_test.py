#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python_market_data_collector.hotspine_reader import HotSpineReader

def test_basic_functionality():
    """Test basic HotSpine reader functionality"""
    print("Testing basic HotSpine reader functionality...")
    
    try:
        # Test 1: Create reader
        print("1. Creating HotSpineReader...")
        reader = HotSpineReader()
        print("   ✓ Reader created successfully")
        
        # Test 2: Get buffer utilization
        print("2. Getting buffer utilization...")
        buffer_util = reader.get_buffer_utilization()
        print(f"   ✓ Trade buffer: {buffer_util['trade_count']}/{buffer_util['trade_capacity']}")
        print(f"   ✓ Orderbook buffer: {buffer_util['orderbook_count']}/{buffer_util['orderbook_capacity']}")
        
        # Test 3: Get statistics
        print("3. Getting statistics...")
        stats = reader.get_statistics()
        print(f"   ✓ Statistics retrieved: {len(stats)} metrics")
        
        # Test 4: Get health status
        print("4. Getting health status...")
        health = reader.get_health_status()
        print(f"   ✓ Health status: attached={health['attached']}, healthy={health['healthy']}")
        
        # Test 5: Read trades (non-blocking)
        print("5. Testing trade reading...")
        trades = reader.read_all_trades()
        print(f"   ✓ Trades read: {len(trades)}")
        
        # Test 6: Read orderbooks (non-blocking)
        print("6. Testing orderbook reading...")
        orderbooks = reader.read_all_orderbooks()
        print(f"   ✓ Orderbooks read: {len(orderbooks)}")
        
        # Clean up
        reader.close()
        print("7. Reader closed successfully")
        
        print("\n✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)