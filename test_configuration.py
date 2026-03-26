#!/usr/bin/env python3
"""
Test script to verify the new configuration options for HotSpineRuntime
"""

import sys
import os

# Add the dependencies path to sys.path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dependencies'))

from backtrader.hotspine.reader import HotSpineRuntime, create_hotspine_runtime
from backtrader.bigbraincentral.storage_mssql import MSSQLConfig


class DummyStrategy:
    """Dummy strategy for testing"""
    def __init__(self):
        self.data = None
        self.datas = []
        self.broker = None
        self.position = 0
    
    def next(self):
        """Dummy next method"""
        pass


def test_valid_configuration():
    """Test valid configuration combinations"""
    print("Testing valid configurations...")
    
    # Test 1: SQL storage enabled, no hotswap
    try:
        runtime = HotSpineRuntime(
            DummyStrategy,
            enable_sql_storage=True,
            exclusive_hotswap_mode=False
        )
        print("✓ Test 1 passed: SQL storage enabled, no hotswap")
        runtime.reader.close()
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test 2: Both SQL storage and hotswap enabled
    try:
        runtime = HotSpineRuntime(
            DummyStrategy,
            enable_sql_storage=True,
            exclusive_hotswap_mode=True
        )
        print("✓ Test 2 passed: Both SQL storage and hotswap enabled")
        runtime.reader.close()
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")


def test_invalid_configuration():
    """Test invalid configuration combinations"""
    print("\nTesting invalid configurations...")
    
    # Test 3: Hotswap enabled without SQL storage (should fail)
    try:
        runtime = HotSpineRuntime(
            DummyStrategy,
            enable_sql_storage=False,
            exclusive_hotswap_mode=True
        )
        print("✗ Test 3 failed: Should have raised ValueError")
        runtime.reader.close()
    except ValueError as e:
        print(f"✓ Test 3 passed: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ Test 3 failed with unexpected error: {e}")


def test_factory_function():
    """Test the factory function with new parameters"""
    print("\nTesting factory function...")
    
    try:
        runtime = create_hotspine_runtime(
            DummyStrategy,
            enable_sql_storage=True,
            exclusive_hotswap_mode=True
        )
        print("✓ Factory function test passed")
        runtime.reader.close()
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")


def test_configuration_attributes():
    """Test that configuration attributes are properly set"""
    print("\nTesting configuration attributes...")
    
    try:
        runtime = HotSpineRuntime(
            DummyStrategy,
            enable_sql_storage=True,
            exclusive_hotswap_mode=True
        )
        
        assert hasattr(runtime, 'enable_sql_storage'), "Missing enable_sql_storage attribute"
        assert hasattr(runtime, 'exclusive_hotswap_mode'), "Missing exclusive_hotswap_mode attribute"
        assert runtime.enable_sql_storage == True, "enable_sql_storage not set correctly"
        assert runtime.exclusive_hotswap_mode == True, "exclusive_hotswap_mode not set correctly"
        
        print("✓ Configuration attributes test passed")
        runtime.reader.close()
    except Exception as e:
        print(f"✗ Configuration attributes test failed: {e}")


if __name__ == "__main__":
    print("Running HotSpineRuntime configuration tests...\n")
    
    test_valid_configuration()
    test_invalid_configuration()
    test_factory_function()
    test_configuration_attributes()
    
    print("\nAll tests completed!")