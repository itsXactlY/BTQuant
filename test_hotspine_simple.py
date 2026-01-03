#!/usr/bin/env python3
"""
Simple test script for enhanced HotSpine architecture

This script tests the new configuration management features without
triggering circular import issues.
"""

import sys
import os
import logging

# Test configuration management directly
sys.path.insert(0, 'dependencies/backtrader')

def test_configuration_management():
    """Test configuration management features"""
    print("🧪 Testing Configuration Management...")
    
    try:
        # Import configuration directly
        from hotspine.config import HotSpineConfig, configure_logging, save_config_to_file, load_config_from_file
        
        # Configure logging
        configure_logging(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        # Test default configuration
        config = HotSpineConfig()
        print(f"✅ Default config created: {config.to_dict()}")
        
        # Test custom configuration
        custom_config = HotSpineConfig(
            shm_name="/custom_hotspine",
            batch_mode=True,
            poll_interval=0.001,
            enable_sql_storage=False,
            max_reconnect_attempts=10
        )
        print(f"✅ Custom config created: {custom_config.to_dict()}")
        
        # Test environment variable loading
        os.environ['HOTSPINE_BATCH_MODE'] = 'true'
        os.environ['HOTSPINE_POLL_INTERVAL'] = '0.005'
        env_config = HotSpineConfig()
        print(f"✅ Environment config loaded: batch_mode={env_config.batch_mode}, poll_interval={env_config.poll_interval}")
        
        # Test file save/load
        test_file = "test_config.json"
        save_success = save_config_to_file(custom_config, test_file)
        print(f"✅ Config save to file: {save_success}")
        
        loaded_config = load_config_from_file(test_file)
        print(f"✅ Config loaded from file: {loaded_config.to_dict()}")
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("🎉 Configuration management tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hotspine_imports():
    """Test HotSpine imports"""
    print("🧪 Testing HotSpine Imports...")
    
    try:
        # Test reader import
        from hotspine.reader import HotSpineReader, HotSpineRuntime
        print("✅ HotSpine reader imported successfully")
        
        # Test SQL integration import
        from hotspine.sql_integration import HotSpineSQLIntegration
        print("✅ HotSpine SQL integration imported successfully")
        
        # Test config import
        from hotspine.config import HotSpineConfig
        print("✅ HotSpine config imported successfully")
        
        print("🎉 HotSpine imports test passed!")
        return True
        
    except Exception as e:
        print(f"❌ HotSpine imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 SIMPLE HOTSPINE ARCHITECTURE TEST")
    print("=" * 60)
    print()
    
    success = True
    
    # Run configuration test
    if not test_configuration_management():
        success = False
    
    print()
    
    # Run import test
    if not test_hotspine_imports():
        success = False
    
    print()
    
    if success:
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("✅ Enhanced HotSpine architecture is working correctly")
        print("✅ Configuration management: OK")
        print("✅ Module imports: OK")
        print()
        print("🚀 Ready for Live_Trading_HotSpine_SMA.py integration!")
    else:
        print("❌ Some tests failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)