#!/usr/bin/env python3
"""
Comprehensive test to understand the logging setup
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from utils.logging_setup import LoggingSetup

def print_handler_info(logger, name):
    """Print handler information for a logger"""
    print(f"\n{name} Logger:")
    print(f"  Level: {logging.getLevelName(logger.level)}")
    print(f"  Handlers: {len(logger.handlers)}")
    for i, h in enumerate(logger.handlers):
        print(f"    {i}: {type(h).__name__} - {h}")
        if hasattr(h, 'level'):
            print(f"       Level: {logging.getLevelName(h.level)}")

def test_logging_setup():
    """Test the complete logging setup"""
    print("=== Comprehensive Logging Test ===\n")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    print("1. Before setup:")
    root_logger = logging.getLogger()
    print_handler_info(root_logger, "Root")
    
    # Set up logging
    logging_setup = LoggingSetup(config)
    
    print("\n2. After LoggingSetup.__init__:")
    print_handler_info(root_logger, "Root")
    print_handler_info(logging.getLogger('LoggingSetup'), "LoggingSetup")
    
    logging_setup.setup_logging()
    
    print("\n3. After setup_logging():")
    print_handler_info(root_logger, "Root")
    print_handler_info(logging.getLogger('LoggingSetup'), "LoggingSetup")
    
    # Test with a new logger
    test_logger = logging_setup.get_logger('test')
    print("\n4. Testing with new logger:")
    print_handler_info(test_logger, "Test")
    
    print("\n=== Testing Log Messages ===")
    test_logger.info("Test message 1")
    test_logger.info("Test message 2")
    
    print("\n=== Final Analysis ===")
    root_handlers = root_logger.handlers
    stream_handlers = []
    file_handlers = []
    
    for h in root_handlers:
        is_stream = isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        is_file = isinstance(h, logging.FileHandler)
        print(f"Handler: {type(h).__name__} - Pure Stream: {is_stream}, File: {is_file}")
        if is_stream:
            stream_handlers.append(h)
        if is_file:
            file_handlers.append(h)
    
    print(f"Root logger total handlers: {len(root_handlers)}")
    print(f"StreamHandlers: {len(stream_handlers)}")
    print(f"FileHandlers: {len(file_handlers)}")
    
    # Check propagation
    print(f"\nPropagation settings:")
    print(f"  Root logger propagate: {root_logger.propagate}")
    print(f"  LoggingSetup logger propagate: {logging.getLogger('LoggingSetup').propagate}")
    print(f"  Test logger propagate: {test_logger.propagate}")
    
    if len(stream_handlers) == 1 and len(file_handlers) == 1:
        print("✅ SUCCESS: Correct configuration!")
    else:
        print("❌ ISSUE: Incorrect handler configuration!")

if __name__ == "__main__":
    test_logging_setup()