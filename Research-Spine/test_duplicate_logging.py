#!/usr/bin/env python3
"""
Test script to reproduce the duplicate logging issue
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from utils.logging_setup import LoggingSetup

def test_duplicate_logging():
    """Test to reproduce duplicate logging"""
    print("=== Testing Duplicate Logging Issue ===\n")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    # Set up logging
    logging_setup = LoggingSetup(config)
    logging_setup.setup_logging()
    
    # Get logger
    logger = logging_setup.get_logger('test')
    
    print("\n=== Checking Root Logger Handlers ===")
    root_logger = logging.getLogger()
    print(f"Root logger level: {root_logger.level}")
    print(f"Number of handlers: {len(root_logger.handlers)}")
    
    for i, handler in enumerate(root_logger.handlers):
        print(f"Handler {i}: {type(handler).__name__} - {handler}")
    
    print("\n=== Testing Log Messages ===")
    logger.info("Test message 1")
    logger.info("Test message 2")
    logger.info("Test message 3")
    
    print("\n=== Analysis ===")
    console_handlers = []
    file_handlers = []
    for h in root_logger.handlers:
        is_file = isinstance(h, logging.FileHandler)
        is_stream = isinstance(h, logging.StreamHandler) and not is_file
        print(f"  Handler: {type(h).__name__} - File: {is_file}, Pure Stream: {is_stream}")
        if is_file:
            file_handlers.append(h)
        elif is_stream:
            console_handlers.append(h)
    
    print(f"Number of StreamHandlers: {len(console_handlers)}")
    print(f"Number of FileHandlers: {len(file_handlers)}")
    print(f"Total handlers: {len(root_logger.handlers)}")
    
    # Check for duplicates
    if len(console_handlers) > 1:
        print("❌ ISSUE: Multiple StreamHandlers detected!")
        print("This will cause duplicate log messages.")
    elif len(console_handlers) == 1 and len(file_handlers) == 1:
        print("✅ SUCCESS: Correct handler configuration!")
        print("Expected: 1 StreamHandler + 1 FileHandler")
    else:
        print("⚠️  Unexpected handler configuration")

if __name__ == "__main__":
    test_duplicate_logging()