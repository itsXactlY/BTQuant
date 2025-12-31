"""
Logging Setup Module

Handles the configuration and setup of the logging framework.
"""

import logging
import logging.handlers
import json
from typing import Dict, Any
from pathlib import Path

class LoggingSetup:
    """Class for setting up and configuring logging"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_basic_logger()
        
    def _setup_basic_logger(self) -> logging.Logger:
        """
        Set up a basic logger for initial logging
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger('LoggingSetup')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def setup_logging(self) -> None:
        """
        Set up the complete logging framework based on configuration
        """
        try:
            # Create logs directory if it doesn't exist
            Path('logs').mkdir(parents=True, exist_ok=True)
            
            # Get logging configuration
            log_config = self.config.get('logging', {})
            level = log_config.get('level', 'INFO')
            log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            log_file = log_config.get('file', 'logs/system.log')
            console_logging = log_config.get('console', True)
            
            # Set logging level
            numeric_level = getattr(logging, level.upper(), logging.INFO)
            
            # Configure basic logging
            logging.basicConfig(
                level=numeric_level,
                format=log_format
            )
            
            # Set up file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # Set up console handler if enabled
            if console_logging:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(numeric_level)
                console_handler.setFormatter(logging.Formatter(log_format))
            
            # Get root logger and add handlers
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            if console_logging:
                root_logger.addHandler(console_handler)
            
            self.logger.info("Logging framework configured successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up logging: {e}", exc_info=True)
            raise
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger with the specified name
        
        Args:
            name: Name for the logger
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)