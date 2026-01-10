#!/usr/bin/env python3
"""
HotSpine Configuration Management Module

This module provides centralized configuration management for all HotSpine components,
including parameter validation, environment variable support, and logging configuration.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json


@dataclass
class HotSpineConfig:
    """
    Centralized configuration for HotSpine components
    
    This class provides a standardized way to manage configuration across
    all HotSpine modules with validation and environment variable support.
    """
    
    # Shared memory configuration
    shm_name: str = "/btquant_hotspine"
    shm_capacity: int = 1000000
    
    # Reader configuration
    poll_interval: float = 0.0001  # 100 microseconds
    batch_mode: bool = False
    max_batch_size: int = 1000
    
    # SQL integration configuration
    enable_sql_storage: bool = True
    sql_batch_size: int = 100
    sql_queue_size: int = 10000
    
    # Performance monitoring
    enable_monitoring: bool = True
    metrics_interval: float = 1.0
    
    # Error handling
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    
    # Symbol management
    symbol_mapping: Optional[Dict[int, Dict[str, str]]] = None  # symbol_id -> {symbol, market_type}
    
    # NEW: Market type filtering
    market_type_filter: str = "all"  # "spot", "futures", or "all"
    
    # NEW: Symbol whitelisting
    symbol_whitelist: Optional[List[str]] = None  # List of allowed symbol IDs
    
    def __post_init__(self):
        """Initialize configuration with environment variables and validation"""
        self._load_from_environment()
        self._validate()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Shared memory configuration
        if 'HOTSPINE_SHM_NAME' in os.environ:
            self.shm_name = os.environ['HOTSPINE_SHM_NAME']
        if 'HOTSPINE_SHM_CAPACITY' in os.environ:
            self.shm_capacity = int(os.environ['HOTSPINE_SHM_CAPACITY'])
        
        # Reader configuration
        if 'HOTSPINE_POLL_INTERVAL' in os.environ:
            self.poll_interval = float(os.environ['HOTSPINE_POLL_INTERVAL'])
        if 'HOTSPINE_BATCH_MODE' in os.environ:
            self.batch_mode = os.environ['HOTSPINE_BATCH_MODE'].lower() in ('true', '1', 'yes')
        if 'HOTSPINE_MAX_BATCH_SIZE' in os.environ:
            self.max_batch_size = int(os.environ['HOTSPINE_MAX_BATCH_SIZE'])
        
        # SQL configuration
        if 'HOTSPINE_ENABLE_SQL_STORAGE' in os.environ:
            self.enable_sql_storage = os.environ['HOTSPINE_ENABLE_SQL_STORAGE'].lower() in ('true', '1', 'yes')
        if 'HOTSPINE_SQL_BATCH_SIZE' in os.environ:
            self.sql_batch_size = int(os.environ['HOTSPINE_SQL_BATCH_SIZE'])
        if 'HOTSPINE_SQL_QUEUE_SIZE' in os.environ:
            self.sql_queue_size = int(os.environ['HOTSPINE_SQL_QUEUE_SIZE'])
        
        # Monitoring configuration
        if 'HOTSPINE_ENABLE_MONITORING' in os.environ:
            self.enable_monitoring = os.environ['HOTSPINE_ENABLE_MONITORING'].lower() in ('true', '1', 'yes')
        if 'HOTSPINE_METRICS_INTERVAL' in os.environ:
            self.metrics_interval = float(os.environ['HOTSPINE_METRICS_INTERVAL'])
        
        # Error handling
        if 'HOTSPINE_MAX_RECONNECT_ATTEMPTS' in os.environ:
            self.max_reconnect_attempts = int(os.environ['HOTSPINE_MAX_RECONNECT_ATTEMPTS'])
        if 'HOTSPINE_RECONNECT_DELAY' in os.environ:
            self.reconnect_delay = float(os.environ['HOTSPINE_RECONNECT_DELAY'])
        
        # Symbol mapping
        if 'HOTSPINE_SYMBOL_MAPPING' in os.environ:
            try:
                self.symbol_mapping = json.loads(os.environ['HOTSPINE_SYMBOL_MAPPING'])
            except json.JSONDecodeError:
                logging.warning("Invalid HOTSPINE_SYMBOL_MAPPING JSON format")
        
        # NEW: Market type filtering
        if 'HOTSPINE_MARKET_TYPE_FILTER' in os.environ:
            market_type = os.environ['HOTSPINE_MARKET_TYPE_FILTER'].lower()
            if market_type in ['spot', 'futures', 'all']:
                self.market_type_filter = market_type
            else:
                logging.warning(f"Invalid HOTSPINE_MARKET_TYPE_FILTER value: {market_type}")
        
        # NEW: Symbol whitelisting
        if 'HOTSPINE_SYMBOL_WHITELIST' in os.environ:
            try:
                self.symbol_whitelist = json.loads(os.environ['HOTSPINE_SYMBOL_WHITELIST'])
                if not isinstance(self.symbol_whitelist, list):
                    logging.warning("HOTSPINE_SYMBOL_WHITELIST must be a JSON array")
                    self.symbol_whitelist = None
            except json.JSONDecodeError:
                logging.warning("Invalid HOTSPINE_SYMBOL_WHITELIST JSON format")
    
    def _validate(self):
        """Validate configuration parameters"""
        # Validate shared memory configuration
        if not isinstance(self.shm_name, str) or not self.shm_name:
            raise ValueError("shm_name must be a non-empty string")
        if self.shm_capacity <= 0:
            raise ValueError("shm_capacity must be positive")
        
        # Validate reader configuration
        if self.poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        
        # Validate SQL configuration
        if self.sql_batch_size <= 0:
            raise ValueError("sql_batch_size must be positive")
        if self.sql_queue_size <= 0:
            raise ValueError("sql_queue_size must be positive")
        
        # Validate error handling
        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts must be non-negative")
        if self.reconnect_delay < 0:
            raise ValueError("reconnect_delay must be non-negative")
        
        # Validate monitoring
        if self.metrics_interval < 0.1:
            raise ValueError("metrics_interval must be at least 0.1 seconds")
        
        # NEW: Validate market type filtering
        if self.market_type_filter not in ['spot', 'futures', 'all']:
            raise ValueError("market_type_filter must be 'spot', 'futures', or 'all'")
        
        # NEW: Validate symbol whitelisting
        if self.symbol_whitelist is not None and not isinstance(self.symbol_whitelist, list):
            raise ValueError("symbol_whitelist must be a list or None")
        
        if self.symbol_whitelist:
            for symbol in self.symbol_whitelist:
                if not isinstance(symbol, str):
                    raise ValueError("symbol_whitelist items must be strings")
    
    def to_dict(self) -> Dict[str, Any]:
            """Convert configuration to dictionary"""
            return {
                'shm_name': self.shm_name,
                'shm_capacity': self.shm_capacity,
                'poll_interval': self.poll_interval,
                'batch_mode': self.batch_mode,
                'max_batch_size': self.max_batch_size,
                'enable_sql_storage': self.enable_sql_storage,
                'sql_batch_size': self.sql_batch_size,
                'sql_queue_size': self.sql_queue_size,
                'enable_monitoring': self.enable_monitoring,
                'metrics_interval': self.metrics_interval,
                'max_reconnect_attempts': self.max_reconnect_attempts,
                'reconnect_delay': self.reconnect_delay,
                'symbol_mapping': self.symbol_mapping,
                'market_type_filter': self.market_type_filter,
                'symbol_whitelist': self.symbol_whitelist
            }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary"""
        return cls(**config_dict)


def configure_logging(level: int = logging.INFO, 
                     format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     log_file: Optional[str] = None):
    """
    Configure logging for HotSpine components
    
    Args:
        level: Logging level (default: INFO)
        format_str: Log format string
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)


def get_default_config() -> HotSpineConfig:
    """Get default HotSpine configuration"""
    return HotSpineConfig()


def load_config_from_file(file_path: str) -> HotSpineConfig:
    """Load configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return HotSpineConfig.from_dict(config_dict)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to load config from {file_path}: {e}")
        return get_default_config()


def save_config_to_file(config: HotSpineConfig, file_path: str) -> bool:
    """Save configuration to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        return True
    except IOError as e:
        logging.error(f"Failed to save config to {file_path}: {e}")
        return False