"""
Trading Configuration Module

This module provides structured configuration management for trading strategies,
ensuring separation of concerns and modular design.
"""

from typing import Dict, Any, Optional
import json
import os


class TradingConfig:
    """
    Comprehensive trading configuration with validation and persistence
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize trading configuration
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        # Default configuration
        self._defaults = {
            # Trading parameters
            'coin': 'BTC',
            'collateral': 'USDT',
            'exchange': 'Binance',
            'amount': 11.0,
            'asset': None,  # Added asset parameter
             
            # HotSpine specific parameters
            'symbol_id': 123,
            'shm_name': '/btquant_hotspine',
            'batch_mode': False,
            'poll_interval': 0.0001,
             
            # Strategy parameters
            'enable_alerts': False,
            'alert_channel': -100,
            'backtest': False,
            'enable_ml_filtering': True,
            'volatility_threshold': 0.05,
            'trend_strength_threshold': 0.3,
            'regime_detection': True,
            'adaptive_position_sizing': True
        }
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
        else:
            self._config = self._defaults.copy()
        
        # Validate configuration
        self._validate()
    
    def _load_from_file(self, config_file: str):
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Merge with defaults
            self._config = {**self._defaults, **file_config}
            
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
            print("Using default configuration")
            self._config = self._defaults.copy()
    
    def _validate(self):
        """Validate configuration parameters"""
        errors = []
        
        # Validate symbol_id
        if not isinstance(self._config['symbol_id'], int) or self._config['symbol_id'] <= 0:
            errors.append("symbol_id must be a positive integer")
        
        # Validate amount
        if not isinstance(self._config['amount'], (int, float)) or self._config['amount'] <= 0:
            errors.append("amount must be a positive number")
        
        # Validate poll_interval
        if not isinstance(self._config['poll_interval'], (int, float)) or self._config['poll_interval'] <= 0:
            errors.append("poll_interval must be a positive number")
        
        # Validate thresholds
        if not isinstance(self._config['volatility_threshold'], (int, float)) or self._config['volatility_threshold'] <= 0:
            errors.append("volatility_threshold must be a positive number")
        
        if not isinstance(self._config['trend_strength_threshold'], (int, float)) or self._config['trend_strength_threshold'] <= 0:
            errors.append("trend_strength_threshold must be a positive number")
        
        if errors:
            raise ValueError("Configuration validation failed: " + "; ".join(errors))
    
    def save_to_file(self, config_file: str):
        """
        Save configuration to JSON file
        
        Args:
            config_file: Path to save configuration
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def __getattr__(self, name: str) -> Any:
        """
        Get configuration value with attribute-style access
        
        Args:
            name: Configuration parameter name
            
        Returns:
            Configuration value
        """
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any):
        """
        Set configuration value with attribute-style access
        
        Args:
            name: Configuration parameter name
            value: Value to set
        """
        if name == '_config' or name == '_defaults':
            super().__setattr__(name, value)
        elif hasattr(self, name):
            self._config[name] = value
            # Validate after setting the value
            self._validate()
        else:
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        Update configuration from dictionary
        
        Args:
            config_dict: Dictionary of configuration values
        """
        self._config.update(config_dict)
        self._validate()


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary
    
    Returns:
        Default configuration dictionary
    """
    return TradingConfig().get_config_dict()