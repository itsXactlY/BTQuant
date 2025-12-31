"""
Configuration Loader Module

Handles loading and managing system configuration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Class for loading and managing system configuration"""
    
    def __init__(self, config_path: str = "config/system_config.json"):
        self.logger = logging.getLogger('ConfigLoader')
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Returns:
            Dictionary containing the system configuration
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in configuration file: {self.config_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration
        
        Returns:
            Dictionary containing the system configuration
        """
        return self.config
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a specific section from the configuration
        
        Args:
            section_name: Name of the configuration section
            
        Returns:
            Dictionary containing the requested section
        """
        if section_name in self.config:
            return self.config[section_name]
        else:
            self.logger.warning(f"Configuration section not found: {section_name}")
            return {}
    
    def validate_config(self) -> bool:
        """
        Validate the loaded configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_sections = ['system', 'logging', 'strategy_generation', 'backtesting', 
                           'evolutionary_selection', 'documentation', 'deployment']
        
        for section in required_sections:
            if section not in self.config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        self.logger.info("Configuration validation passed")
        return True