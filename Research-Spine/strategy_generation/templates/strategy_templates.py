"""
Strategy Template System

Defines parameterized trading strategy templates for the strategy generation engine.
"""

import logging
from typing import Dict, Any, List
import json
import os

class StrategyTemplateManager:
    """Manages strategy templates and their parameter definitions"""
    
    def __init__(self):
        self.logger = logging.getLogger('StrategyTemplateManager')
        self.logger.info("StrategyTemplateManager initialized")
        self.templates = {}
        self.load_default_templates()
        
    def load_default_templates(self):
        """Load default strategy templates"""
        self.logger.info("Loading default strategy templates")
        
        # Moving Average Crossover Template
        self.templates['moving_average_crossover'] = {
            'name': 'moving_average_crossover',
            'description': 'Strategy based on moving average crossovers',
            'parameters': {
                'fast_period': {
                    'type': 'int',
                    'range': [5, 50],
                    'default': 10,
                    'description': 'Period for fast moving average'
                },
                'slow_period': {
                    'type': 'int',
                    'range': [20, 200],
                    'default': 50,
                    'description': 'Period for slow moving average'
                },
                'ma_type': {
                    'type': 'str',
                    'options': ['SMA', 'EMA', 'WMA'],
                    'default': 'SMA',
                    'description': 'Type of moving average to use'
                },
                'stop_loss_pct': {
                    'type': 'float',
                    'range': [0.01, 0.10],
                    'default': 0.05,
                    'description': 'Stop loss percentage'
                },
                'take_profit_pct': {
                    'type': 'float',
                    'range': [0.05, 0.30],
                    'default': 0.15,
                    'description': 'Take profit percentage'
                }
            },
            'constraints': [
                {'type': 'greater_than', 'params': ['slow_period', 'fast_period'], 'message': 'Slow period must be greater than fast period'}
            ]
        }
        
        # RSI Mean Reversion Template
        self.templates['rsi_mean_reversion'] = {
            'name': 'rsi_mean_reversion',
            'description': 'Mean reversion strategy using RSI indicator',
            'parameters': {
                'rsi_period': {
                    'type': 'int',
                    'range': [5, 30],
                    'default': 14,
                    'description': 'Period for RSI calculation'
                },
                'overbought_threshold': {
                    'type': 'int',
                    'range': [60, 80],
                    'default': 70,
                    'description': 'RSI level considered overbought'
                },
                'oversold_threshold': {
                    'type': 'int',
                    'range': [20, 40],
                    'default': 30,
                    'description': 'RSI level considered oversold'
                },
                'position_size_pct': {
                    'type': 'float',
                    'range': [0.01, 0.50],
                    'default': 0.10,
                    'description': 'Percentage of capital to risk per trade'
                },
                'max_holding_period': {
                    'type': 'int',
                    'range': [1, 20],
                    'default': 5,
                    'description': 'Maximum holding period in days'
                }
            },
            'constraints': [
                {'type': 'greater_than', 'params': ['overbought_threshold', 'oversold_threshold'], 'message': 'Overbought threshold must be greater than oversold threshold'}
            ]
        }
        
        # Bollinger Bands Breakout Template
        self.templates['bollinger_bands_breakout'] = {
            'name': 'bollinger_bands_breakout',
            'description': 'Breakout strategy using Bollinger Bands',
            'parameters': {
                'bb_period': {
                    'type': 'int',
                    'range': [10, 50],
                    'default': 20,
                    'description': 'Period for Bollinger Bands calculation'
                },
                'bb_std_dev': {
                    'type': 'float',
                    'range': [1.0, 3.0],
                    'default': 2.0,
                    'description': 'Number of standard deviations for bands'
                },
                'atr_period': {
                    'type': 'int',
                    'range': [5, 30],
                    'default': 14,
                    'description': 'Period for ATR calculation'
                },
                'atr_multiplier': {
                    'type': 'float',
                    'range': [0.5, 3.0],
                    'default': 1.5,
                    'description': 'ATR multiplier for stop loss'
                },
                'breakout_confirmation': {
                    'type': 'bool',
                    'default': True,
                    'description': 'Whether to require breakout confirmation'
                }
            }
        }
        
        self.logger.info(f"Loaded {len(self.templates)} default strategy templates")
        
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get a strategy template by name
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Strategy template dictionary
            
        Raises:
            ValueError: If template doesn't exist
        """
        if template_name not in self.templates:
            self.logger.error(f"Template {template_name} not found")
            raise ValueError(f"Template {template_name} not found")
            
        self.logger.debug(f"Retrieving template: {template_name}")
        return self.templates[template_name]
        
    def get_all_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available strategy templates
        
        Returns:
            Dictionary of all strategy templates
        """
        self.logger.debug("Retrieving all strategy templates")
        return self.templates
        
    def validate_template_parameters(self, template_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters against a template's constraints
        
        Args:
            template_name: Name of the template
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        template = self.get_template(template_name)
        
        # Check required parameters
        for param_name, param_config in template['parameters'].items():
            if param_name not in parameters:
                self.logger.error(f"Missing required parameter: {param_name}")
                return False
            
            # Check parameter types
            expected_type = param_config['type']
            actual_value = parameters[param_name]
            
            if expected_type == 'int' and not isinstance(actual_value, int):
                self.logger.error(f"Parameter {param_name} should be int, got {type(actual_value)}")
                return False
            elif expected_type == 'float' and not isinstance(actual_value, (int, float)):
                self.logger.error(f"Parameter {param_name} should be float, got {type(actual_value)}")
                return False
            elif expected_type == 'bool' and not isinstance(actual_value, bool):
                self.logger.error(f"Parameter {param_name} should be bool, got {type(actual_value)}")
                return False
            elif expected_type == 'str' and not isinstance(actual_value, str):
                self.logger.error(f"Parameter {param_name} should be str, got {type(actual_value)}")
                return False
        
        # Check template constraints
        if 'constraints' in template:
            for constraint in template['constraints']:
                if constraint['type'] == 'greater_than':
                    param1, param2 = constraint['params']
                    if parameters[param1] <= parameters[param2]:
                        self.logger.error(constraint['message'])
                        return False
        
        self.logger.info(f"Parameters validated successfully for template: {template_name}")
        return True
        
    def save_template(self, template_name: str, template_data: Dict[str, Any]) -> None:
        """
        Save a custom strategy template
        
        Args:
            template_name: Name of the template
            template_data: Template data to save
        """
        self.templates[template_name] = template_data
        self.logger.info(f"Saved custom template: {template_name}")
        
    def load_template_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a strategy template from a JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded template dictionary
        """
        try:
            with open(file_path, 'r') as f:
                template = json.load(f)
            
            # Validate template structure
            if 'name' not in template or 'parameters' not in template:
                raise ValueError("Invalid template structure")
            
            self.templates[template['name']] = template
            self.logger.info(f"Loaded template from file: {file_path}")
            return template
            
        except Exception as e:
            self.logger.error(f"Failed to load template from {file_path}: {str(e)}")
            raise