#!/usr/bin/env python3
"""
Data Source Connectors for Financial Models

This module provides connectors to various market data sources for the financial models.
It includes connectors for historical data, real-time data, and alternative data sources.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path


class MarketDataConnector:
    """
    Market Data Connector Base Class
    
    Base class for all market data connectors, providing common functionality
    and interface for data retrieval and processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the market data connector
        
        Args:
            config: Configuration dictionary for the connector
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
        # Default configuration
        self.default_config = {
            'data_source': 'generic',
            'cache_enabled': True,
            'cache_path': 'data/market_data_cache.json',
            'default_timeframe': '1D',
            'max_records': 10000
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize cache
        self._initialize_cache()
        
    def _initialize_cache(self) -> None:
        """Initialize the data cache"""
        os.makedirs(Path(self.config['cache_path']).parent, exist_ok=True)
        
        # Load existing cache if available
        self.cache = {}
        try:
            if os.path.exists(self.config['cache_path']):
                with open(self.config['cache_path'], 'r') as f:
                    self.cache = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            self.cache = {}
            
    def _save_cache(self) -> None:
        """Save the data cache to disk"""
        try:
            with open(self.config['cache_path'], 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D'
    ) -> pd.DataFrame:
        """
        Get historical market data for a symbol
        
        Args:
            symbol: Market symbol (e.g., 'AAPL', 'BTC/USD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe (e.g., '1D', '1H', '15M')
            
        Returns:
            Pandas DataFrame containing market data
        """
        # Generate cache key
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        
        # Check cache first
        if self.config['cache_enabled'] and cache_key in self.cache:
            self.logger.info(f"Retrieving {symbol} data from cache")
            cached_data = self.cache[cache_key]
            return pd.DataFrame(cached_data['data'], columns=cached_data['columns'])
        
        # If not in cache, fetch fresh data
        self.logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        # This will be implemented by specific connectors
        data = self._fetch_historical_data(symbol, start_date, end_date, timeframe)
        
        # Cache the data
        if self.config['cache_enabled']:
            self.cache[cache_key] = {
                'data': data.values.tolist(),
                'columns': list(data.columns),
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
        
        return data
        
    def _fetch_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D'
    ) -> pd.DataFrame:
        """
        Fetch historical data from the data source (to be implemented by subclasses)
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Pandas DataFrame containing market data
        """
        raise NotImplementedError("Subclasses must implement _fetch_historical_data")
        
    def get_real_time_data(self, symbol: str, timeframe: str = '1D') -> pd.DataFrame:
        """
        Get real-time market data for a symbol
        
        Args:
            symbol: Market symbol
            timeframe: Data timeframe
            
        Returns:
            Pandas DataFrame containing real-time market data
        """
        self.logger.info(f"Fetching real-time data for {symbol}")
        
        # This will be implemented by specific connectors
        return self._fetch_real_time_data(symbol, timeframe)
        
    def _fetch_real_time_data(self, symbol: str, timeframe: str = '1D') -> pd.DataFrame:
        """
        Fetch real-time data from the data source (to be implemented by subclasses)
        
        Args:
            symbol: Market symbol
            timeframe: Data timeframe
            
        Returns:
            Pandas DataFrame containing real-time market data
        """
        raise NotImplementedError("Subclasses must implement _fetch_real_time_data")
        
    def get_multiple_symbols(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols
        
        Args:
            symbols: List of market symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.get_historical_data(symbol, start_date, end_date, timeframe)
        return result
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data quality
        
        Args:
            data: Market data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check for minimum required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in data.columns:
                self.logger.error(f"Missing required column: {col}")
                return False
        
        # Check for sufficient data points
        if len(data) < 10:
            self.logger.error("Insufficient data points")
            return False
        
        # Check for missing values
        if data.isnull().values.any():
            self.logger.warning("Data contains missing values")
            # Fill missing values for basic validation
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
        
        return True
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for use with financial models
        
        Args:
            data: Raw market data
            
        Returns:
            Processed market data
        """
        self.logger.info("Preprocessing market data")
        
        # Make a copy to avoid modifying original
        processed_data = data.copy()
        
        # Convert index to datetime if not already
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            processed_data.index = pd.to_datetime(processed_data.index)
        
        # Sort by date
        processed_data.sort_index(inplace=True)
        
        # Calculate additional features
        processed_data['returns'] = processed_data['close'].pct_change()
        processed_data['log_returns'] = np.log(processed_data['close'] / processed_data['close'].shift(1))
        processed_data['volatility'] = processed_data['returns'].rolling(20).std()
        processed_data['momentum'] = processed_data['close'].pct_change(periods=5)
        processed_data['volume_change'] = processed_data['volume'].pct_change()
        
        # Handle missing values
        processed_data.fillna(0, inplace=True)
        
        return processed_data


class CSVDataConnector(MarketDataConnector):
    """
    CSV File Data Connector
    
    Connects to market data stored in CSV files for testing and development.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CSV data connector
        
        Args:
            config: Configuration dictionary for the connector
        """
        super().__init__(config)
        
        # Update default config for CSV connector
        self.default_config.update({
            'data_source': 'csv',
            'data_directory': 'data/market_data',
            'default_filename_pattern': '{symbol}_{timeframe}.csv'
        })
        
        self.config = {**self.default_config, **(config or {})}
        
        # Create data directory if it doesn't exist
        os.makedirs(self.config['data_directory'], exist_ok=True)
        
    def _fetch_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D'
    ) -> pd.DataFrame:
        """
        Fetch historical data from CSV files
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Pandas DataFrame containing market data
        """
        # Generate filename
        filename = self.config['default_filename_pattern'].format(
            symbol=symbol.replace('/', '_'),
            timeframe=timeframe
        )
        
        file_path = os.path.join(self.config['data_directory'], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            self.logger.warning(f"CSV file not found: {file_path}")
            # Generate synthetic data for testing
            return self._generate_synthetic_data(symbol, start_date, end_date, timeframe)
        
        try:
            # Read CSV file
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Filter by date range
            data = data.loc[start_date:end_date]
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"Missing column {col} in {filename}, generating synthetic data")
                    return self._generate_synthetic_data(symbol, start_date, end_date, timeframe)
            
            self.logger.info(f"Successfully loaded data from {filename}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to read CSV file {file_path}: {e}")
            # Fall back to synthetic data
            return self._generate_synthetic_data(symbol, start_date, end_date, timeframe)
        
    def _generate_synthetic_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D'
    ) -> pd.DataFrame:
        """
        Generate synthetic market data for testing
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Pandas DataFrame containing synthetic market data
        """
        self.logger.info(f"Generating synthetic data for {symbol} from {start_date} to {end_date}")
        
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate date range
        if timeframe == '1D':
            date_range = pd.date_range(start_dt, end_dt, freq='D')
        elif timeframe == '1H':
            date_range = pd.date_range(start_dt, end_dt, freq='H')
        elif timeframe == '15M':
            date_range = pd.date_range(start_dt, end_dt, freq='15T')
        else:
            date_range = pd.date_range(start_dt, end_dt, freq='D')
        
        # Generate synthetic price data
        n_points = len(date_range)
        
        # Base price (different for different symbols)
        if 'BTC' in symbol or 'ETH' in symbol:
            base_price = 50000
            volatility = 0.03
        elif 'AAPL' in symbol or 'MSFT' in symbol:
            base_price = 150
            volatility = 0.02
        elif 'SPY' in symbol or 'QQQ' in symbol:
            base_price = 400
            volatility = 0.015
        else:
            base_price = 100
            volatility = 0.025
        
        # Generate price series with trend and noise
        trend = np.linspace(0, np.random.uniform(-0.1, 0.1), n_points)
        noise = np.random.normal(0, volatility, n_points)
        prices = base_price * (1 + trend + noise.cumsum())
        
        # Ensure prices stay positive
        prices = np.maximum(prices, base_price * 0.8)
        
        # Generate OHLC data
        open_prices = prices.copy()
        close_prices = prices.copy()
        
        # Add some variation for high/low
        spread = np.abs(np.random.normal(0, volatility/2, n_points))
        high_prices = close_prices * (1 + spread)
        low_prices = close_prices * (1 - spread)
        
        # Generate volume data
        volumes = np.random.randint(1000, 100000, n_points)
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=date_range)
        
        return data
        
    def _fetch_real_time_data(self, symbol: str, timeframe: str = '1D') -> pd.DataFrame:
        """
        Fetch real-time data (simulated for CSV connector)
        
        Args:
            symbol: Market symbol
            timeframe: Data timeframe
            
        Returns:
            Pandas DataFrame containing simulated real-time market data
        """
        # For CSV connector, we'll generate recent synthetic data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        return self._generate_synthetic_data(symbol, start_date, end_date, timeframe)


class DataSourceManager:
    """
    Data Source Manager
    
    Manages multiple data connectors and provides a unified interface for data access.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data source manager
        
        Args:
            config: Configuration dictionary for the manager
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Data Source Manager")
        
        # Default configuration
        self.default_config = {
            'default_connector': 'csv',
            'connectors': {
                'csv': {
                    'class': CSVDataConnector,
                    'config': {}
                }
            }
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize connectors
        self.connectors = {}
        self._initialize_connectors()
        
    def _initialize_connectors(self) -> None:
        """Initialize all configured data connectors"""
        for name, connector_config in self.config['connectors'].items():
            try:
                connector_class = connector_config['class']
                connector_instance = connector_class(connector_config.get('config', {}))
                self.connectors[name] = connector_instance
                self.logger.info(f"Initialized connector: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize connector {name}: {e}")
        
    def get_connector(self, connector_name: str = None) -> MarketDataConnector:
        """
        Get a specific data connector
        
        Args:
            connector_name: Name of the connector to retrieve
            
        Returns:
            The requested data connector instance
        """
        name = connector_name or self.config['default_connector']
        
        if name not in self.connectors:
            raise ValueError(f"Connector {name} not available")
        
        return self.connectors[name]
        
    def get_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D',
        connector_name: str = None
    ) -> pd.DataFrame:
        """
        Get historical market data using the specified connector
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            connector_name: Name of the connector to use
            
        Returns:
            Pandas DataFrame containing market data
        """
        connector = self.get_connector(connector_name)
        return connector.get_historical_data(symbol, start_date, end_date, timeframe)
        
    def get_real_time_data(
        self, 
        symbol: str, 
        timeframe: str = '1D',
        connector_name: str = None
    ) -> pd.DataFrame:
        """
        Get real-time market data using the specified connector
        
        Args:
            symbol: Market symbol
            timeframe: Data timeframe
            connector_name: Name of the connector to use
            
        Returns:
            Pandas DataFrame containing real-time market data
        """
        connector = self.get_connector(connector_name)
        return connector.get_real_time_data(symbol, timeframe)
        
    def get_multiple_symbols(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D',
        connector_name: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols
        
        Args:
            symbols: List of market symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            connector_name: Name of the connector to use
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        connector = self.get_connector(connector_name)
        return connector.get_multiple_symbols(symbols, start_date, end_date, timeframe)
        
    def get_data_for_model(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str, 
        timeframe: str = '1D',
        connector_name: str = None
    ) -> pd.DataFrame:
        """
        Get and preprocess data specifically for financial models
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            connector_name: Name of the connector to use
            
        Returns:
            Preprocessed DataFrame ready for financial model consumption
        """
        connector = self.get_connector(connector_name)
        
        # Get raw data
        raw_data = connector.get_historical_data(symbol, start_date, end_date, timeframe)
        
        # Validate data
        if not connector.validate_data(raw_data):
            raise ValueError("Invalid market data received")
        
        # Preprocess data for financial models
        processed_data = connector.preprocess_data(raw_data)
        
        return processed_data