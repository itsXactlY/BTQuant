"""
HotSpine Store for Backtrader

This module provides a store abstraction layer for HotSpine data feeds,
mirroring the architecture of BinanceStore for consistency and modularity.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from backtrader.dataseries import TimeFrame
from backtrader.feeds.hotspine_feed import HotSpineData

# Import BTQuant configuration management
from backtrader.config.trading_config import TradingConfig

# Configure logging using standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HotSpineStore:
    """
    HotSpine Store for Backtrader
    
    This store provides a unified interface for accessing HotSpine data feeds,
    following the same architectural pattern as BinanceStore for consistency.
    """
    
    _GRANULARITIES = {
        (TimeFrame.Ticks, 1): 'ticks',
        (TimeFrame.Seconds, 1): '1s',
        (TimeFrame.Minutes, 1): '1m',
        (TimeFrame.Minutes, 5): '5m',
        (TimeFrame.Minutes, 15): '15m',
        (TimeFrame.Minutes, 30): '30m',
        (TimeFrame.Minutes, 60): '1h',
        (TimeFrame.Days, 1): '1d',
    }
    
    def __init__(self, symbol_id: int, shm_name: str = "/btquant_hotspine", 
                 batch_mode: bool = False, poll_interval: float = 0.0001):
        """
        Initialize HotSpine Store
        
        Args:
            symbol_id: Symbol ID to filter trades
            shm_name: Shared memory segment name
            batch_mode: Whether to use batch reading
            poll_interval: Polling interval in seconds
        """
        self.symbol_id = symbol_id
        self.shm_name = shm_name
        self.batch_mode = batch_mode
        self.poll_interval = poll_interval
        self._data = None
        self._running = False
        
        # Store configuration using BTQuant's TradingConfig
        self.config = TradingConfig()
        # Override HotSpine-specific parameters
        self.config.shm_name = shm_name
        self.config.batch_mode = batch_mode
        self.config.poll_interval = poll_interval
        
        # Performance metrics
        self._store_metrics = {
            'data_requests': 0,
            'feed_errors': 0,
            'start_time': datetime.now().timestamp(),
            'last_request_time': 0
        }
        
        logger.info(f"HotSpineStore initialized for symbol_id={symbol_id}")
    
    def getdata(self, start_date=None, timeframe=TimeFrame.Ticks, compression=1):
        """
        Get HotSpine data feed with comprehensive error handling
        
        Args:
            start_date: Optional start date for historical data
            timeframe: Timeframe for data aggregation
            compression: Compression factor
            
        Returns:
            Configured HotSpineData instance
            
        Raises:
            RuntimeError: If data feed cannot be created after retries
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if self._data is None:
                    self._data = self._create_hotspine_data_feed(timeframe, compression)
                
                self._store_metrics['data_requests'] += 1
                self._store_metrics['last_request_time'] = datetime.now().timestamp()
                
                return self._data
                
            except Exception as e:
                self._store_metrics['feed_errors'] += 1
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Failed to get data feed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    error_msg = f"Max retries ({max_retries}) exceeded for HotSpine data feed"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
    
    def _create_hotspine_data_feed(self, timeframe, compression):
        """
        Create and configure HotSpine data feed
        
        Args:
            timeframe: Timeframe for data
            compression: Compression factor
            
        Returns:
            Configured HotSpineData instance
        """
        try:
            # Create HotSpine data feed with enhanced configuration
            data = HotSpineData(
                symbol_id=self.symbol_id,
                shm_name=self.shm_name,
                batch_mode=self.batch_mode,
                poll_interval=self.poll_interval,
                timeframe=timeframe,
                compression=compression
            )
            
            # Set data name for identification
            data._dataname = f"HotSpine_{self.symbol_id}"
            
            logger.info(f"HotSpine data feed created for symbol_id={self.symbol_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to create HotSpine data feed: {e}")
            self._store_metrics['feed_errors'] += 1
            raise
    
    def get_interval(self, timeframe, compression):
        """
        Get interval string for given timeframe and compression
        
        Args:
            timeframe: Timeframe
            compression: Compression factor
            
        Returns:
            Interval string or None if not supported
        """
        return self._GRANULARITIES.get((timeframe, compression))
    
    def start(self):
        """
        Start the HotSpine store
        """
        if self._running:
            return
        
        self._running = True
        logger.info(f"HotSpineStore started for symbol_id={self.symbol_id}")
    
    def stop(self):
        """
        Stop the HotSpine store and clean up resources
        """
        if not self._running:
            return
        
        self._running = False
        
        # Clean up data feed if it exists
        if self._data:
            try:
                self._data.stop()
            except Exception as e:
                logger.error(f"Error stopping HotSpine data feed: {e}")
            finally:
                self._data = None
        
        logger.info(f"HotSpineStore stopped for symbol_id={self.symbol_id}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get store health status
        
        Returns:
            Dictionary containing health status information
        """
        return {
            'store_healthy': self._running,
            'data_requests': self._store_metrics['data_requests'],
            'feed_errors': self._store_metrics['feed_errors'],
            'uptime_seconds': datetime.now().timestamp() - self._store_metrics['start_time']
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get store performance metrics
        
        Returns:
            Dictionary containing store metrics
        """
        metrics = self._store_metrics.copy()
        
        # Add derived metrics
        uptime = metrics['last_request_time'] - metrics['start_time']
        if uptime > 0 and metrics['data_requests'] > 0:
            metrics['requests_per_second'] = metrics['data_requests'] / uptime
        else:
            metrics['requests_per_second'] = 0
        
        return metrics
    
    def put_notification(self, msg, *args, **kwargs):
        """
        Put notification message
        
        Args:
            msg: Notification message
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        logger.info(f"HotSpineStore notification: {msg}")
    
    def get_notifications(self):
        """
        Get pending notifications
        
        Returns:
            List of notifications
        """
        # For HotSpine, we don't have a queue system like BinanceStore
        # This method is provided for interface consistency
        return []


def create_hotspine_store(
    symbol_id: int,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001
) -> HotSpineStore:
    """
    Factory function to create a HotSpine store
    
    Args:
        symbol_id: Symbol ID to filter trades
        shm_name: Shared memory segment name
        batch_mode: Whether to use batch reading
        poll_interval: Polling interval in seconds
        
    Returns:
        Configured HotSpineStore instance
    """
    return HotSpineStore(
        symbol_id=symbol_id,
        shm_name=shm_name,
        batch_mode=batch_mode,
        poll_interval=poll_interval
    )