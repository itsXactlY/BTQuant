"""
HotSpine Data Feed for Backtrader

This module provides a Backtrader-compatible data feed that reads from HotSpine
shared memory for live trading scenarios.
"""

import backtrader as bt
from backtrader.feed import DataBase
from datetime import datetime
import time
import logging
from typing import Dict, Any

# Import configuration management
from backtrader.hotspine.config import HotSpineConfig, configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


# Import HotTrade structure locally to avoid circular imports
def _import_hottrade():
    """Local import to avoid circular imports"""
    from backtrader.hotspine.reader import HotTrade
    return HotTrade


# Import HotSpineReader locally when needed
def _import_hotspine_reader():
    """Local import to avoid circular imports"""
    from backtrader.hotspine.reader import HotSpineReader
    return HotSpineReader


class HotSpineData(DataBase):
    """
    HotSpine Data Feed for Backtrader
    
    This data feed integrates with HotSpine shared memory to provide
    low-latency trade data for live trading while maintaining compatibility
    with Backtrader's data feed interface.
    """
    
    params = (
        ('shm_name', '/btquant_hotspine'),  # Shared memory segment name
        ('symbol_id', None),              # Symbol ID to filter trades
        ('symbol', ''),                   # Symbol string (e.g. BTC/USDT)
        ('timeframe', bt.TimeFrame.Ticks), # Timeframe (ticks for live trading)
        ('compression', 1),               # Compression (1 for raw ticks)
        ('batch_mode', False),            # Whether to use batch reading
        ('poll_interval', 0.0001),        # Polling interval in seconds
    )
    
    def __init__(self):
        super(HotSpineData, self).__init__()
        
        # Initialize HotSpine reader
        self.hotspine_reader = None
        self.symbol_id = self.p.symbol_id
        self.symbol = self.p.symbol or self.p.dataname
        self.batch_mode = self.p.batch_mode
        self.poll_interval = self.p.poll_interval
        self._last_trade_time = 0
        
        # Trade buffer for batch processing
        self._trade_buffer = []
        self._buffer_index = 0
        
        # Set live data flag
        self._islive = True
        
        # Performance metrics
        self._feed_metrics = {
            'trades_processed': 0,
            'processing_errors': 0,
            'start_time': 0,
            'last_trade_time': 0
        }
    
    def start(self):
        """Start the HotSpine data feed with enhanced configuration"""
        super(HotSpineData, self).start()
        
        try:
            # Create configuration for HotSpine reader
            config = HotSpineConfig()
            config.shm_name = self.p.shm_name
            config.batch_mode = self.p.batch_mode
            config.poll_interval = self.p.poll_interval
            
            # Initialize HotSpineReader (local import to avoid circular imports)
            HotSpineReader = _import_hotspine_reader()
            self.hotspine_reader = HotSpineReader(
                shm_name=config.shm_name,
                poll_interval=config.poll_interval
            )
            logger.info(f"HotSpineData: Connected to shared memory {self.p.shm_name}")
            
            # Initialize metrics
            self._feed_metrics['start_time'] = time.time()
            
            # Mark as live data feed
            self.put_notification(self.LIVE)
            
        except Exception as e:
            logger.error(f"HotSpineData: Failed to initialize HotSpine reader: {e}")
            raise
    
    def stop(self):
        """Stop the HotSpine data feed"""
        if self.hotspine_reader:
            self.hotspine_reader.close()
            logger.info("HotSpineData: Disconnected from HotSpine")
        super(HotSpineData, self).stop()
    
    def islive(self):
        """Return True to indicate this is a live data feed"""
        return True
    
    def _load(self):
        """
        Load next trade from HotSpine
        
        Returns:
            True if trade was loaded, False if no more data available
        """
        try:
            if self.batch_mode:
                return self._load_batch()
            else:
                return self._load_single()
                
        except Exception as e:
            logger.error(f"HotSpineData: Error loading trade: {e}")
            return False
    
    def _load_single(self):
        """Load single trade in polling mode, blocking until data is available"""
        while True:
            try:
                trade = self.hotspine_reader.poll_trade()
                
                if trade:
                    if self.symbol_id is None or trade.symbol_id == self.symbol_id:
                        return self._process_trade(trade)
                else:
                    time.sleep(self.poll_interval)
                    
            except Exception as e:
                logger.error(f"HotSpineData: Exception in _load_single: {e}")
                time.sleep(1.0) # Wait before retry
    
    def _load_batch(self):
        """Load trades in batch mode, blocking until a batch is available"""
        while self._buffer_index >= len(self._trade_buffer):
            # Buffer is empty, try to fetch new batch
            trades = self.hotspine_reader.read_all_trades()
            
            if trades:
                # Filter by symbol if specified
                if self.symbol_id:
                    trades = [t for t in trades if t.symbol_id == self.symbol_id]
                
                if trades:
                    self._trade_buffer = trades
                    self._buffer_index = 0
                    break
            
            # No trades found, wait and try again
            time.sleep(self.poll_interval)
        
        if self._buffer_index < len(self._trade_buffer):
            trade = self._trade_buffer[self._buffer_index]
            self._buffer_index += 1
            return self._process_trade(trade)
        
        return False
    
    def _process_trade(self, trade):
        """
        Process HotTrade and convert to Backtrader data format with performance monitoring
        
        Args:
            trade: HotTrade object from HotSpine
            
        Returns:
            True if trade was processed successfully
        """
        try:
            # Update feed metrics
            self._feed_metrics['trades_processed'] += 1
            self._feed_metrics['last_trade_time'] = time.time()
            
            # Convert microseconds to seconds for datetime
            trade_time = trade.ts_exchange / 1_000_000.0
            
            # Convert to datetime (UTC)
            dt = datetime.utcfromtimestamp(trade_time)
            
            # Set data lines (ensure we have initialized timezone)
            if not hasattr(self, '_tz'):
                self._tz = None  # Default timezone for live data
                self.lines.datetime._settz(self._tz)
            
            self.lines.datetime[0] = self.date2num(dt)
            self.lines.open[0] = float(trade.price)
            self.lines.high[0] = float(trade.price)
            self.lines.low[0] = float(trade.price)
            self.lines.close[0] = float(trade.price)
            self.lines.volume[0] = float(trade.size)
            
            # Set tick values for live trading
            self.tick_datetime = self.lines.datetime[0]
            self.tick_open = self.lines.open[0]
            self.tick_high = self.lines.high[0]
            self.tick_low = self.lines.low[0]
            self.tick_close = self.lines.close[0]
            self.tick_volume = self.lines.volume[0]
            
            # Update last trade time
            self._last_trade_time = trade_time
            
            return True
            
        except Exception as e:
            logger.error(f"HotSpineData: Error processing trade: {e}")
            self._feed_metrics['processing_errors'] += 1
            return False
    
    def haslivedata(self):
        """Return True to indicate this feed has live data"""
        return True
    
    def qcheck(self, limit):
        """
        Override qcheck to provide custom behavior for live data
        
        Args:
            limit: Maximum time to wait for new data
            
        Returns:
            True if new data is available, False otherwise
        """
        # For HotSpine, we want to check for new trades immediately
        # but respect the polling interval
        time.sleep(min(self.poll_interval, limit))
        return True
    
    def get_feed_metrics(self) -> Dict[str, Any]:
        """
        Get feed performance metrics
        
        Returns:
            Dictionary containing feed metrics
        """
        metrics = self._feed_metrics.copy()
        
        # Add derived metrics
        if metrics['trades_processed'] > 0 and metrics['start_time'] > 0:
            runtime = metrics['last_trade_time'] - metrics['start_time']
            if runtime > 0:
                metrics['trades_per_second'] = metrics['trades_processed'] / runtime
            else:
                metrics['trades_per_second'] = 0
        else:
            metrics['trades_per_second'] = 0
        
        # Add reader metrics if available
        if self.hotspine_reader:
            metrics['reader_metrics'] = self.hotspine_reader.get_statistics()
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get feed health status
        
        Returns:
            Dictionary containing health status information
        """
        return {
            'feed_healthy': self.hotspine_reader.is_healthy() if self.hotspine_reader else False,
            'trades_processed': self._feed_metrics['trades_processed'],
            'processing_errors': self._feed_metrics['processing_errors']
        }


class HotSpineFeed(bt.feed.FeedBase):
    """
    HotSpine Feed for Backtrader
    
    This feed class provides a factory for creating HotSpine data feeds.
    """
    
    DataCls = HotSpineData
    
    params = (
        ('shm_name', '/btquant_hotspine'),  # Shared memory segment name
        ('symbol_id', None),              # Symbol ID to filter trades
        ('symbol', ''),                   # Symbol string
        ('batch_mode', False),            # Whether to use batch reading
        ('poll_interval', 0.0001),        # Polling interval in seconds
    )
    
    def __init__(self):
        super(HotSpineFeed, self).__init__()
        self._shm_name = self.p.shm_name
        self._symbol_id = self.p.symbol_id
        self._batch_mode = self.p.batch_mode
        self._poll_interval = self.p.poll_interval
    
    def _getdata(self, dataname, **kwargs):
        """
        Create a HotSpineData instance with the specified parameters
        
        Args:
            dataname: Data name (symbol)
            **kwargs: Additional parameters
            
        Returns:
            HotSpineData instance
        """
        # Merge feed parameters with data parameters
        params = {
            'shm_name': self._shm_name,
            'symbol_id': self._symbol_id,
            'symbol': self.p.symbol,
            'batch_mode': self._batch_mode,
            'poll_interval': self._poll_interval,
        }
        
        # Override with any kwargs
        params.update(kwargs)
        
        # Create and return HotSpineData instance
        return HotSpineData(**params)


def create_hotspine_data_feed(
    symbol_id: int,
    symbol: str = "",
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001
) -> HotSpineData:
    """
    Factory function to create a HotSpine data feed
    
    Args:
        symbol_id: Symbol ID to filter trades
        symbol: Symbol string (e.g. BTC/USDT)
        shm_name: Shared memory segment name
        batch_mode: Whether to use batch reading
        poll_interval: Polling interval in seconds
        
    Returns:
        Configured HotSpineData instance
    """
    return HotSpineData(
        symbol_id=symbol_id,
        symbol=symbol,
        shm_name=shm_name,
        batch_mode=batch_mode,
        poll_interval=poll_interval
    )


def create_hotspine_feed(
    shm_name: str = "/btquant_hotspine",
    symbol_id: int = None,
    symbol: str = "",
    batch_mode: bool = False,
    poll_interval: float = 0.0001
) -> HotSpineFeed:
    """
    Factory function to create a HotSpine feed
    
    Args:
        shm_name: Shared memory segment name
        symbol_id: Symbol ID to filter trades
        batch_mode: Whether to use batch reading
        poll_interval: Polling interval in seconds
        
    Returns:
        Configured HotSpineFeed instance
    """
    return HotSpineFeed(
        shm_name=shm_name,
        symbol_id=symbol_id,
        batch_mode=batch_mode,
        poll_interval=poll_interval
    )