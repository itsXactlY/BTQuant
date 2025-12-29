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
        self.batch_mode = self.p.batch_mode
        self.poll_interval = self.p.poll_interval
        self._last_trade_time = 0
        
        # Trade buffer for batch processing
        self._trade_buffer = []
        self._buffer_index = 0
        
        # Set live data flag
        self._islive = True
    
    def start(self):
        """Start the HotSpine data feed"""
        super(HotSpineData, self).start()
        
        try:
            # Initialize HotSpine reader (local import to avoid circular imports)
            HotSpineReader = _import_hotspine_reader()
            self.hotspine_reader = HotSpineReader(self.p.shm_name)
            logger.info(f"HotSpineData: Connected to shared memory {self.p.shm_name}")
            
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
        """Load single trade in polling mode"""
        trade = self.hotspine_reader.poll_trade()
        
        if trade:
            # Filter by symbol if specified
            if self.symbol_id is None or trade.symbol_id == self.symbol_id:
                return self._process_trade(trade)
            else:
                # Trade doesn't match our symbol filter, try again
                time.sleep(self.poll_interval)
                return self._load_single()
        else:
            # No trade available
            time.sleep(self.poll_interval)
            return False
    
    def _load_batch(self):
        """Load trades in batch mode"""
        if self._buffer_index >= len(self._trade_buffer):
            # Buffer is empty, fetch new batch
            trades = self.hotspine_reader.read_all_trades()
            
            # Filter by symbol if specified
            if self.symbol_id:
                trades = [t for t in trades if t.symbol_id == self.symbol_id]
            
            self._trade_buffer = trades
            self._buffer_index = 0
            
            if not self._trade_buffer:
                time.sleep(self.poll_interval)
                return False
        
        if self._buffer_index < len(self._trade_buffer):
            trade = self._trade_buffer[self._buffer_index]
            self._buffer_index += 1
            return self._process_trade(trade)
        
        return False
    
    def _process_trade(self, trade):
        """
        Process HotTrade and convert to Backtrader data format
        
        Args:
            trade: HotTrade object from HotSpine
            
        Returns:
            True if trade was processed successfully
        """
        try:
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


class HotSpineFeed(bt.feed.FeedBase):
    """
    HotSpine Feed for Backtrader
    
    This feed class provides a factory for creating HotSpine data feeds.
    """
    
    DataCls = HotSpineData
    
    params = (
        ('shm_name', '/btquant_hotspine'),  # Shared memory segment name
        ('symbol_id', None),              # Symbol ID to filter trades
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
            'batch_mode': self._batch_mode,
            'poll_interval': self._poll_interval,
        }
        
        # Override with any kwargs
        params.update(kwargs)
        
        # Create and return HotSpineData instance
        return HotSpineData(**params)


def create_hotspine_data_feed(
    symbol_id: int,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001
) -> HotSpineData:
    """
    Factory function to create a HotSpine data feed
    
    Args:
        symbol_id: Symbol ID to filter trades
        shm_name: Shared memory segment name
        batch_mode: Whether to use batch reading
        poll_interval: Polling interval in seconds
        
    Returns:
        Configured HotSpineData instance
    """
    return HotSpineData(
        symbol_id=symbol_id,
        shm_name=shm_name,
        batch_mode=batch_mode,
        poll_interval=poll_interval
    )


def create_hotspine_feed(
    shm_name: str = "/btquant_hotspine",
    symbol_id: int = None,
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