"""
HotSpine SQL Integration for long-term storage, replay, analytics, and debugging

This module provides SQL integration that aligns with the HotSpine architecture:
- SQL is NOT used for live trading data ingestion (HotSpine handles that)
- SQL is used for: long-term storage, historical replay, analytics, and debugging
- Data flows: HotSpine -> Strategy -> SQL (asynchronous storage)
"""

import threading
import queue
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Import SQL storage classes
from backtrader.bigbraincentral.storage_mssql import MarketDataStorage, MSSQLConfig
# Import HotTrade using a local definition to avoid circular import
import ctypes

class HotTrade(ctypes.Structure):
    """Python representation of a HotSpine trade structure"""
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),  # Exchange timestamp in microseconds
        ("ts_local", ctypes.c_uint64),     # Local receive timestamp in microseconds
        ("price", ctypes.c_double),        # Trade price
        ("size", ctypes.c_double),         # Trade size
        ("symbol_id", ctypes.c_uint32),    # Symbol ID (hash or mapping)
        ("side", ctypes.c_uint8),         # 0=buy, 1=sell
    ]

    def __repr__(self) -> str:
        side_str = "BUY" if self.side == 0 else "SELL"
        return (f"HotTrade(ts_exchange={self.ts_exchange}, ts_local={self.ts_local}, "
                f"price={self.price}, size={self.size}, symbol_id={self.symbol_id}, "
                f"side={side_str})")

    def to_dict(self) -> dict:
        """Convert trade to dictionary format"""
        return {
            "ts_exchange": self.ts_exchange,
            "ts_local": self.ts_local,
            "price": self.price,
            "size": self.size,
            "symbol_id": self.symbol_id,
            "side": "BUY" if self.side == 0 else "SELL"
        }

logger = logging.getLogger(__name__)


class HotSpineSQLIntegration:
    """
    SQL Integration for HotSpine that focuses on long-term storage and analytics
    
    This class provides:
    1. Asynchronous storage of HotSpine trade data to SQL
    2. Historical data replay capabilities
    3. Analytics and debugging support
    4. Complete separation from live trading data path
    """
    
    def __init__(self, sql_config: Optional[MSSQLConfig] = None):
        """
        Initialize HotSpine SQL Integration
        
        Args:
            sql_config: SQL Server configuration. If None, uses default config.
        """
        self.sql_config = sql_config or MSSQLConfig()
        self.storage = MarketDataStorage(self.sql_config)
        
        # Queue for asynchronous storage
        self.storage_queue = queue.Queue(maxsize=10000)
        self.storage_thread = None
        self.running = False
        
        # Statistics
        self.trades_stored = 0
        self.storage_errors = 0
        self.last_storage_time = 0
        
        # Connect to SQL Server
        self._connect_sql()
    
    def _connect_sql(self):
        """Establish connection to SQL Server"""
        try:
            self.storage.connect()
            logger.info("HotSpine SQL Integration connected to SQL Server")
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {e}")
            raise
    
    def start_async_storage(self):
        """Start the asynchronous storage thread"""
        if self.running:
            return
        
        self.running = True
        self.storage_thread = threading.Thread(
            target=self._storage_worker,
            name="HotSpineSQLStorage",
            daemon=True
        )
        self.storage_thread.start()
        logger.info("HotSpine SQL Integration async storage started")
    
    def stop_async_storage(self):
        """Stop the asynchronous storage thread"""
        if not self.running:
            return
        
        self.running = False
        if self.storage_thread:
            self.storage_thread.join(timeout=5.0)
            if self.storage_thread.is_alive():
                logger.warning("Storage thread did not stop gracefully")
        
        # Clear any remaining items in queue
        while not self.storage_queue.empty():
            try:
                self.storage_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("HotSpine SQL Integration async storage stopped")
    
    def _storage_worker(self):
        """Worker thread for asynchronous SQL storage"""
        logger.info("Storage worker thread started")
        
        while self.running:
            try:
                # Get batch of trades from queue
                batch = []
                start_time = time.time()
                
                # Process items in batches for efficiency
                while self.running and len(batch) < 100:
                    try:
                        # Timeout to allow periodic checks of self.running
                        item = self.storage_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        break
                
                if batch:
                    self._store_batch(batch)
                    batch_process_time = time.time() - start_time
                    self.last_storage_time = batch_process_time
                    
                    # Sleep briefly to avoid overwhelming SQL server
                    time.sleep(0.01)
                else:
                    # No items, sleep longer to reduce CPU usage
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Storage worker error: {e}")
                time.sleep(1.0)  # Wait before retrying
        
        logger.info("Storage worker thread stopped")
    
    def _store_batch(self, batch: List[Dict[str, Any]]):
        """Store a batch of trades to SQL database"""
        try:
            success_count = 0
            
            for item in batch:
                if isinstance(item, HotTrade):
                    # Convert HotTrade to dict format for storage
                    trade_data = self._convert_hottrade_to_dict(item)
                    if self.storage.store_trade(trade_data):
                        success_count += 1
                elif isinstance(item, dict):
                    # Assume it's already in the right format
                    if 'price' in item and 'quantity' in item:
                        if self.storage.store_trade(item):
                            success_count += 1
                    elif 'open' in item:  # Likely OHLCV data
                        if self.storage.store_ohlcv(item):
                            success_count += 1
            
            self.trades_stored += success_count
            
            if success_count < len(batch):
                self.storage_errors += (len(batch) - success_count)
                logger.warning(f"Partial storage success: {success_count}/{len(batch)} items stored")
            
        except Exception as e:
            logger.error(f"Batch storage failed: {e}")
            self.storage_errors += len(batch)
    
    def _convert_hottrade_to_dict(self, trade: HotTrade) -> Dict[str, Any]:
        """Convert HotTrade object to dictionary format for SQL storage"""
        return {
            'timestamp': trade.ts_exchange * 1000,  # Convert microseconds to milliseconds
            'exchange': 'hotspine',  # Could be enhanced with symbol mapping
            'symbol': f'symbol_{trade.symbol_id}',  # Placeholder - needs symbol mapping
            'market_type': 'spot',
            'trade_id': f'hotspine_{trade.ts_exchange}_{trade.symbol_id}',
            'price': float(trade.price),
            'quantity': float(trade.size),
            'side': 'buy' if trade.side == 0 else 'sell',
            'is_buyer_maker': None
        }
    
    def store_trade_async(self, trade: HotTrade):
        """
        Asynchronously store a HotSpine trade to SQL database
        
        This method is non-blocking and returns immediately.
        The actual storage happens in a background thread.
        """
        if not self.running:
            logger.warning("Cannot store trade: async storage not running")
            return False
        
        try:
            self.storage_queue.put_nowait(trade)
            return True
        except queue.Full:
            logger.warning("Storage queue full, dropping trade")
            self.storage_errors += 1
            return False
    
    def store_ohlcv_async(self, ohlcv_data: Dict[str, Any]):
        """
        Asynchronously store OHLCV data to SQL database
        
        Args:
            ohlcv_data: Dictionary containing OHLCV data with keys:
                       exchange, symbol, timestamp, timeframe, open, high, low, close, volume
        """
        if not self.running:
            logger.warning("Cannot store OHLCV: async storage not running")
            return False
        
        try:
            self.storage_queue.put_nowait(ohlcv_data)
            return True
        except queue.Full:
            logger.warning("Storage queue full, dropping OHLCV data")
            self.storage_errors += 1
            return False
    
    # ========================================================================
    # Replay and Analytics Methods
    # ========================================================================
    
    def get_historical_trades(self, exchange: str, symbol: str, 
                             start: datetime, end: Optional[datetime] = None,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical trades for replay and analytics
        
        Args:
            exchange: Exchange name
            symbol: Symbol/pair name
            start: Start datetime
            end: End datetime (optional)
            limit: Maximum number of trades to retrieve (optional)
            
        Returns:
            List of trade dictionaries
        """
        try:
            return self.storage.get_trades(exchange, symbol, start, end, limit)
        except Exception as e:
            logger.error(f"Failed to get historical trades: {e}")
            return []
    
    def get_historical_ohlcv(self, exchange: str, symbol: str, timeframe: str,
                            start: datetime, end: Optional[datetime] = None,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical OHLCV data for replay and analytics
        
        Args:
            exchange: Exchange name
            symbol: Symbol/pair name
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            start: Start datetime
            end: End datetime (optional)
            limit: Maximum number of candles to retrieve (optional)
            
        Returns:
            List of OHLCV dictionaries
        """
        try:
            return self.storage.get_ohlcv(exchange, symbol, timeframe, start, end, limit)
        except Exception as e:
            logger.error(f"Failed to get historical OHLCV: {e}")
            return []
    
    def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
        """
        Get latest price for analytics and monitoring
        
        Args:
            exchange: Exchange name
            symbol: Symbol/pair name
            
        Returns:
            Latest price or None if not available
        """
        try:
            return self.storage.get_latest_price(exchange, symbol)
        except Exception as e:
            logger.error(f"Failed to get latest price: {e}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring and debugging
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            return self.storage.get_stats()
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for monitoring
        
        Returns:
            Dictionary containing storage statistics
        """
        return {
            'trades_stored': self.trades_stored,
            'storage_errors': self.storage_errors,
            'queue_size': self.storage_queue.qsize(),
            'last_storage_time_ms': self.last_storage_time * 1000 if self.last_storage_time else 0,
            'running': self.running
        }
    
    # ========================================================================
    # Debugging Methods
    # ========================================================================
    
    def log_trade_for_debugging(self, trade: HotTrade, context: str = ""):
        """
        Log trade information for debugging purposes
        
        Args:
            trade: HotTrade object
            context: Additional context information
        """
        trade_dict = self._convert_hottrade_to_dict(trade)
        log_message = f"DEBUG TRADE {context}: {trade_dict}"
        
        # Store in SQL for debugging
        try:
            debug_data = {
                'timestamp': datetime.now().timestamp() * 1000,
                'exchange': 'debug',
                'symbol': f'debug_{trade.symbol_id}',
                'market_type': 'debug',
                'trade_id': f'debug_{trade.ts_exchange}',
                'price': trade_dict['price'],
                'quantity': trade_dict['quantity'],
                'side': trade_dict['side'],
                'is_buyer_maker': None,
                'debug_context': context
            }
            
            # Use synchronous storage for debugging to ensure it's captured
            self.storage.store_trade(debug_data)
            logger.debug(log_message)
            
        except Exception as e:
            logger.error(f"Failed to log debug trade: {e}")
    
    def create_replay_data_feed(self, exchange: str, symbol: str,
                               start: datetime, end: datetime,
                               timeframe: str = '1m') -> List[Dict[str, Any]]:
        """
        Create a data feed for replay purposes
        
        Args:
            exchange: Exchange name
            symbol: Symbol/pair name
            start: Start datetime
            end: End datetime
            timeframe: Timeframe for OHLCV data
            
        Returns:
            List of data points for replay
        """
        try:
            # Get OHLCV data for replay
            ohlcv_data = self.get_historical_ohlcv(exchange, symbol, timeframe, start, end)
            
            # Get trade data for detailed replay
            trade_data = self.get_historical_trades(exchange, symbol, start, end, limit=1000)
            
            # Combine and sort by timestamp
            combined = []
            
            for item in ohlcv_data:
                combined.append({
                    'type': 'ohlcv',
                    'timestamp': item['timestamp'],
                    'data': item
                })
            
            for item in trade_data:
                combined.append({
                    'type': 'trade',
                    'timestamp': item['timestamp'],
                    'data': item
                })
            
            # Sort by timestamp
            combined.sort(key=lambda x: x['timestamp'])
            
            return combined
            
        except Exception as e:
            logger.error(f"Failed to create replay data feed: {e}")
            return []


def create_hotspine_sql_integration(sql_config: Optional[MSSQLConfig] = None) -> HotSpineSQLIntegration:
    """
    Factory function to create HotSpine SQL Integration instance
    
    Args:
        sql_config: Optional SQL Server configuration
        
    Returns:
        HotSpineSQLIntegration instance
    """
    return HotSpineSQLIntegration(sql_config)
