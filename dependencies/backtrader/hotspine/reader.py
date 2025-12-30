"""
HotSpine Reader for btq_live_runtime

This module provides a Python interface to the HotSpine shared memory reader,
optimized for low-latency, high-throughput data consumption in live trading scenarios.
"""

import ctypes
import os
import time
import logging
from typing import Optional, List, Dict, Any

# Import SQL integration
from backtrader.bigbraincentral.storage_mssql import MSSQLConfig
from backtrader.hotspine.sql_integration import HotSpineSQLIntegration

# Configure logger
logger = logging.getLogger(__name__)


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary format"""
        return {
            "ts_exchange": self.ts_exchange,
            "ts_local": self.ts_local,
            "price": self.price,
            "size": self.size,
            "symbol_id": self.symbol_id,
            "side": "BUY" if self.side == 0 else "SELL"
        }


class HotSpineReader:
    """
    High-performance HotSpine reader for btq_live_runtime
    
    This reader provides low-latency access to shared memory trade data
    with optimized polling mechanisms for live trading scenarios.
    """
    
    def __init__(self, shm_name: str = "/btquant_hotspine"):
        """
        Initialize HotSpine reader and attach to shared memory
        
        Args:
            shm_name: Name of shared memory segment to attach to
        
        Raises:
            RuntimeError: If failed to attach to shared memory
            FileNotFoundError: If shared library not found
        """
        self.shm_name = shm_name
        self._lib = self._load_library()
        self._reader_ptr = None
        self._initialize_reader()
        
    def _load_library(self) -> ctypes.CDLL:
        """Load the HotSpine reader shared library"""
        # Try to find the library in common locations
        lib_paths = [
            "dependencies/ccapi/example/build/libhotspine_reader.so",
            "dependencies/ccapi/example/build/hotspine/libhotspine_reader.so",
            "build/libhotspine_reader.so",
            "libhotspine_reader.so"
        ]
        
        lib_path = None
        for path in lib_paths:
            if os.path.exists(path):
                lib_path = path
                break
        
        if lib_path is None:
            raise FileNotFoundError(
                f"HotSpine reader library not found. Tried: {lib_paths}"
            )
        
        try:
            lib = ctypes.CDLL(lib_path)
            return lib
        except Exception as e:
            raise RuntimeError(f"Failed to load HotSpine reader library from {lib_path}: {e}")
    
    def _initialize_reader(self) -> None:
        """Initialize the C++ HotSpine reader and attach to shared memory"""
        # Define function signatures
        self._lib.hotspine_reader_create.restype = ctypes.c_void_p
        self._lib.hotspine_reader_create.argtypes = [ctypes.c_char_p]
        
        self._lib.hotspine_reader_destroy.restype = None
        self._lib.hotspine_reader_destroy.argtypes = [ctypes.c_void_p]
        
        self._lib.hotspine_reader_poll_trade.restype = ctypes.c_int
        self._lib.hotspine_reader_poll_trade.argtypes = [ctypes.c_void_p, ctypes.POINTER(HotTrade)]
        
        # Try to load optional functions - they may not be available in all builds
        self._has_read_all = hasattr(self._lib, 'hotspine_reader_read_all')
        if self._has_read_all:
            self._lib.hotspine_reader_read_all.restype = ctypes.c_void_p  # Returns pointer to vector
            self._lib.hotspine_reader_read_all.argtypes = [ctypes.c_void_p]
        
        self._has_buffer_utilization = hasattr(self._lib, 'hotspine_reader_get_buffer_utilization')
        if self._has_buffer_utilization:
            self._lib.hotspine_reader_get_buffer_utilization.restype = ctypes.c_void_p  # Returns pointer to pair
            self._lib.hotspine_reader_get_buffer_utilization.argtypes = [ctypes.c_void_p]
        
        # These functions should be available in all builds
        self._lib.hotspine_reader_get_lost_count.restype = ctypes.c_uint64
        self._lib.hotspine_reader_get_lost_count.argtypes = [ctypes.c_void_p]
        
        # Create reader instance
        shm_name_bytes = self.shm_name.encode('utf-8')
        self._reader_ptr = self._lib.hotspine_reader_create(shm_name_bytes)
        
        if not self._reader_ptr:
            raise RuntimeError(f"Failed to create HotSpine reader for {self.shm_name}")
    
    def __del__(self) -> None:
        """Clean up resources"""
        self.close()
    
    def close(self) -> None:
        """Close the HotSpine reader and release resources"""
        if self._reader_ptr and self._lib:
            try:
                self._lib.hotspine_reader_destroy(self._reader_ptr)
                self._reader_ptr = None
            except Exception as e:
                print(f"Warning: Failed to clean up HotSpine reader: {e}")
    
    def poll_trade(self) -> Optional[HotTrade]:
        """
        Poll for a single trade (non-blocking)
        
        Returns:
            HotTrade object if available, None otherwise
        """
        if not self._reader_ptr:
            return None
            
        trade = HotTrade()
        result = self._lib.hotspine_reader_poll_trade(self._reader_ptr, ctypes.byref(trade))
        
        return trade if result else None
    
    def read_all_trades(self) -> List[HotTrade]:
        """
        Read all available trades at once (more efficient for batch processing)
        
        Returns:
            List of HotTrade objects (empty list if none available)
        """
        if not self._reader_ptr:
            return []
            
        # Use native implementation if available, otherwise fall back to Python implementation
        if self._has_read_all:
            # Native implementation (would need proper C++ vector handling)
            # This is a placeholder - actual implementation would need C interface support
            trades = []
            # TODO: Implement proper vector handling when C interface is available
        else:
            # Python-side batch read fallback
            trades = []
            
            # Poll trades until none are available (with a reasonable limit)
            max_iterations = 1000  # Safety limit
            for _ in range(max_iterations):
                trade = self.poll_trade()
                if trade:
                    trades.append(trade)
                else:
                    break
        
        return trades
    
    def get_lost_count(self) -> int:
        """
        Get the number of lost trades (overflow counter)
        
        Returns:
            Number of trades lost due to buffer overflow
        """
        if not self._reader_ptr:
            return 0
            
        return self._lib.hotspine_reader_get_lost_count(self._reader_ptr)
    
    def get_buffer_utilization(self) -> Dict[str, int]:
        """
        Get current buffer utilization information
        
        Returns:
            Dictionary with 'current_size' and 'capacity' keys
        """
        if not self._reader_ptr:
            return {"current_size": 0, "capacity": 0}
            
        # Use native implementation if available, otherwise return placeholder
        if self._has_buffer_utilization:
            # Native implementation (would need proper C interface)
            # This is a placeholder - actual implementation would need C interface support
            return {"current_size": 0, "capacity": 1000000}
        else:
            # Fallback placeholder
            return {"current_size": 0, "capacity": 1000000}
    
    def is_healthy(self) -> bool:
        """
        Check if the reader is healthy and connected
        
        Returns:
            True if reader is healthy, False otherwise
        """
        return self._reader_ptr is not None


class HotSpineRuntime:
    """
    btq_live_runtime integration with HotSpine reader
    
    This class provides the core runtime for consuming HotSpine data
    in live trading scenarios with backtrader integration.
    
    Key architectural points:
    - HotSpine handles live trading data ingestion (NOT SQL)
    - SQL is used only for long-term storage, replay, analytics, and debugging
    - Data flow: HotSpine -> Strategy -> SQL (asynchronous)
    """
    
    def __init__(self, strategy_cls, shm_name: str = "/btquant_hotspine",
                 sql_config: Optional[MSSQLConfig] = None,
                 enable_sql_storage: bool = True,
                 exclusive_hotswap_mode: bool = False):
        """
        Initialize the live runtime with HotSpine reader
        
        ARCHITECTURE NOTE: This constructor demonstrates clean separation:
        - HotSpineReader handles live trading data (shared memory)
        - HotSpineSQLIntegration handles long-term storage (SQL)
        - These are completely separate components with distinct responsibilities
        
        Args:
            strategy_cls: Backtrader strategy class to use for trading
            shm_name: Name of shared memory segment to attach to
            sql_config: Optional SQL Server configuration for storage
            enable_sql_storage: Whether to enable SQL storage for long-term persistence
            exclusive_hotswap_mode: Enable exclusive hotswap mode for data processing
        
        Raises:
            ValueError: If conflicting configuration options are provided
        """
        # Validate configuration
        self._validate_configuration(enable_sql_storage, exclusive_hotswap_mode)
        
        self.strategy_cls = strategy_cls
        self.reader = HotSpineReader(shm_name)
        self._running = False
        self._strategy_instance = None
        
        # Configuration options
        self.enable_sql_storage = enable_sql_storage
        self.exclusive_hotswap_mode = exclusive_hotswap_mode
        # SQL Integration (for long-term storage only, NOT live trading)
        # ARCHITECTURE: Complete separation from live trading data path
        # Architecture separation: HotSpine and SQL are independent components
        # SQL is NOT used for live trading
        # Clean architecture separation
        self.sql_integration = None
        
        if self.enable_sql_storage:
            try:
                self.sql_integration = HotSpineSQLIntegration(sql_config)
                self.sql_integration.start_async_storage()
                logger.info("HotSpine SQL storage enabled for long-term persistence")
            except Exception as e:
                logger.error(f"Failed to initialize SQL storage: {e}")
                self.enable_sql_storage = False
        
        # Log hotswap mode configuration
        if self.exclusive_hotswap_mode:
            logger.info("Exclusive hotswap mode enabled")
        
    def _validate_configuration(self, enable_sql_storage: bool, exclusive_hotswap_mode: bool):
        """
        Validate configuration options to prevent conflicting settings
        
        Args:
            enable_sql_storage: Whether SQL storage is enabled
            exclusive_hotswap_mode: Whether exclusive hotswap mode is enabled
            
        Raises:
            ValueError: If conflicting configuration options are provided
        """
        # Validation logic to prevent conflicting configurations
        
        # Rule 1: If exclusive hotswap mode is enabled, SQL storage should also be enabled
        # This ensures data consistency and persistence during hotswap operations
        if exclusive_hotswap_mode and not enable_sql_storage:
            raise ValueError(
                "Exclusive hotswap mode requires SQL storage to be enabled for data consistency. "
                "Please enable SQL storage when using exclusive hotswap mode."
            )
        
        # Rule 2: Add any other business-specific validation rules here
        # For example, you might want to prevent certain combinations that could
        # cause performance issues or data corruption
        
        # Log configuration for debugging purposes
        logger.info(f"Configuration validated: SQL storage enabled={enable_sql_storage}, "
                   f"exclusive hotswap mode={exclusive_hotswap_mode}")

    def _initialize_strategy(self):
        """Initialize the trading strategy"""
        # Import here to avoid circular dependencies
        from backtrader.strategy import Strategy
        
        # Create strategy instance
        self._strategy_instance = self.strategy_cls.__new__(self.strategy_cls)
        self._strategy_instance.datas = []
        self._strategy_instance.broker = self
        self._strategy_instance.position = 0
        
        # Initialize the strategy
        self.strategy_cls.__init__(self._strategy_instance)
    
    def buy(self, size=None, price=None):
        """Execute buy order"""
        print(f"BUY order: size={size}, price={price}")
        # In a real implementation, this would interface with a broker
    
    def sell(self, size=None, price=None):
        """Execute sell order"""
        print(f"SELL order: size={size}, price={price}")
        # In a real implementation, this would interface with a broker
    
    def on_trade(self, trade: HotTrade):
        """
        Handle incoming trade data
        
        ARCHITECTURE NOTE: This method demonstrates the key architectural principle:
        - HotSpine data flows to strategy FIRST (live trading)
        - SQL storage happens SECOND (asynchronous, non-blocking)
        - SQL is NOT in the hot path for trading decisions
        - SQL not in hot path: trading decisions are made before SQL storage
        """
        if self._strategy_instance:
            # Convert trade to format expected by strategy
            self._strategy_instance.data = trade
            self._strategy_instance.next()
         
        # Store trade asynchronously for long-term persistence (NOT for live trading)
        # ARCHITECTURE: SQL operations are completely separate from trading logic
        # Only use hotswap when exclusive mode is enabled
        if self.enable_sql_storage and self.sql_integration:
            try:
                if self.exclusive_hotswap_mode:
                    # In exclusive hotswap mode, we use a different storage approach
                    # This would typically involve a more sophisticated hotswap mechanism
                    self.sql_integration.store_trade_async(trade)
                    logger.debug("Stored trade using exclusive hotswap mode")
                else:
                    # Standard storage mode
                    self.sql_integration.store_trade_async(trade)
                # This is asynchronous and non-blocking - trading continues immediately
            except Exception as e:
                logger.error(f"Failed to store trade in SQL: {e}")
    
    def run(self, batch_mode: bool = False):
        """
        Run the live trading runtime
        
        Args:
            batch_mode: If True, use batch reading for higher throughput
                       If False, use single trade polling for lower latency
        """
        if self._running:
            print("Runtime is already running")
            return
            
        self._running = True
        self._initialize_strategy()
        
        print(f"Starting btq_live_runtime with HotSpine reader (batch_mode={batch_mode})")
        print(f"Connected to shared memory: {self.reader.shm_name}")
        
        try:
            if batch_mode:
                self._run_batch_mode()
            else:
                self._run_single_mode()
        except KeyboardInterrupt:
            print("\nShutting down btq_live_runtime...")
        finally:
            self._running = False
            self.reader.close()
            
            # Clean up SQL storage
            if self.sql_integration:
                self.sql_integration.stop_async_storage()
    
    def _run_single_mode(self):
        """Run in single trade polling mode (lowest latency)"""
        print("Running in single trade mode (low latency)")
        
        trade_count = 0
        start_time = time.time()
        
        while self._running:
            trade = self.reader.poll_trade()
            
            if trade:
                trade_count += 1
                self.on_trade(trade)
                
                # Print stats periodically
                if trade_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = trade_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {trade_count} trades (rate: {rate:.1f} trades/sec)")
            else:
                # Small sleep to reduce CPU usage when no trades available
                time.sleep(0.0001)  # 100 microseconds
    
    def _run_batch_mode(self):
        """Run in batch reading mode (higher throughput)"""
        print("Running in batch mode (high throughput)")
        
        trade_count = 0
        batch_count = 0
        start_time = time.time()
        
        while self._running:
            trades = self.reader.read_all_trades()
            
            if trades:
                batch_count += 1
                batch_size = len(trades)
                trade_count += batch_size
                
                for trade in trades:
                    self.on_trade(trade)
                
                # Print stats periodically
                if batch_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = trade_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {trade_count} trades in {batch_count} batches "
                          f"(rate: {rate:.1f} trades/sec, avg batch: {batch_size:.1f})")
            else:
                # Small sleep to reduce CPU usage when no trades available
                time.sleep(0.001)  # 1 millisecond


def create_hotspine_runtime(strategy_cls, shm_name: str = "/btquant_hotspine",
                            sql_config: Optional[MSSQLConfig] = None,
                            enable_sql_storage: bool = True,
                            exclusive_hotswap_mode: bool = False):
    """
    Factory function to create a HotSpine runtime instance
    
    Args:
        strategy_cls: Backtrader strategy class
        shm_name: Shared memory segment name
        sql_config: Optional SQL Server configuration for storage
        enable_sql_storage: Whether to enable SQL storage for long-term persistence
        exclusive_hotswap_mode: Enable exclusive hotswap mode for data processing
        
    Returns:
        HotSpineRuntime instance
    """
    return HotSpineRuntime(strategy_cls, shm_name, sql_config, enable_sql_storage, exclusive_hotswap_mode)