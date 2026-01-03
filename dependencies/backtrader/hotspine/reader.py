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
from datetime import datetime
import threading
import queue

# Import configuration management
from backtrader.hotspine.config import HotSpineConfig, configure_logging

# Import SQL integration
from backtrader.bigbraincentral.storage_mssql import MSSQLConfig
from backtrader.hotspine.sql_integration import HotSpineSQLIntegration

# Configure logger
configure_logging()
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
        ("market_type", ctypes.c_uint8),  # 0=spot, 1=futures, 2=other
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
    
    Enhanced with:
    - Configuration management
    - Error handling and automatic recovery
    - Performance monitoring
    - Health monitoring
    """
    
    def __init__(self, config: Optional[HotSpineConfig] = None):
        """
        Initialize HotSpine reader and attach to shared memory
        
        Args:
            config: HotSpine configuration. If None, uses default configuration.
        
        Raises:
            RuntimeError: If failed to attach to shared memory
            FileNotFoundError: If shared library not found
        """
        # Load configuration
        self.config = config or HotSpineConfig()
        self.shm_name = self.config.shm_name
        
        # Initialize state
        self._lib = None
        self._reader_ptr = None
        self._healthy = False
        self._reconnect_attempts = 0
        
        # Performance metrics
        self._metrics = {
            'trades_read': 0,
            'bytes_read': 0,
            'read_errors': 0,
            'last_read_time': 0,
            'read_latency_sum': 0,
            'read_latency_count': 0,
            'trades_filtered': 0,  # NEW: Track filtered trades
            'filtered_by_market_type': 0,  # NEW: Track market type filtering
            'filtered_by_symbol_whitelist': 0  # NEW: Track symbol whitelist filtering
        }
        
        # Health monitoring
        self._health_monitor_thread = None
        self._health_monitor_running = False
        
        # Initialize reader with error handling
        self._initialize_reader()
        
        # Start health monitoring if enabled
        if self.config.enable_monitoring:
            self._start_health_monitoring()
        
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
        """Initialize the C++ HotSpine reader and attach to shared memory with error handling"""
        try:
            # Load library with error handling
            self._lib = self._load_library()
            
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
            
            self._healthy = True
            self._reconnect_attempts = 0
            logger.info(f"HotSpine reader initialized successfully for {self.shm_name}")
            
        except Exception as e:
            self._healthy = False
            logger.error(f"Failed to initialize HotSpine reader: {e}")
            raise
    
    def _try_reconnect(self) -> bool:
        """
        Attempt to reconnect to shared memory
        
        Returns:
            True if reconnection successful, False otherwise
        """
        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.config.max_reconnect_attempts}) reached")
            return False
        
        self._reconnect_attempts += 1
        logger.warning(f"Attempting to reconnect to HotSpine ({self._reconnect_attempts}/{self.config.max_reconnect_attempts})...")
        
        try:
            # Close existing connection if any
            if self._reader_ptr:
                try:
                    self._lib.hotspine_reader_destroy(self._reader_ptr)
                except:
                    pass
                self._reader_ptr = None
            
            # Wait before reconnecting
            time.sleep(self.config.reconnect_delay)
            
            # Reinitialize
            self._initialize_reader()
            logger.info("HotSpine reconnection successful")
            return True
            
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")
            return False
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        if not self.config.enable_monitoring:
            return
        
        self._health_monitor_running = True
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor,
            name="HotSpineHealthMonitor",
            daemon=True
        )
        self._health_monitor_thread.start()
        logger.info("HotSpine health monitoring started")
    
    def _health_monitor(self):
        """Health monitoring thread that periodically checks reader health"""
        while self._health_monitor_running:
            try:
                # Check if reader is healthy
                if not self.is_healthy():
                    logger.warning("HotSpine reader health check failed")
                    self._try_reconnect()
                
                # Log metrics periodically
                if self.config.enable_monitoring:
                    self._log_metrics()
                
                # Sleep for monitoring interval
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(5.0)  # Longer sleep on error
    
    def _log_metrics(self):
        """Log performance metrics"""
        if self._metrics['read_latency_count'] > 0:
            avg_latency = self._metrics['read_latency_sum'] / self._metrics['read_latency_count']
            logger.debug(f"HotSpine Metrics - Trades: {self._metrics['trades_read']}, "
                        f"Avg Latency: {avg_latency:.6f}s, "
                        f"Errors: {self._metrics['read_errors']}")
    
    def __del__(self) -> None:
        """Clean up resources"""
        self.close()
    
    def close(self) -> None:
        """Close the HotSpine reader and release resources"""
        # Stop health monitoring
        self._health_monitor_running = False
        if self._health_monitor_thread:
            try:
                self._health_monitor_thread.join(timeout=2.0)
            except:
                pass
        
        # Close reader connection
        if self._reader_ptr and self._lib:
            try:
                self._lib.hotspine_reader_destroy(self._reader_ptr)
                self._reader_ptr = None
            except Exception as e:
                logger.warning(f"Failed to clean up HotSpine reader: {e}")
        
        self._healthy = False
        logger.info("HotSpine reader closed")
    
    def poll_trade(self) -> Optional[HotTrade]:
        """
        Poll for a single trade (non-blocking) with performance monitoring
        
        Returns:
            HotTrade object if available, None otherwise
        """
        if not self._reader_ptr:
            if not self._healthy:
                # Try to reconnect if not healthy
                if self._try_reconnect():
                    return self.poll_trade()  # Retry after successful reconnection
            return None
        
        try:
            trade = HotTrade()
            start_time = time.time()
            result = self._lib.hotspine_reader_poll_trade(self._reader_ptr, ctypes.byref(trade))
            
            # Update metrics
            if result:
                self._metrics['trades_read'] += 1
                self._metrics['bytes_read'] += ctypes.sizeof(HotTrade)
                latency = time.time() - start_time
                self._metrics['read_latency_sum'] += latency
                self._metrics['read_latency_count'] += 1
                self._metrics['last_read_time'] = time.time()
            
            return trade if result else None
            
        except Exception as e:
            self._metrics['read_errors'] += 1
            self._healthy = False
            logger.error(f"Error polling trade: {e}")
            return None
    
    def poll_trade_with_filtering(self) -> Optional[HotTrade]:
        """
        Poll for a single trade and apply filtering based on configuration
        
        Returns:
            HotTrade object if available and passes filters, None otherwise
        """
        trade = self.poll_trade()
        if trade and self._filter_trade(trade):
            self._metrics['trades_filtered'] += 1
            return None
        return trade
    
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
        if not self._healthy:
            return False
        
        # Additional health checks
        if not self._reader_ptr or not self._lib:
            self._healthy = False
            return False
        
        try:
            # Test connection by getting lost count
            _ = self.get_lost_count()
            return True
        except:
            self._healthy = False
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
            """
            Get current performance metrics
            
            Returns:
                Dictionary containing performance metrics
            """
            metrics = self._metrics.copy()
            
            # Calculate derived metrics
            if metrics['read_latency_count'] > 0:
                metrics['avg_read_latency'] = metrics['read_latency_sum'] / metrics['read_latency_count']
            else:
                metrics['avg_read_latency'] = 0
            
            # Calculate filtering efficiency
            total_trades = metrics['trades_read'] + metrics['trades_filtered']
            if total_trades > 0:
                metrics['filtering_efficiency'] = metrics['trades_filtered'] / total_trades
            else:
                metrics['filtering_efficiency'] = 0
            
            # Add health status
            metrics['healthy'] = self.is_healthy()
            metrics['reconnect_attempts'] = self._reconnect_attempts
            
            return metrics
    
    def _get_market_type_from_symbol(self, symbol_id: int) -> str:
        """
        Get market type from symbol mapping or default to spot
        
        Args:
            symbol_id: Symbol ID to look up
            
        Returns:
            Market type as string ('spot', 'futures', 'other')
        """
        if self.config.symbol_mapping and symbol_id in self.config.symbol_mapping:
            symbol_info = self.config.symbol_mapping[symbol_id]
            if isinstance(symbol_info, dict) and 'market_type' in symbol_info:
                return symbol_info['market_type']
            # Backward compatibility: if symbol_mapping is old format (symbol_id -> symbol_name)
            return 'spot'
        return 'spot'
    
    def _filter_trade(self, trade: HotTrade) -> bool:
        """
        Filter trade based on market type and symbol whitelist configuration
        
        Args:
            trade: HotTrade object to filter
            
        Returns:
            True if trade should be filtered (excluded), False if it should pass through
        """
        # Check market type filtering
        if self.config.market_type_filter != "all":
            market_type = self._get_market_type_from_symbol(trade.symbol_id)
            if market_type != self.config.market_type_filter:
                self._metrics['filtered_by_market_type'] += 1
                return True
        
        # Check symbol whitelist filtering
        if self.config.symbol_whitelist:
            symbol_str = f"symbol_{trade.symbol_id}"
            if symbol_str not in self.config.symbol_whitelist:
                self._metrics['filtered_by_symbol_whitelist'] += 1
                return True
        
        return False
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self._metrics = {
            'trades_read': 0,
            'bytes_read': 0,
            'read_errors': 0,
            'last_read_time': 0,
            'read_latency_sum': 0,
            'read_latency_count': 0,
            'trades_filtered': 0,
            'filtered_by_market_type': 0,
            'filtered_by_symbol_whitelist': 0
        }
    
    def get_config(self) -> HotSpineConfig:
        """Get current configuration"""
        return self.config


class HotSpineRuntime:
    """
    btq_live_runtime integration with HotSpine reader
    
    This class provides the core runtime for consuming HotSpine data
    in live trading scenarios with backtrader integration.
    
    Key architectural points:
    - HotSpine handles live trading data ingestion (NOT SQL)
    - SQL is used only for long-term storage, replay, analytics, and debugging
    - Data flow: HotSpine -> Strategy -> SQL (asynchronous)
    
    Enhanced with:
    - Centralized configuration management
    - Enhanced error handling and recovery
    - Performance monitoring
    - Symbol management
    """
    
    def __init__(self, strategy_cls,
                 config: Optional[HotSpineConfig] = None,
                 sql_config: Optional[MSSQLConfig] = None,
                 exclusive_hotswap_mode: bool = False):
        """
        Initialize the live runtime with HotSpine reader
        
        ARCHITECTURE NOTE: This constructor demonstrates clean separation:
        - HotSpineReader handles live trading data (shared memory)
        - HotSpineSQLIntegration handles long-term storage (SQL)
        - These are completely separate components with distinct responsibilities
        
        Args:
            strategy_cls: Backtrader strategy class to use for trading
            config: HotSpine configuration. If None, uses default configuration.
            sql_config: Optional SQL Server configuration for storage
            exclusive_hotswap_mode: Enable exclusive hotswap mode for data processing
        
        Raises:
            ValueError: If conflicting configuration options are provided
        """
        # Load configuration
        self.config = config or HotSpineConfig()
        
        # Validate configuration
        self._validate_configuration(self.config.enable_sql_storage, exclusive_hotswap_mode)
        
        self.strategy_cls = strategy_cls
        self.reader = HotSpineReader(self.config)
        self._running = False
        self._strategy_instance = None
        
        # Configuration options
        self.enable_sql_storage = self.config.enable_sql_storage
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
        
        # Performance metrics
        self._runtime_metrics = {
            'trades_processed': 0,
            'strategy_errors': 0,
            'start_time': 0,
            'last_trade_time': 0
        }
        
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
        Handle incoming trade data with enhanced error handling and monitoring
        
        ARCHITECTURE NOTE: This method demonstrates the key architectural principle:
        - HotSpine data flows to strategy FIRST (live trading)
        - SQL storage happens SECOND (asynchronous, non-blocking)
        - SQL is NOT in the hot path for trading decisions
        - SQL not in hot path: trading decisions are made before SQL storage
        
        Enhanced with:
        - Performance monitoring
        - Error handling and recovery
        - Metrics collection
        """
        try:
            # Update runtime metrics
            self._runtime_metrics['trades_processed'] += 1
            self._runtime_metrics['last_trade_time'] = time.time()
            
            if self._runtime_metrics['trades_processed'] == 1:
                self._runtime_metrics['start_time'] = self._runtime_metrics['last_trade_time']
            
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
                    self._runtime_metrics['strategy_errors'] += 1
                    
        except Exception as e:
            logger.error(f"Error processing trade in strategy: {e}")
            self._runtime_metrics['strategy_errors'] += 1
            
            # Check if reader is still healthy
            if not self.reader.is_healthy():
                logger.warning("Reader became unhealthy during trade processing")
    
    def run(self, batch_mode: Optional[bool] = None):
        """
        Run the live trading runtime with enhanced configuration
        
        Args:
            batch_mode: If True, use batch reading for higher throughput
                       If False, use single trade polling for lower latency
                       If None, uses configuration setting
        """
        if self._running:
            logger.info("Runtime is already running")
            return
            
        # Use configured batch mode if not specified
        if batch_mode is None:
            batch_mode = self.config.batch_mode
        
        self._running = True
        self._initialize_strategy()
        
        logger.info(f"Starting btq_live_runtime with HotSpine reader (batch_mode={batch_mode})")
        logger.info(f"Connected to shared memory: {self.reader.shm_name}")
        
        try:
            if batch_mode:
                self._run_batch_mode()
            else:
                self._run_single_mode()
        except KeyboardInterrupt:
            logger.info("\nShutting down btq_live_runtime...")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            raise
        finally:
            self._running = False
            self.reader.close()
            
            # Clean up SQL storage
            if self.sql_integration:
                self.sql_integration.stop_async_storage()
            
            # Log final metrics
            self._log_final_metrics()
    
    def _log_final_metrics(self):
        """Log final performance metrics"""
        runtime = self._runtime_metrics['last_trade_time'] - self._runtime_metrics['start_time']
        if runtime > 0:
            trades_per_second = self._runtime_metrics['trades_processed'] / runtime
            logger.info(f"Runtime completed: {self._runtime_metrics['trades_processed']} trades in {runtime:.2f}s "
                       f"({trades_per_second:.1f} trades/sec)")
        
        if self._runtime_metrics['strategy_errors'] > 0:
            logger.warning(f"Encountered {self._runtime_metrics['strategy_errors']} strategy errors")
    
    def get_runtime_metrics(self) -> Dict[str, Any]:
        """
        Get runtime performance metrics
        
        Returns:
            Dictionary containing runtime metrics
        """
        metrics = self._runtime_metrics.copy()
        
        # Add derived metrics
        if metrics['trades_processed'] > 0 and metrics['start_time'] > 0:
            runtime = metrics['last_trade_time'] - metrics['start_time']
            if runtime > 0:
                metrics['trades_per_second'] = metrics['trades_processed'] / runtime
            else:
                metrics['trades_per_second'] = 0
        else:
            metrics['trades_per_second'] = 0
        
        # Add reader metrics
        metrics['reader_metrics'] = self.reader.get_metrics()
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status
        
        Returns:
            Dictionary containing health status information
        """
        return {
            'runtime_healthy': self._running,
            'reader_healthy': self.reader.is_healthy(),
            'sql_healthy': self.sql_integration.is_healthy() if self.sql_integration else True,
            'trades_processed': self._runtime_metrics['trades_processed'],
            'strategy_errors': self._runtime_metrics['strategy_errors']
        }
    
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


def create_hotspine_runtime(strategy_cls,
                             config: Optional[HotSpineConfig] = None,
                             sql_config: Optional[MSSQLConfig] = None,
                             exclusive_hotswap_mode: bool = False):
    """
    Factory function to create a HotSpine runtime instance
    
    Args:
        strategy_cls: Backtrader strategy class
        config: HotSpine configuration. If None, uses default configuration.
        sql_config: Optional SQL Server configuration for storage
        exclusive_hotswap_mode: Enable exclusive hotswap mode for data processing
        
    Returns:
        HotSpineRuntime instance
    """
    return HotSpineRuntime(strategy_cls, config, sql_config, exclusive_hotswap_mode)


def create_hotspine_runtime_legacy(strategy_cls, shm_name: str = "/btquant_hotspine",
                                   sql_config: Optional[MSSQLConfig] = None,
                                   enable_sql_storage: bool = True,
                                   exclusive_hotswap_mode: bool = False):
    """
    Legacy factory function for backward compatibility
    
    Args:
        strategy_cls: Backtrader strategy class
        shm_name: Shared memory segment name
        sql_config: Optional SQL Server configuration for storage
        enable_sql_storage: Whether to enable SQL storage for long-term persistence
        exclusive_hotswap_mode: Enable exclusive hotswap mode for data processing
        
    Returns:
        HotSpineRuntime instance
    """
    # Create legacy configuration
    config = HotSpineConfig()
    config.shm_name = shm_name
    config.enable_sql_storage = enable_sql_storage
    
    return HotSpineRuntime(strategy_cls, config, sql_config, exclusive_hotswap_mode)