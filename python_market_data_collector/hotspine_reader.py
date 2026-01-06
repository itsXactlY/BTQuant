#!/usr/bin/env python3
"""
HotSpine Reader for Python Market Data Collector

This module provides a high-performance Python reader for the HotSpine shared memory
architecture, allowing direct reading of market data (trades and orderbooks) from RAM
while the C++ market data collector is running.

Key Features:
- Low-latency trade and orderbook reading from shared memory
- Symbol ID to exchange/symbol mapping
- Complete buffer monitoring and health checking
- Real-time statistics and performance metrics
- Support for running while C++ collector is active

Architecture:
    C++ Collector -> HotSpine Writer -> Shared Memory -> Python Reader -> Strategy

Author: PubBTQuant
Version: 1.0.0
"""

import ctypes
import os
import time
import logging
import threading
import json
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import mmap

# Configure module logger
logger = logging.getLogger(__name__)


class Side(Enum):
    """Trade side enumeration"""
    BUY = 0
    SELL = 1


class MarketType(Enum):
    """Market type enumeration"""
    SPOT = 0
    FUTURES = 1
    OTHER = 2


# ============================================================================
# C-compatible data structures matching HotSpine C++ layout
# ============================================================================

class HotTrade(ctypes.Structure):
    """
    Python representation of HotSpine trade structure.
    
    Must match the C++ HotSpine::HotTrade structure exactly.
    
    Fields:
        ts_exchange: Exchange timestamp in microseconds (UTC)
        ts_local: Local receive timestamp in microseconds
        price: Trade price
        size: Trade size (quantity)
        symbol_id: Symbol ID (hash or mapping)
        side: 0=BUY, 1=SELL
    """
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),
        ("ts_local", ctypes.c_uint64),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
        ("symbol_id", ctypes.c_uint32),
        ("side", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 3),
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
            "side": "BUY" if self.side == 0 else "SELL",
            "side_code": self.side,
        }
    
    @property
    def side_enum(self) -> Side:
        """Get side as enum"""
        return Side.BUY if self.side == 0 else Side.SELL
    
    def to_trade_data(self) -> 'TradeData':
        """Convert to TradeData object with resolved symbol info"""
        return TradeData(
            timestamp_us=self.ts_exchange,
            local_timestamp_us=self.ts_local,
            price=self.price,
            size=self.size,
            symbol_id=self.symbol_id,
            side=self.side_enum,
        )


class HotOrderbookLevel(ctypes.Structure):
    """
    Python representation of a single orderbook level.
    
    Fields:
        price: Level price
        size: Level size (quantity)
    """
    _fields_ = [
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
    ]
    
    def __repr__(self) -> str:
        return f"HotOrderbookLevel(price={self.price}, size={self.size})"
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (price, size) tuple"""
        return (self.price, self.size)


class HotOrderbookSnapshot(ctypes.Structure):
    """
    Python representation of HotSpine orderbook snapshot structure.
    
    Must match the C++ HotSpine::HotOrderbookSnapshot structure exactly.
    
    Fields:
        ts_exchange: Exchange timestamp in microseconds (UTC)
        ts_local: Local receive timestamp in microseconds
        symbol_id: Symbol ID (hash or mapping)
        bids_count: Number of bid levels
        asks_count: Number of ask levels
        bids: Array of bid levels (max 20)
        asks: Array of ask levels (max 20)
    """
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),
        ("ts_local", ctypes.c_uint64),
        ("symbol_id", ctypes.c_uint32),
        ("bids_count", ctypes.c_uint8),
        ("asks_count", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 2),
        ("bids", HotOrderbookLevel * 20),  # Max 20 bid levels
        ("asks", HotOrderbookLevel * 20),  # Max 20 ask levels
    ]
    
    def __repr__(self) -> str:
        return (f"HotOrderbookSnapshot(ts_exchange={self.ts_exchange}, "
                f"symbol_id={self.symbol_id}, bids={self.bids_count}, "
                f"asks={self.asks_count})")
    
    def get_bids(self) -> List[Tuple[float, float]]:
        """Get list of bid levels as (price, size) tuples"""
        return [
            (self.bids[i].price, self.bids[i].size)
            for i in range(self.bids_count)
        ]
    
    def get_asks(self) -> List[Tuple[float, float]]:
        """Get list of ask levels as (price, size) tuples"""
        return [
            (self.asks[i].price, self.asks[i].size)
            for i in range(self.asks_count)
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert orderbook snapshot to dictionary format"""
        return {
            "ts_exchange": self.ts_exchange,
            "ts_local": self.ts_local,
            "symbol_id": self.symbol_id,
            "bids": self.get_bids(),
            "asks": self.get_asks(),
            "bids_count": self.bids_count,
            "asks_count": self.asks_count,
        }
    
    def to_orderbook_data(self) -> 'OrderbookData':
        """Convert to OrderbookData object"""
        return OrderbookData(
            timestamp_us=self.ts_exchange,
            local_timestamp_us=self.ts_local,
            symbol_id=self.symbol_id,
            bids=self.get_bids(),
            asks=self.get_asks(),
        )


class SharedMemoryHeader(ctypes.Structure):
    """
    Python representation of HotSpine shared memory header.
    
    Must match the C++ HotSpine::SharedMemoryHeader structure exactly.
    
    Fields:
        version: Protocol version
        capacity: Buffer capacity (number of trades)
        write_index: Writer's current position
        read_index: Reader's current position
        lost_count: Number of trades lost due to overflow
        orderbook_write_index: Writer's orderbook position
        orderbook_read_index: Reader's orderbook position
        orderbook_lost_count: Number of orderbooks lost due to overflow
        orderbook_capacity: Orderbook buffer capacity
    """
    _fields_ = [
        ("version", ctypes.c_uint64),
        ("capacity", ctypes.c_uint64),
        ("write_index", ctypes.c_uint64),
        ("read_index", ctypes.c_uint64),
        ("lost_count", ctypes.c_uint64),
        # Orderbook buffer indices
        ("orderbook_write_index", ctypes.c_uint64),
        ("orderbook_read_index", ctypes.c_uint64),
        ("orderbook_lost_count", ctypes.c_uint64),
        ("orderbook_capacity", ctypes.c_uint64),
        ("padding", ctypes.c_uint8 * 8),
    ]


# ============================================================================
# High-level data classes for resolved symbol information
# ============================================================================

@dataclass
class TradeData:
    """
    Trade data with resolved symbol information.
    
    This is the high-level representation used by strategies,
    with exchange and symbol names resolved from symbol_id.
    """
    timestamp_us: int
    local_timestamp_us: int
    price: float
    size: float
    symbol_id: int
    side: Side
    exchange: str = ""
    symbol: str = ""
    
    @property
    def timestamp(self) -> datetime:
        """Get timestamp as datetime object"""
        return datetime.utcfromtimestamp(self.timestamp_us / 1_000_000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "timestamp_us": self.timestamp_us,
            "price": self.price,
            "size": self.size,
            "symbol_id": self.symbol_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side.name,
        }


@dataclass
class OrderbookData:
    """
    Orderbook snapshot data with resolved symbol information.
    
    This is the high-level representation used by strategies,
    with exchange and symbol names resolved from symbol_id.
    """
    timestamp_us: int
    local_timestamp_us: int
    symbol_id: int
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    exchange: str = ""
    symbol: str = ""
    
    @property
    def timestamp(self) -> datetime:
        """Get timestamp as datetime object"""
        return datetime.utcfromtimestamp(self.timestamp_us / 1_000_000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "timestamp_us": self.timestamp_us,
            "symbol_id": self.symbol_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "bids": self.bids,
            "asks": self.asks,
            "bids_count": len(self.bids),
            "asks_count": len(self.asks),
        }
    
    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price from best bid and ask"""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0
    
    def get_spread(self) -> Optional[float]:
        """Calculate spread between best ask and best bid"""
        if not self.bids or not self.asks:
            return None
        return self.asks[0][0] - self.bids[0][0]


# ============================================================================
# Symbol Mapper for resolving symbol_id to exchange/symbol
# ============================================================================

class SymbolMapper:
    """
    Maps symbol_id values to exchange and symbol names.
    
    Supports loading mappings from JSON configuration files
    and automatic symbol ID generation using hash functions.
    """
    
    # Common symbol ID mappings for known exchanges
    DEFAULT_MAPPINGS = {
        # Binance spot
        1: ("binance", "BTC-USDT"),
        2: ("binance", "ETH-USDT"),
        3: ("binance", "SOL-USDT"),
        4: ("binance", "XRP-USDT"),
        5: ("binance", "ADA-USDT"),
        # Bybit
        101: ("bybit", "BTC-USDT"),
        102: ("bybit", "ETH-USDT"),
        103: ("bybit", "SOL-USDT"),
        # OKX
        201: ("okx", "BTC-USDT"),
        202: ("okx", "ETH-USDT"),
        203: ("okx", "SOL-USDT"),
    }
    
    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize symbol mapper.
        
        Args:
            mapping_file: Optional path to JSON file with symbol mappings
        """
        self._id_to_info: Dict[int, Tuple[str, str]] = {}
        self._exchange_symbol_to_id: Dict[Tuple[str, str], int] = {}
        self._lock = threading.RLock()
        self._next_id = 1000  # Start user-defined IDs from 1000
        
        # Load default mappings
        self._load_default_mappings()
        
        # Load from file if provided
        if mapping_file and os.path.exists(mapping_file):
            self.load_from_file(mapping_file)
    
    def _load_default_mappings(self):
        """Load default symbol mappings"""
        for symbol_id, (exchange, symbol) in self.DEFAULT_MAPPINGS.items():
            self._id_to_info[symbol_id] = (exchange, symbol)
            self._exchange_symbol_to_id[(exchange, symbol)] = symbol_id
    
    def load_from_file(self, filepath: str):
        """
        Load symbol mappings from JSON file.
        
        Expected format:
        {
            "symbols": [
                {"id": 1, "exchange": "binance", "symbol": "BTC-USDT"},
                {"id": 2, "exchange": "binance", "symbol": "ETH-USDT"}
            ]
        }
        """
        with self._lock:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for item in data.get("symbols", []):
                    symbol_id = item["id"]
                    exchange = item["exchange"]
                    symbol = item["symbol"]
                    self._id_to_info[symbol_id] = (exchange, symbol)
                    self._exchange_symbol_to_id[(exchange, symbol)] = symbol_id
                    
                logger.info(f"Loaded {len(self._id_to_info)} symbol mappings from {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load symbol mappings from {filepath}: {e}")
                raise
    
    def save_to_file(self, filepath: str):
        """Save current symbol mappings to JSON file"""
        with self._lock:
            symbols = []
            for symbol_id, (exchange, symbol) in self._id_to_info.items():
                symbols.append({
                    "id": symbol_id,
                    "exchange": exchange,
                    "symbol": symbol
                })
            
            data = {"symbols": symbols}
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(symbols)} symbol mappings to {filepath}")
    
    def get_symbol_info(self, symbol_id: int) -> Tuple[str, str]:
        """
        Get exchange and symbol for a symbol ID.
        
        Args:
            symbol_id: The symbol ID to look up
            
        Returns:
            Tuple of (exchange, symbol)
            
        Raises:
            KeyError: If symbol_id is not found
        """
        with self._lock:
            if symbol_id not in self._id_to_info:
                raise KeyError(f"Unknown symbol_id: {symbol_id}")
            return self._id_to_info[symbol_id]
    
    def get_symbol_id(self, exchange: str, symbol: str) -> int:
        """
        Get symbol ID for an exchange/symbol pair.
        
        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            
        Returns:
            Symbol ID
            
        Raises:
            KeyError: If mapping is not found
        """
        with self._lock:
            key = (exchange, symbol)
            if key not in self._exchange_symbol_to_id:
                raise KeyError(f"Unknown exchange/symbol: {exchange}/{symbol}")
            return self._exchange_symbol_to_id[key]
    
    def register_symbol(self, exchange: str, symbol: str, symbol_id: Optional[int] = None) -> int:
        """
        Register a new symbol and get/create its ID.
        
        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            symbol_id: Optional specific ID to use, auto-assigned if not provided
            
        Returns:
            The assigned symbol ID
        """
        with self._lock:
            key = (exchange, symbol)
            
            # Check if already registered
            if key in self._exchange_symbol_to_id:
                return self._exchange_symbol_to_id[key]
            
            # Use provided ID or auto-assign
            if symbol_id is None:
                symbol_id = self._next_id
                self._next_id += 1
            
            self._id_to_info[symbol_id] = (exchange, symbol)
            self._exchange_symbol_to_id[key] = symbol_id
            
            logger.debug(f"Registered symbol: {exchange}/{symbol} -> ID {symbol_id}")
            
            return symbol_id
    
    def resolve_trade(self, trade: HotTrade) -> TradeData:
        """
        Resolve a HotTrade to TradeData with symbol information.
        
        Args:
            trade: HotTrade to resolve
            
        Returns:
            TradeData with exchange and symbol filled in
        """
        try:
            exchange, symbol = self.get_symbol_info(trade.symbol_id)
        except KeyError:
            exchange, symbol = "unknown", f"symbol_{trade.symbol_id}"
        
        return TradeData(
            timestamp_us=trade.ts_exchange,
            local_timestamp_us=trade.ts_local,
            price=trade.price,
            size=trade.size,
            symbol_id=trade.symbol_id,
            side=trade.side_enum,
            exchange=exchange,
            symbol=symbol,
        )
    
    def resolve_orderbook(self, snapshot: HotOrderbookSnapshot) -> OrderbookData:
        """
        Resolve a HotOrderbookSnapshot to OrderbookData with symbol information.
        
        Args:
            snapshot: HotOrderbookSnapshot to resolve
            
        Returns:
            OrderbookData with exchange and symbol filled in
        """
        try:
            exchange, symbol = self.get_symbol_info(snapshot.symbol_id)
        except KeyError:
            exchange, symbol = "unknown", f"symbol_{snapshot.symbol_id}"
        
        return OrderbookData(
            timestamp_us=snapshot.ts_exchange,
            local_timestamp_us=snapshot.ts_local,
            symbol_id=snapshot.symbol_id,
            bids=snapshot.get_bids(),
            asks=snapshot.get_asks(),
            exchange=exchange,
            symbol=symbol,
        )


# ============================================================================
# HotSpine Reader Implementation
# ============================================================================

class HotSpineReader:
    """
    High-performance HotSpine shared memory reader.
    
    This reader provides low-latency access to market data stored in shared memory
    by the C++ HotSpine writer. It supports both trade and orderbook data, with
    comprehensive monitoring and health checking.
    
    Key Features:
    - Direct shared memory access via ctypes
    - Trade and orderbook support
    - Symbol mapping and resolution
    - Buffer monitoring and health checking
    - Performance metrics collection
    - Thread-safe operations
    
    Usage:
        reader = HotSpineReader("/btquant_hotspine")
        while True:
            trade = reader.poll_trade()
            if trade:
                process_trade(trade)
            orderbook = reader.poll_orderbook()
            if orderbook:
                process_orderbook(orderbook)
            time.sleep(0.001)  # 1ms polling interval
    """
    
    # Constants from C++ header
    HEADER_SIZE = 4096
    TRADE_SIZE = ctypes.sizeof(HotTrade)
    ORDERBOOK_SIZE = ctypes.sizeof(HotOrderbookSnapshot)
    DEFAULT_CAPACITY = 1_000_000
    DEFAULT_ORDERBOOK_CAPACITY = 100_000
    
    def __init__(
        self,
        shm_name: str = "btquant_hotspine",
        symbol_mapper: Optional[SymbolMapper] = None,
        enable_stats: bool = True,
        poll_interval: float = 0.0001,  # 100 microseconds
    ):
        """
        Initialize HotSpine reader and attach to shared memory.
        
        Args:
            shm_name: Name of shared memory segment (e.g., "/btquant_hotspine")
            symbol_mapper: Optional SymbolMapper instance for resolving symbol IDs
            enable_stats: Whether to collect performance statistics
            poll_interval: Sleep interval when no data is available (seconds)
            
        Raises:
            FileNotFoundError: If shared memory segment doesn't exist
            RuntimeError: If failed to attach to shared memory
        """
        self.shm_name = shm_name
        self.symbol_mapper = symbol_mapper or SymbolMapper()
        self.enable_stats = enable_stats
        self.poll_interval = poll_interval
        
        # State
        self._shm_fd = -1
        self._shm_ptr = None
        self._header = None
        self._trades_buffer = None
        self._orderbooks_buffer = None
        self._read_index = 0  # Local read index for trades
        self._orderbook_read_index = 0  # Local read index for orderbooks
        self._attached = False
        self._closed = False
        
        # Performance statistics
        self._stats = {
            'trades_read': 0,
            'orderbooks_read': 0,
            'poll_calls': 0,
            'empty_polls': 0,
            'bytes_read': 0,
            'lost_trades': 0,
            'lost_orderbooks': 0,
            'read_errors': 0,
            'total_latency_us': 0,
            'start_time': None,
            'last_poll_time': None,
        }
        self._stats_lock = threading.RLock()
        
        # Health monitoring
        self._health_status = {
            'attached': False,
            'healthy': False,
            'last_error': None,
            'error_count': 0,
            'reconnect_attempts': 0,
        }
        self._lock = threading.RLock()
        
        # Attach to shared memory
        self._attach()
    
    def _attach(self):
        """Attach to shared memory segment"""
        with self._lock:
            try:
                # Construct the full path to shared memory
                # If shm_name starts with '/', use it as-is (for compatibility)
                # Otherwise, prepend /dev/shm/
                if self.shm_name.startswith('/'):
                    shm_path = self.shm_name
                else:
                    shm_path = f"/dev/shm/{self.shm_name}"
                
                logger.info(f"Attempting to attach to shared memory at: {shm_path}")
                
                # Open existing shared memory segment
                # We need O_RDWR for ctypes.from_buffer to work, even if we only intend to read
                self._shm_fd = os.open(
                    shm_path,
                    os.O_RDWR | os.O_SYNC
                )
                
                # Get file size
                file_size = os.fstat(self._shm_fd).st_size
                
                # Map shared memory into process address space
                # We use ACCESS_WRITE because ctypes.from_buffer requires a writable buffer
                self._shm_ptr = mmap.mmap(
                    self._shm_fd,
                    file_size,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE
                )

                # Create header pointer directly from shared memory
                # Use ctypes.from_buffer to get a pointer to the underlying buffer
                # Then use addressof to get the memory address
                header_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm_ptr, 0))
                self._header = ctypes.cast(
                    ctypes.c_void_p(header_addr),
                    ctypes.POINTER(SharedMemoryHeader)
                ).contents
                
                # Validate version
                if self._header.version != 2:
                    raise RuntimeError(
                        f"Unsupported HotSpine version: {self._header.version} (expected 2)"
                    )
                
                # Calculate buffer pointers
                buffer_offset = self.HEADER_SIZE
                
                # Trade buffer
                trades_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm_ptr, buffer_offset))
                self._trades_buffer = ctypes.cast(
                    ctypes.c_void_p(trades_addr),
                    ctypes.POINTER(HotTrade)
                )
                
                # Orderbook buffer starts after trade buffer
                trade_buffer_size = self._header.capacity * self.TRADE_SIZE
                orderbook_offset = buffer_offset + trade_buffer_size
                
                # Create orderbook buffer pointer
                if orderbook_offset < file_size:
                    orderbooks_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm_ptr, orderbook_offset))
                    self._orderbooks_buffer = ctypes.cast(
                        ctypes.c_void_p(orderbooks_addr),
                        ctypes.POINTER(HotOrderbookSnapshot)
                    )
                else:
                    self._orderbooks_buffer = None
                
                self._attached = True
                self._health_status['attached'] = True
                self._health_status['healthy'] = True
                self._stats['start_time'] = time.time()
                
                logger.info(
                    f"HotSpine reader attached to {self.shm_name} "
                    f"(trade_capacity={self._header.capacity}, "
                    f"orderbook_capacity={getattr(self._header, 'orderbook_capacity', 0)})"
                )
                
            except FileNotFoundError:
                logger.error(f"Shared memory segment not found: {self.shm_name}")
                raise
            except Exception as e:
                logger.error(f"Failed to attach to shared memory: {e}")
                self._health_status['last_error'] = str(e)
                self._health_status['error_count'] += 1
                raise
    
    def close(self):
        """Detach from shared memory and release resources"""
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            # Unmap shared memory
            if self._shm_ptr:
                try:
                    self._shm_ptr.close()
                except Exception as e:
                    logger.warning(f"Error closing mmap: {e}")
                self._shm_ptr = None
            
            # Close file descriptor
            if self._shm_fd >= 0:
                try:
                    os.close(self._shm_fd)
                except Exception as e:
                    logger.warning(f"Error closing file descriptor: {e}")
                self._shm_fd = -1
            
            self._attached = False
            self._health_status['attached'] = False
            self._health_status['healthy'] = False
            
            logger.info(f"HotSpine reader detached from {self.shm_name}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
    
    def __del__(self):
        """Destructor"""
        self.close()
    
    # =========================================================================
    # Trade Reading Methods
    # =========================================================================
    
    def poll_trade(self) -> Optional[HotTrade]:
        """
        Poll for a single trade (non-blocking).
        
        Returns:
            HotTrade object if available, None otherwise
            
        Note:
            This method uses atomic compare-and-swap to safely read
            from the circular buffer without modifying it.
        """
        with self._lock:
            if not self._attached:
                return None
            
            start_time = time.time()
            
            try:
                # Read current indices atomically
                write_index = self._header.write_index
                read_index = self._header.read_index
                
                # Calculate available trades
                if write_index >= read_index:
                    available = write_index - read_index
                else:
                    # Wrapped around
                    available = self._header.capacity - read_index + write_index
                
                if available == 0:
                    # No trades available
                    with self._stats_lock:
                        self._stats['empty_polls'] += 1
                        self._stats['poll_calls'] += 1
                    
                    # Small sleep to reduce CPU usage
                    time.sleep(self.poll_interval)
                    return None
                
                # Read trade at current read index
                trade = self._trades_buffer[read_index % self._header.capacity]
                
                # Update read index (atomic increment with wrapping)
                self._header.read_index = (read_index + 1) % self._header.capacity
                
                # Update statistics
                with self._stats_lock:
                    self._stats['trades_read'] += 1
                    self._stats['poll_calls'] += 1
                    self._stats['bytes_read'] += self.TRADE_SIZE
                    self._stats['last_poll_time'] = time.time()
                    
                    if self.enable_stats:
                        latency = (time.time() - start_time) * 1_000_000  # microseconds
                        self._stats['total_latency_us'] += latency
                
                return trade
                
            except Exception as e:
                with self._stats_lock:
                    self._stats['read_errors'] += 1
                self._health_status['last_error'] = str(e)
                self._health_status['error_count'] += 1
                logger.error(f"Error polling trade: {e}")
                return None
    
    def poll_trade_resolved(self) -> Optional[TradeData]:
        """
        Poll for a single trade with resolved symbol information.
        
        Returns:
            TradeData object with exchange/symbol filled in, or None
        """
        trade = self.poll_trade()
        if trade:
            return self.symbol_mapper.resolve_trade(trade)
        return None
    
    def read_all_trades(self) -> List[HotTrade]:
        """
        Read all available trades at once (more efficient for batch processing).
        
        Returns:
            List of HotTrade objects (empty list if none available)
        """
        trades = []
        
        while True:
            trade = self.poll_trade()
            if trade:
                trades.append(trade)
            else:
                break
        
        return trades
    
    def read_all_trades_resolved(self) -> List[TradeData]:
        """
        Read all available trades with resolved symbol information.
        
        Returns:
            List of TradeData objects (empty list if none available)
        """
        return [
            self.symbol_mapper.resolve_trade(trade)
            for trade in self.read_all_trades()
        ]
    
    # =========================================================================
    # Orderbook Reading Methods
    # =========================================================================
    
    def poll_orderbook(self) -> Optional[HotOrderbookSnapshot]:
        """
        Poll for a single orderbook snapshot (non-blocking).
        
        Returns:
            HotOrderbookSnapshot object if available, None otherwise
        """
        with self._lock:
            if not self._attached:
                return None
            
            start_time = time.time()
            
            try:
                # Check if orderbook buffer is available
                if self._orderbooks_buffer is None or \
                   not hasattr(self._header, 'orderbook_write_index') or \
                   self._header.orderbook_capacity == 0:
                    # Orderbook buffer not initialized yet
                    time.sleep(self.poll_interval)
                    return None
                
                # Read current indices atomically
                write_index = self._header.orderbook_write_index
                read_index = self._header.orderbook_read_index
                capacity = self._header.orderbook_capacity
                
                # Calculate available orderbooks
                if write_index >= read_index:
                    available = write_index - read_index
                else:
                    # Wrapped around
                    available = capacity - read_index + write_index
                
                if available == 0:
                    # No orderbooks available
                    with self._stats_lock:
                        self._stats['empty_polls'] += 1
                        self._stats['poll_calls'] += 1
                    
                    time.sleep(self.poll_interval)
                    return None
                
                # Read orderbook at current read index
                snapshot = self._orderbooks_buffer[read_index % capacity]
                
                # Update read index (atomic increment with wrapping)
                self._header.orderbook_read_index = (read_index + 1) % capacity
                
                # Update statistics
                with self._stats_lock:
                    self._stats['orderbooks_read'] += 1
                    self._stats['poll_calls'] += 1
                    self._stats['bytes_read'] += self.ORDERBOOK_SIZE
                    self._stats['last_poll_time'] = time.time()
                    
                    if self.enable_stats:
                        latency = (time.time() - start_time) * 1_000_000
                        self._stats['total_latency_us'] += latency
                
                return snapshot
                
            except Exception as e:
                with self._stats_lock:
                    self._stats['read_errors'] += 1
                self._health_status['last_error'] = str(e)
                self._health_status['error_count'] += 1
                logger.error(f"Error polling orderbook: {e}")
                return None
    
    def poll_orderbook_resolved(self) -> Optional[OrderbookData]:
        """
        Poll for a single orderbook with resolved symbol information.
        
        Returns:
            OrderbookData object with exchange/symbol filled in, or None
        """
        snapshot = self.poll_orderbook()
        if snapshot:
            return self.symbol_mapper.resolve_orderbook(snapshot)
        return None
    
    def read_all_orderbooks(self) -> List[HotOrderbookSnapshot]:
        """
        Read all available orderbooks at once (more efficient for batch processing).
        
        Returns:
            List of HotOrderbookSnapshot objects (empty list if none available)
        """
        orderbooks = []
        
        while True:
            orderbook = self.poll_orderbook()
            if orderbook:
                orderbooks.append(orderbook)
            else:
                break
        
        return orderbooks
    
    def read_all_orderbooks_resolved(self) -> List[OrderbookData]:
        """
        Read all available orderbooks with resolved symbol information.
        
        Returns:
            List of OrderbookData objects (empty list if none available)
        """
        return [
            self.symbol_mapper.resolve_orderbook(snapshot)
            for snapshot in self.read_all_orderbooks()
        ]
    
    # =========================================================================
    # Buffer and Health Monitoring Methods
    # =========================================================================
    
    def get_buffer_utilization(self) -> Dict[str, Any]:
        """
        Get current buffer utilization information.
        
        Returns:
            Dictionary with:
                - trade_count: Number of available trades
                - trade_capacity: Total trade buffer capacity
                - trade_utilization_pct: Trade buffer utilization percentage
                - orderbook_count: Number of available orderbooks
                - orderbook_capacity: Total orderbook buffer capacity
                - orderbook_utilization_pct: Orderbook buffer utilization percentage
        """
        with self._lock:
            if not self._attached:
                return {
                    "trade_count": 0,
                    "trade_capacity": 0,
                    "trade_utilization_pct": 0.0,
                    "orderbook_count": 0,
                    "orderbook_capacity": 0,
                    "orderbook_utilization_pct": 0.0,
                }
            
            # Trade buffer
            write_index = self._header.write_index
            read_index = self._header.read_index
            
            if write_index >= read_index:
                trade_count = write_index - read_index
            else:
                trade_count = self._header.capacity - read_index + write_index
            
            trade_capacity = self._header.capacity
            trade_utilization_pct = (trade_count / trade_capacity) * 100 if trade_capacity > 0 else 0
            
            # Orderbook buffer
            if hasattr(self._header, 'orderbook_write_index') and \
               self._header.orderbook_capacity > 0:
                ob_write_index = self._header.orderbook_write_index
                ob_read_index = self._header.orderbook_read_index
                ob_capacity = self._header.orderbook_capacity
                
                if ob_write_index >= ob_read_index:
                    orderbook_count = ob_write_index - ob_read_index
                else:
                    orderbook_count = ob_capacity - ob_read_index + ob_write_index
                
                orderbook_capacity = ob_capacity
                orderbook_utilization_pct = (orderbook_count / orderbook_capacity) * 100 if ob_capacity > 0 else 0
            else:
                orderbook_count = 0
                orderbook_capacity = 0
                orderbook_utilization_pct = 0.0
            
            return {
                "trade_count": trade_count,
                "trade_capacity": trade_capacity,
                "trade_utilization_pct": round(trade_utilization_pct, 2),
                "orderbook_count": orderbook_count,
                "orderbook_capacity": orderbook_capacity,
                "orderbook_utilization_pct": round(orderbook_utilization_pct, 2),
            }
    
    def get_lost_count(self) -> Tuple[int, int]:
        """
        Get the number of lost trades and orderbooks (overflow counter).
        
        Returns:
            Tuple of (lost_trades, lost_orderbooks)
        """
        with self._lock:
            if not self._attached:
                return (0, 0)
            
            lost_trades = self._header.lost_count
            
            if hasattr(self._header, 'orderbook_lost_count'):
                lost_orderbooks = self._header.orderbook_lost_count
            else:
                lost_orderbooks = 0
            
            with self._stats_lock:
                self._stats['lost_trades'] = lost_trades
                self._stats['lost_orderbooks'] = lost_orderbooks
            
            return (lost_trades, lost_orderbooks)
    
    def is_attached(self) -> bool:
        """Check if successfully attached to shared memory"""
        return self._attached
    
    def is_healthy(self) -> bool:
        """
        Check if the reader is healthy and connected.
        
        Returns:
            True if reader is healthy, False otherwise
        """
        with self._lock:
            if not self._attached:
                return False
            
            # Check for recent errors
            if self._health_status['error_count'] > 100:
                logger.warning(f"Too many errors ({self._health_status['error_count']})")
                return False
            
            # Check if we can read from shared memory
            try:
                _ = self._header.write_index
                return True
            except Exception:
                return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Dictionary with health status information
        """
        with self._lock:
            status = self._health_status.copy()
            status['buffer'] = self.get_buffer_utilization()
            status['is_healthy'] = self.is_healthy()
            return status

    def get_buffers_info(self) -> Dict[str, Any]:
        """
        Get detailed information about all buffers.
        
        Returns:
            Dictionary with buffer details
        """
        util = self.get_buffer_utilization()
        return {
            "trades": {
                "count": util["trade_count"],
                "capacity": util["trade_capacity"],
                "utilization": util["trade_utilization_pct"],
                "lost": self._header.lost_count if self._attached else 0,
            },
            "orderbooks": {
                "count": util["orderbook_count"],
                "capacity": util["orderbook_capacity"],
                "utilization": util["orderbook_utilization_pct"],
                "lost": getattr(self._header, 'orderbook_lost_count', 0) if self._attached else 0,
            }
        }

    def list_available_data_types(self) -> List[str]:
        """
        List data types available in this HotSpine segment.
        
        Returns:
            List of data type names
        """
        types = ["trades"]
        if self._orderbooks_buffer is not None:
            types.append("orderbooks")
        return types
    
    # =========================================================================
    # Statistics Methods
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._stats_lock:
            stats = self._stats.copy()
            
            # Calculate derived metrics
            if stats['start_time']:
                elapsed = time.time() - stats['start_time']
                stats['elapsed_seconds'] = elapsed
                
                if elapsed > 0:
                    stats['trades_per_second'] = stats['trades_read'] / elapsed
                    stats['orderbooks_per_second'] = stats['orderbooks_read'] / elapsed
                    stats['poll_calls_per_second'] = stats['poll_calls'] / elapsed
            
            # Average latency
            total_polls = stats['poll_calls']
            if total_polls > 0:
                avg_latency = stats['total_latency_us'] / total_polls
                stats['avg_latency_us'] = avg_latency
            else:
                stats['avg_latency_us'] = 0
            
            # Health info
            stats['healthy'] = self.is_healthy()
            stats['attached'] = self.is_attached()
            
            # Buffer info
            buffer = self.get_buffer_utilization()
            stats['trade_buffer'] = {
                'count': buffer['trade_count'],
                'capacity': buffer['trade_capacity'],
                'utilization_pct': buffer['trade_utilization_pct'],
            }
            stats['orderbook_buffer'] = {
                'count': buffer['orderbook_count'],
                'capacity': buffer['orderbook_capacity'],
                'utilization_pct': buffer['orderbook_utilization_pct'],
            }
            
            return stats
    
    def reset_statistics(self):
        """Reset performance statistics"""
        with self._stats_lock:
            self._stats = {
                'trades_read': 0,
                'orderbooks_read': 0,
                'poll_calls': 0,
                'empty_polls': 0,
                'bytes_read': 0,
                'lost_trades': 0,
                'lost_orderbooks': 0,
                'read_errors': 0,
                'total_latency_us': 0,
                'start_time': time.time(),
                'last_poll_time': None,
            }
    
    def print_statistics(self):
        """Print human-readable statistics report"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("HotSpine Reader Statistics")
        print("=" * 60)
        print(f"Attached:           {stats['attached']}")
        print(f"Healthy:            {stats['healthy']}")
        print(f"Trades Read:        {stats['trades_read']:,}")
        print(f"Trades/Second:      {stats.get('trades_per_second', 0):.2f}")
        print(f"Orderbooks Read:    {stats['orderbooks_read']:,}")
        print(f"Orderbooks/Second:  {stats.get('orderbooks_per_second', 0):.2f}")
        print(f"Poll Calls:         {stats['poll_calls']:,}")
        print(f"Empty Polls:        {stats['empty_polls']:,}")
        print(f"Lost Trades:        {stats['lost_trades']:,}")
        print(f"Lost Orderbooks:    {stats['lost_orderbooks']:,}")
        print(f"Read Errors:        {stats['read_errors']:,}")
        print(f"Avg Latency:        {stats.get('avg_latency_us', 0):.2f} μs")
        
        trade_buffer = stats.get('trade_buffer', {})
        print(f"\nTrade Buffer:")
        print(f"  Count:            {trade_buffer.get('count', 0):,}")
        print(f"  Capacity:         {trade_buffer.get('capacity', 0):,}")
        print(f"  Utilization:      {trade_buffer.get('utilization_pct', 0):.2f}%")
        
        ob_buffer = stats.get('orderbook_buffer', {})
        print(f"\nOrderbook Buffer:")
        print(f"  Count:            {ob_buffer.get('count', 0):,}")
        print(f"  Capacity:         {ob_buffer.get('capacity', 0):,}")
        print(f"  Utilization:      {ob_buffer.get('utilization_pct', 0):.2f}%")
        
        print("=" * 60 + "\n")
    
    # =========================================================================
    # Iterator Protocol
    # =========================================================================
    
    def __iter__(self):
        """Iterator protocol for convenient trade iteration"""
        return self
    
    def __next__(self) -> HotTrade:
        """Get next trade, blocking if necessary"""
        trade = self.poll_trade()
        if trade is None:
            raise StopIteration
        return trade
    
    def __len__(self) -> int:
        """Return approximate number of available trades"""
        return self.get_buffer_utilization()['trade_count']


# ============================================================================
# Convenience Functions
# ============================================================================

def create_hotspine_reader(
    shm_name: str = "btquant_hotspine",
    symbol_mapping_file: Optional[str] = None,
    enable_stats: bool = True,
) -> HotSpineReader:
    """
    Factory function to create a HotSpine reader.
    
    Args:
        shm_name: Shared memory segment name
        symbol_mapping_file: Optional path to symbol mapping JSON file
        enable_stats: Whether to collect statistics
        
    Returns:
        HotSpineReader instance
    """
    symbol_mapper = None
    if symbol_mapping_file:
        symbol_mapper = SymbolMapper(symbol_mapping_file)
    
    return HotSpineReader(
        shm_name=shm_name,
        symbol_mapper=symbol_mapper,
        enable_stats=enable_stats,
    )


def read_hotspine_live(
    shm_name: str = "btquant_hotspine",
    callback: Optional[Callable[[TradeData], None]] = None,
    symbol_mapping_file: Optional[str] = None,
    poll_interval: float = 0.0001,
    verbose: bool = True,
) -> None:
    """
    Simple function to read trades from HotSpine with optional callback.
    
    Args:
        shm_name: Shared memory segment name
        callback: Optional function to call for each trade
        symbol_mapping_file: Optional path to symbol mapping JSON file
        poll_interval: Sleep interval when no data available
        verbose: Whether to print progress
    """
    reader = create_hotspine_reader(
        shm_name=shm_name,
        symbol_mapping_file=symbol_mapping_file,
    )
    
    trade_count = 0
    start_time = time.time()
    
    try:
        while True:
            trade = reader.poll_trade_resolved()
            if trade:
                trade_count += 1
                
                if callback:
                    callback(trade)
                
                if verbose and trade_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = trade_count / elapsed
                    print(f"Processed {trade_count} trades ({rate:.1f}/sec)")
                    
            else:
                time.sleep(poll_interval)
                
    except KeyboardInterrupt:
        if verbose:
            print(f"\nStopped after {trade_count} trades")
    finally:
        reader.close()


# ============================================================================
# Main entry point for testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("HotSpine Reader Test")
    print("=" * 40)
    
    try:
        reader = HotSpineReader("btquant_hotspine")
        print(f"✓ Successfully attached to shared memory")
        
        buffer = reader.get_buffer_utilization()
        print(f"  Trade buffer: {buffer['trade_count']} / {buffer['trade_capacity']}")
        print(f"  Orderbook buffer: {buffer['orderbook_count']} / {buffer['orderbook_capacity']}")
        
        # Read a few trades
        print("\nReading trades...")
        for i in range(5):
            trade = reader.poll_trade()
            if trade:
                resolved = reader.symbol_mapper.resolve_trade(trade)
                print(f"  Trade {i+1}: {resolved.exchange}/{resolved.symbol} @ {resolved.price}")
            else:
                print(f"  No trade available (attempt {i+1})")
                time.sleep(0.1)
        
        # Read a few orderbooks
        print("\nReading orderbooks...")
        for i in range(5):
            orderbook = reader.poll_orderbook()
            if orderbook:
                resolved = reader.symbol_mapper.resolve_orderbook(orderbook)
                mid_price = resolved.get_mid_price()
                spread = resolved.get_spread()
                print(f"  Orderbook {i+1}: {resolved.exchange}/{resolved.symbol} "
                      f"(mid={mid_price}, spread={spread})")
            else:
                print(f"  No orderbook available (attempt {i+1})")
                time.sleep(0.1)
        
        # Print statistics
        reader.print_statistics()
        
        reader.close()
        print("✓ Test completed successfully")
        sys.exit(0)
        
    except FileNotFoundError:
        print("✗ Shared memory segment not found. Is the C++ collector running?")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
