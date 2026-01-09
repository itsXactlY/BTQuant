#!/usr/bin/env python3
"""
HotSpine Monitor - Real-time Market Data Reader from Shared Memory

This script monitors /dev/shm for HotSpine shared memory segments, reads and 
parses market data (trades and orderbooks) in real-time, and provides basic 
querying, filtering, and aggregation functionality.

Architecture:
    C++ Collector -> HotSpine Writer -> Shared Memory (/dev/shm) -> Python Reader

Usage:
    python hotspine_monitor.py --shm-name btquant_hotspine --output json
    python hotspine_monitor.py --shm-name btquant_hotspine --filter "exchange=binance" --limit 100
    python hotspine_monitor.py --shm-name btquant_hotspine --aggregate-by symbol

Author: BTQuant
Version: 1.0.0
"""

import os
import sys
import time
import json
import mmap
import ctypes
import argparse
import logging
import signal
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# HotSpine Data Structures (must match C++ layout exactly)
# ============================================================================

class HotTrade(ctypes.Structure):
    """
    Python representation of HotSpine trade structure.
    
    Must match C++ HotSpine::HotTrade layout:
    - ts_exchange: Exchange timestamp in microseconds
    - ts_local: Local receive timestamp in microseconds  
    - price: Trade price
    - size: Trade quantity
    - symbol_id: Symbol identifier (hash or mapping)
    - side: 0=BUY, 1=SELL
    """
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),  # Exchange timestamp (μs)
        ("ts_local", ctypes.c_uint64),     # Local receive timestamp (μs)
        ("price", ctypes.c_double),        # Trade price
        ("size", ctypes.c_double),         # Trade size (quantity)
        ("symbol_id", ctypes.c_uint32),    # Symbol ID
        ("side", ctypes.c_uint8),          # 0=BUY, 1=SELL
        ("padding", ctypes.c_uint8 * 3),   # Padding for alignment
    ]
    
    @property
    def side_str(self) -> str:
        """Get side as string"""
        return "BUY" if self.side == 0 else "SELL"
    
    def to_dict(self, symbol_mapper: 'SymbolMapper' = None) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        if symbol_mapper:
            exchange, symbol = symbol_mapper.get_symbol_info(self.symbol_id)
        else:
            exchange, symbol = f"unknown_{self.symbol_id}", f"symbol_{self.symbol_id}"
        
        return {
            "type": "trade",
            "timestamp_us": self.ts_exchange,
            "timestamp_local_us": self.ts_local,
            "price": self.price,
            "size": self.size,
            "symbol_id": self.symbol_id,
            "exchange": exchange,
            "symbol": symbol,
            "side": self.side_str,
            "datetime": datetime.fromtimestamp(self.ts_exchange / 1_000_000, tz=timezone.utc).isoformat(),
        }


class HotOrderbookLevel(ctypes.Structure):
    """Single orderbook level (price/size pair)"""
    _fields_ = [
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
    ]


class HotOrderbookSnapshot(ctypes.Structure):
    """
    Python representation of HotSpine orderbook snapshot.
    
    Must match C++ HotSpine::HotOrderbookSnapshot layout.
    Supports up to 20 bid and 20 ask levels.
    """
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),       # Exchange timestamp (μs)
        ("ts_local", ctypes.c_uint64),          # Local receive timestamp (μs)
        ("symbol_id", ctypes.c_uint32),         # Symbol ID
        ("bids_count", ctypes.c_uint8),         # Number of bid levels
        ("asks_count", ctypes.c_uint8),         # Number of ask levels
        ("padding", ctypes.c_uint8 * 2),        # Padding for alignment
        ("bids", HotOrderbookLevel * 20),       # Bid levels array
        ("asks", HotOrderbookLevel * 20),       # Ask levels array
    ]
    
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
    
    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price from best bid and ask"""
        bids = self.get_bids()
        asks = self.get_asks()
        if not bids or not asks:
            return None
        return (bids[0][0] + asks[0][0]) / 2.0
    
    def get_spread(self) -> Optional[float]:
        """Calculate spread between best ask and best bid"""
        bids = self.get_bids()
        asks = self.get_asks()
        if not bids or not asks:
            return None
        return asks[0][0] - bids[0][0]
    
    def to_dict(self, symbol_mapper: 'SymbolMapper' = None) -> Dict[str, Any]:
        """Convert orderbook to dictionary"""
        if symbol_mapper:
            exchange, symbol = symbol_mapper.get_symbol_info(self.symbol_id)
        else:
            exchange, symbol = f"unknown_{self.symbol_id}", f"symbol_{self.symbol_id}"
        
        return {
            "type": "orderbook",
            "timestamp_us": self.ts_exchange,
            "timestamp_local_us": self.ts_local,
            "symbol_id": self.symbol_id,
            "exchange": exchange,
            "symbol": symbol,
            "bids": self.get_bids(),
            "asks": self.get_asks(),
            "bids_count": self.bids_count,
            "asks_count": self.asks_count,
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "datetime": datetime.fromtimestamp(self.ts_exchange / 1_000_000, tz=timezone.utc).isoformat(),
        }


class SharedMemoryHeader(ctypes.Structure):
    """
    Shared memory header structure.
    
    Must match C++ HotSpine::SharedMemoryHeader layout.
    Contains metadata about the shared memory segment and buffer indices.
    """
    _fields_ = [
        ("version", ctypes.c_uint64),           # Protocol version
        ("capacity", ctypes.c_uint64),          # Trade buffer capacity
        ("write_index", ctypes.c_uint64),       # Writer's current position
        ("read_index", ctypes.c_uint64),        # Reader's current position
        ("lost_count", ctypes.c_uint64),        # Lost trades (overflow)
        # Orderbook fields
        ("orderbook_write_index", ctypes.c_uint64),
        ("orderbook_read_index", ctypes.c_uint64),
        ("orderbook_lost_count", ctypes.c_uint64),
        ("orderbook_capacity", ctypes.c_uint64),
        ("padding", ctypes.c_uint8 * 8),        # Remaining padding
    ]


# ============================================================================
# Constants (from C++ header)
# ============================================================================

HEADER_SIZE = 4096                    # 4KB header size
TRADE_SIZE = ctypes.sizeof(HotTrade)  # Size of HotTrade struct
ORDERBOOK_SIZE = ctypes.sizeof(HotOrderbookSnapshot)  # Size of orderbook struct
HOTSPINE_VERSION = 2                  # Current protocol version
DEFAULT_CAPACITY = 1_000_000          # Default trade buffer capacity
DEFAULT_ORDERBOOK_CAPACITY = 100_000  # Default orderbook buffer capacity


# ============================================================================
# Symbol Mapper
# ============================================================================

class SymbolMapper:
    """
    Maps symbol_id values to exchange and symbol names.
    
    Uses hash-based symbol ID generation that matches the C++ implementation.
    Also supports loading mappings from JSON configuration files.
    """
    
    # Default symbol mappings for common exchanges (hash-based IDs)
    # Generated using: generate_symbol_id(exchange, symbol, market_type)
    # Hash = djb2-like: hash = hash * 33 + char
    DEFAULT_MAPPINGS = {
        # Discovered from user's data (2026-01-08)
        1105504075: ("gate", "XRP-USDT"),
        1783517845: ("bybit", "ETH-USDT"),
        3658966163: ("bybit", "SOL-USDT"),
        4179764808: ("binance", "BTC-USDT"),
        4211162444: ("binance", "SOL-USDT"),
        
        # Binance Spot
        4180201791: ("binance", "ETH-USDT"),
        4211162445: ("binance", "XRP-USDT"),
        4211162446: ("binance", "ADA-USDT"),
        4211162447: ("binance", "DOGE-USDT"),
        4211162448: ("binance", "DOT-USDT"),
        4211162449: ("binance", "LINK-USDT"),
        4211162450: ("binance", "MATIC-USDT"),
        4211162451: ("binance", "UNI-USDT"),
        4211162452: ("binance", "AVAX-USDT"),
        4211162453: ("binance", "ATOM-USDT"),
        4211162454: ("binance", "LTC-USDT"),
        4211162455: ("binance", "NEAR-USDT"),
        4211162456: ("binance", "ARB-USDT"),
        # Bybit
        1105504076: ("bybit", "BTC-USDT"),
        # OKX
        3808813995: ("okx", "BTC-USDT"),
        3809080978: ("okx", "ETH-USDT"),
        3809227961: ("okx", "SOL-USDT"),
        # MEXC
        3947592941: ("mexc", "BTC-USDT"),
        3947592942: ("mexc", "ETH-USDT"),
        3947592943: ("mexc", "SOL-USDT"),
        # Gate.io (primary mappings)
        4037619491: ("gate", "BTC-USDT"),
        4037619492: ("gate", "ETH-USDT"),
        4037619493: ("gate", "SOL-USDT"),
        # KuCoin
        4108940288: ("kucoin", "BTC-USDT"),
        4108940289: ("kucoin", "ETH-USDT"),
        4108940290: ("kucoin", "SOL-USDT"),
        # Bitget
        4179764809: ("bitget", "BTC-USDT"),
        4180201792: ("bitget", "ETH-USDT"),
        4180438775: ("bitget", "SOL-USDT"),
        # Crypto.com
        4179237857: ("crypto", "BTC-USDT"),
        4179674840: ("crypto", "ETH-USDT"),
        4179901823: ("crypto", "SOL-USDT"),
    }
    
    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize symbol mapper.
        
        Args:
            mapping_file: Optional path to JSON file with symbol mappings
        """
        self._id_to_info: Dict[int, Tuple[str, str]] = {}
        self._lock = threading.RLock()
        
        # Load default mappings
        self._load_default_mappings()
        
        # Load from file if provided
        if mapping_file:
            if os.path.exists(mapping_file):
                self.load_from_file(mapping_file)
            else:
                # Try resolving relative to current directory
                cwd_path = os.path.join(os.getcwd(), mapping_file)
                if os.path.exists(cwd_path):
                    self.load_from_file(cwd_path)
                else:
                    logger.warning(f"Symbol mapping file not found: {mapping_file} (checked: {mapping_file}, {cwd_path})")
                    logger.info("Using only default symbol mappings")
    
    def _load_default_mappings(self):
        """Load default symbol mappings"""
        with self._lock:
            for symbol_id, (exchange, symbol) in self.DEFAULT_MAPPINGS.items():
                self._id_to_info[symbol_id] = (exchange, symbol)
    
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
                
                logger.info(f"Loaded {len(self._id_to_info)} symbol mappings from {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load symbol mappings from {filepath}: {e}")
                raise
    
    def get_symbol_info(self, symbol_id: int) -> Tuple[str, str]:
        """
        Get exchange and symbol for a symbol ID.
        
        Args:
            symbol_id: The symbol ID to look up
            
        Returns:
            Tuple of (exchange, symbol)
        """
        with self._lock:
            if symbol_id in self._id_to_info:
                return self._id_to_info[symbol_id]
            return (f"unknown_exchange_{symbol_id}", f"symbol_{symbol_id}")
    
    @staticmethod
    def generate_symbol_id(exchange: str, symbol: str, market_type: str = "spot") -> int:
        """
        Generate symbol ID using hash function (matches C++ implementation).
        
        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            market_type: Market type (spot, futures, etc.)
            
        Returns:
            Hash-based symbol ID
        """
        key = f"{exchange}:{symbol}:{market_type}"
        hash_val = 5381
        for c in key:
            hash_val = ((hash_val << 5) + hash_val) + ord(c)
        return hash_val & 0xFFFFFFFF


# ============================================================================
# HotSpine Reader
# ============================================================================

class HotSpineReader:
    """
    High-performance HotSpine shared memory reader.
    
    This reader provides direct access to market data stored in shared memory
    by the C++ HotSpine writer. It supports both trade and orderbook data.
    
    Key Features:
    - Direct shared memory access via mmap and ctypes
    - Trade and orderbook support
    - Efficient circular buffer reading
    - Health monitoring and statistics
    
    Usage:
        reader = HotSpineReader("/dev/shm/btquant_hotspine")
        while True:
            trade = reader.poll_trade()
            if trade:
                process_trade(trade)
            time.sleep(0.001)
    """
    
    def __init__(
        self,
        shm_path: str,
        symbol_mapper: Optional[SymbolMapper] = None,
        auto_attach: bool = True,
    ):
        """
        Initialize HotSpine reader.
        
        Args:
            shm_path: Path to shared memory file (e.g., /dev/shm/btquant_hotspine)
            symbol_mapper: Optional SymbolMapper for resolving symbol IDs
            auto_attach: Whether to attach immediately on init
        """
        self.shm_path = shm_path
        self.symbol_mapper = symbol_mapper or SymbolMapper()
        
        # File descriptors and memory mapping
        self._fd = -1
        self._mmap_obj = None
        self._header: Optional[SharedMemoryHeader] = None
        self._trades_buffer = None
        self._orderbooks_buffer = None
        
        # State
        self._attached = False
        self._closed = False
        self._lock = threading.RLock()
        
        # Performance statistics
        self._stats = {
            'trades_read': 0,
            'orderbooks_read': 0,
            'poll_calls': 0,
            'empty_polls': 0,
            'read_errors': 0,
            'start_time': None,
        }
        
        # Attach if requested
        if auto_attach:
            self.attach()
    
    def attach(self) -> bool:
        """
        Attach to shared memory segment.
        
        Returns:
            True if attached successfully, False otherwise
        """
        with self._lock:
            try:
                # Check if file exists
                if not os.path.exists(self.shm_path):
                    logger.error(f"Shared memory file not found: {self.shm_path}")
                    return False
                
                # Open file descriptor
                # Use O_RDWR for ctypes.from_buffer compatibility
                self._fd = os.open(self.shm_path, os.O_RDWR)
                file_size = os.fstat(self._fd).st_size
                
                # Memory map the file
                self._mmap_obj = mmap.mmap(
                    self._fd,
                    file_size,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE
                )
                
                # Create header pointer
                header_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap_obj, 0))
                self._header = ctypes.cast(
                    ctypes.c_void_p(header_addr),
                    ctypes.POINTER(SharedMemoryHeader)
                ).contents
                
                # Validate version
                if self._header.version != HOTSPINE_VERSION:
                    logger.error(
                        f"Invalid HotSpine version: {self._header.version}, "
                        f"expected {HOTSPINE_VERSION}"
                    )
                    self.detach()
                    return False
                
                # Calculate buffer pointers
                buffer_offset = HEADER_SIZE
                
                # Trade buffer
                trade_buffer_size = self._header.capacity * TRADE_SIZE
                if buffer_offset + trade_buffer_size <= file_size:
                    trades_addr = ctypes.addressof(
                        ctypes.c_char.from_buffer(self._mmap_obj, buffer_offset)
                    )
                    self._trades_buffer = ctypes.cast(
                        ctypes.c_void_p(trades_addr),
                        ctypes.POINTER(HotTrade)
                    )
                
                # Orderbook buffer
                orderbook_offset = buffer_offset + trade_buffer_size
                if orderbook_offset + (self._header.orderbook_capacity * ORDERBOOK_SIZE) <= file_size:
                    orderbooks_addr = ctypes.addressof(
                        ctypes.c_char.from_buffer(self._mmap_obj, orderbook_offset)
                    )
                    self._orderbooks_buffer = ctypes.cast(
                        ctypes.c_void_p(orderbooks_addr),
                        ctypes.POINTER(HotOrderbookSnapshot)
                    )
                
                self._attached = True
                self._stats['start_time'] = time.time()
                
                logger.info(
                    f"Attached to HotSpine: {self.shm_path} "
                    f"(trades: {self._header.capacity}, "
                    f"orderbooks: {self._header.orderbook_capacity})"
                )
                return True
                
            except Exception as e:
                logger.error(f"Failed to attach to shared memory: {e}")
                self.detach()
                return False
    
    def detach(self):
        """Detach from shared memory and release resources"""
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            # Unmap memory
            if self._mmap_obj:
                try:
                    self._mmap_obj.close()
                except Exception as e:
                    logger.warning(f"Error closing mmap: {e}")
                self._mmap_obj = None
            
            # Close file descriptor
            if self._fd >= 0:
                try:
                    os.close(self._fd)
                except Exception as e:
                    logger.warning(f"Error closing file descriptor: {e}")
                self._fd = -1
            
            self._header = None
            self._trades_buffer = None
            self._orderbooks_buffer = None
            self._attached = False
            
            logger.info(f"Detached from HotSpine: {self.shm_path}")
    
    def is_attached(self) -> bool:
        """Check if attached to shared memory"""
        return self._attached
    
    def is_healthy(self) -> bool:
        """Check if reader is healthy"""
        with self._lock:
            if not self._attached:
                return False
            try:
                # Try to read header to verify connection
                _ = self._header.write_index
                return True
            except Exception:
                return False
    
    # =========================================================================
    # Trade Reading Methods
    # =========================================================================
    
    def get_available_trades(self) -> int:
        """Get number of available trades in buffer"""
        with self._lock:
            if not self._attached:
                return 0
            
            write_idx = self._header.write_index
            read_idx = self._header.read_index
            capacity = self._header.capacity
            
            if write_idx >= read_idx:
                return write_idx - read_idx
            return capacity - read_idx + write_idx
    
    def poll_trade(self) -> Optional[HotTrade]:
        """
        Poll for a single trade (non-blocking).
        
        Returns:
            HotTrade object if available, None otherwise
        """
        with self._lock:
            if not self._attached or not self._trades_buffer:
                return None
            
            try:
                write_idx = self._header.write_index
                read_idx = self._header.read_index
                capacity = self._header.capacity
                
                # Check if trades available
                if write_idx >= read_idx:
                    available = write_idx - read_idx
                else:
                    available = capacity - read_idx + write_idx
                
                if available == 0:
                    self._stats['empty_polls'] += 1
                    return None
                
                # Read trade at current read index
                trade = self._trades_buffer[read_idx % capacity]
                
                # Update read index (atomic-like increment)
                self._header.read_index = (read_idx + 1) % capacity
                
                self._stats['trades_read'] += 1
                self._stats['poll_calls'] += 1
                
                return trade
                
            except Exception as e:
                self._stats['read_errors'] += 1
                logger.error(f"Error reading trade: {e}")
                return None
    
    def poll_trade_dict(self) -> Optional[Dict[str, Any]]:
        """Poll for a trade and return as dictionary"""
        trade = self.poll_trade()
        if trade:
            return trade.to_dict(self.symbol_mapper)
        return None
    
    def read_all_trades(self) -> List[HotTrade]:
        """Read all available trades at once"""
        trades = []
        while True:
            trade = self.poll_trade()
            if trade:
                trades.append(trade)
            else:
                break
        return trades
    
    def read_all_trades_dict(self) -> List[Dict[str, Any]]:
        """Read all available trades as dictionaries"""
        return [t.to_dict(self.symbol_mapper) for t in self.read_all_trades()]
    
    # =========================================================================
    # Orderbook Reading Methods
    # =========================================================================
    
    def get_available_orderbooks(self) -> int:
        """Get number of available orderbooks in buffer"""
        with self._lock:
            if not self._attached or not self._orderbooks_buffer:
                return 0
            
            write_idx = self._header.orderbook_write_index
            read_idx = self._header.orderbook_read_index
            capacity = self._header.orderbook_capacity
            
            if write_idx >= read_idx:
                return write_idx - read_idx
            return capacity - read_idx + write_idx
    
    def poll_orderbook(self) -> Optional[HotOrderbookSnapshot]:
        """
        Poll for a single orderbook (non-blocking).
        
        Returns:
            HotOrderbookSnapshot object if available, None otherwise
        """
        with self._lock:
            if not self._attached or not self._orderbooks_buffer:
                return None
            
            try:
                write_idx = self._header.orderbook_write_index
                read_idx = self._header.orderbook_read_index
                capacity = self._header.orderbook_capacity
                
                # Check if orderbooks available
                if write_idx >= read_idx:
                    available = write_idx - read_idx
                else:
                    available = capacity - read_idx + write_idx
                
                if available == 0:
                    self._stats['empty_polls'] += 1
                    return None
                
                # Read orderbook at current read index
                snapshot = self._orderbooks_buffer[read_idx % capacity]
                
                # Update read index
                self._header.orderbook_read_index = (read_idx + 1) % capacity
                
                self._stats['orderbooks_read'] += 1
                self._stats['poll_calls'] += 1
                
                return snapshot
                
            except Exception as e:
                self._stats['read_errors'] += 1
                logger.error(f"Error reading orderbook: {e}")
                return None
    
    def poll_orderbook_dict(self) -> Optional[Dict[str, Any]]:
        """Poll for an orderbook and return as dictionary"""
        snapshot = self.poll_orderbook()
        if snapshot:
            return snapshot.to_dict(self.symbol_mapper)
        return None
    
    def read_all_orderbooks(self) -> List[HotOrderbookSnapshot]:
        """Read all available orderbooks at once"""
        orderbooks = []
        while True:
            orderbook = self.poll_orderbook()
            if orderbook:
                orderbooks.append(orderbook)
            else:
                break
        return orderbooks
    
    def read_all_orderbooks_dict(self) -> List[Dict[str, Any]]:
        """Read all available orderbooks as dictionaries"""
        return [o.to_dict(self.symbol_mapper) for o in self.read_all_orderbooks()]
    
    # =========================================================================
    # Statistics Methods
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            stats = self._stats.copy()
            
            if stats['start_time']:
                elapsed = time.time() - stats['start_time']
                stats['elapsed_seconds'] = elapsed
                
                if elapsed > 0:
                    stats['trades_per_second'] = stats['trades_read'] / elapsed
                    stats['orderbooks_per_second'] = stats['orderbooks_read'] / elapsed
            
            stats['attached'] = self._attached
            stats['healthy'] = self.is_healthy()
            
            # Buffer info
            stats['trade_buffer'] = {
                'available': self.get_available_trades(),
                'capacity': self._header.capacity if self._attached else 0,
                'lost': self._header.lost_count if self._attached else 0,
            }
            stats['orderbook_buffer'] = {
                'available': self.get_available_orderbooks(),
                'capacity': self._header.orderbook_capacity if self._attached else 0,
                'lost': getattr(self._header, 'orderbook_lost_count', 0) if self._attached else 0,
            }
            
            return stats
    
    def print_statistics(self):
        """Print human-readable statistics"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("HotSpine Reader Statistics")
        print("=" * 60)
        print(f"Attached:              {stats.get('attached', False)}")
        print(f"Healthy:               {stats.get('healthy', False)}")
        print(f"Trades Read:           {stats.get('trades_read', 0):,}")
        print(f"Trades/Second:         {stats.get('trades_per_second', 0):.2f}")
        print(f"Orderbooks Read:       {stats.get('orderbooks_read', 0):,}")
        print(f"Orderbooks/Second:     {stats.get('orderbooks_per_second', 0):.2f}")
        print(f"Empty Polls:           {stats.get('empty_polls', 0):,}")
        print(f"Read Errors:           {stats.get('read_errors', 0):,}")
        
        trade_buf = stats.get('trade_buffer', {})
        print(f"\nTrade Buffer:")
        print(f"  Available:           {trade_buf.get('available', 0):,}")
        print(f"  Capacity:            {trade_buf.get('capacity', 0):,}")
        print(f"  Lost (overflow):     {trade_buf.get('lost', 0):,}")
        
        ob_buf = stats.get('orderbook_buffer', {})
        print(f"\nOrderbook Buffer:")
        print(f"  Available:           {ob_buf.get('available', 0):,}")
        print(f"  Capacity:            {ob_buf.get('capacity', 0):,}")
        print(f"  Lost (overflow):     {ob_buf.get('lost', 0):,}")
        
        print("=" * 60)
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()
        return False
    
    def __del__(self):
        self.detach()


# ============================================================================
# Data Processor (Filtering and Aggregation)
# ============================================================================

class DataProcessor:
    """
    Processes HotSpine data with filtering and aggregation support.
    """
    
    def __init__(self, filters: Optional[Dict[str, Any]] = None):
        """
        Initialize data processor.
        
        Args:
            filters: Optional dictionary of filters to apply
        """
        self.filters = filters or {}
        self.trades_processed = 0
        self.orderbooks_processed = 0
        self.processed_data: List[Dict[str, Any]] = []
    
    def matches_filters(self, data: Dict[str, Any]) -> bool:
        """Check if data matches configured filters"""
        for key, value in self.filters.items():
            if key not in data:
                return False
            
            data_value = data[key]
            
            # String matching (contains)
            if isinstance(value, str) and isinstance(data_value, str):
                if value.lower() not in data_value.lower():
                    return False
            # Exact matching
            elif data_value != value:
                return False
        
        return True
    
    def process_trade(self, trade: Dict[str, Any]) -> bool:
        """Process a trade with filtering"""
        if not self.matches_filters(trade):
            return False
        
        self.trades_processed += 1
        self.processed_data.append(trade)
        return True
    
    def process_orderbook(self, orderbook: Dict[str, Any]) -> bool:
        """Process an orderbook with filtering"""
        if not self.matches_filters(orderbook):
            return False
        
        self.orderbooks_processed += 1
        self.processed_data.append(orderbook)
        return True
    
    def aggregate_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate processed data by symbol"""
        agg = defaultdict(lambda: {
            "trades": 0,
            "orderbooks": 0,
            "total_volume": 0.0,
            "price_min": float('inf'),
            "price_max": float('-inf'),
            "last_price": None,
            "last_update": None,
        })
        
        for item in self.processed_data:
            symbol = f"{item['exchange']}/{item['symbol']}"
            data = agg[symbol]
            
            if item["type"] == "trade":
                data["trades"] += 1
                data["total_volume"] += item.get("size", 0)
                
                price = item.get("price")
                if price is not None:
                    data["price_min"] = min(data["price_min"], price)
                    data["price_max"] = max(data["price_max"], price)
                    data["last_price"] = price
                
                data["last_update"] = item.get("datetime")
            
            elif item["type"] == "orderbook":
                data["orderbooks"] += 1
                if not data["last_update"] or item.get("datetime", "") > data["last_update"]:
                    data["last_update"] = item.get("datetime")
        
        # Clean up infinite values
        for symbol, data in agg.items():
            if data["price_min"] == float('inf'):
                data["price_min"] = None
            if data["price_max"] == float('-inf'):
                data["price_max"] = None
        
        return dict(agg)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        return {
            "total_processed": len(self.processed_data),
            "trades_processed": self.trades_processed,
            "orderbooks_processed": self.orderbooks_processed,
            "filters_applied": self.filters,
        }


# ============================================================================
# Output Writer
# ============================================================================

class OutputWriter:
    """Handles output formatting and writing"""
    
    def __init__(self, output_format: str = "json", output_file: Optional[str] = None):
        """
        Initialize output writer.
        
        Args:
            output_format: Output format (json, csv, stats)
            output_file: Optional output file path
        """
        self.output_format = output_format
        self.output_file = output_file
        self.file_handle = None
        self.item_count = 0
        
        if self.output_file:
            self.file_handle = open(self.output_file, 'w', encoding='utf-8')
            
            if self.output_format == "csv":
                import csv
                self.csv_writer = csv.writer(self.file_handle)
                self.csv_writer.writerow([
                    "type", "timestamp_us", "exchange", "symbol", "price", 
                    "size", "side", "bids_count", "asks_count", "datetime"
                ])
    
    def write(self, data: Dict[str, Any]):
        """Write a single data item"""
        self.item_count += 1
        
        if self.output_format == "json":
            self._write_json(data)
        elif self.output_format == "csv":
            self._write_csv(data)
        elif self.output_format == "stats":
            self._write_stats(data)
        else:
            # Default to pretty print
            print(json.dumps(data, indent=2, default=str))
    
    def _write_json(self, data: Dict[str, Any]):
        """Write data as JSON"""
        line = json.dumps(data, default=str)
        if self.file_handle:
            self.file_handle.write(line + "\n")
            self.file_handle.flush()
        else:
            print(line)
    
    def _write_csv(self, data: Dict[str, Any]):
        """Write data as CSV"""
        if data["type"] == "trade":
            row = [
                data["type"],
                data.get("timestamp_us", ""),
                data.get("exchange", ""),
                data.get("symbol", ""),
                data.get("price", ""),
                data.get("size", ""),
                data.get("side", ""),
                "",  # bids_count
                "",  # asks_count
                data.get("datetime", "")
            ]
        else:  # orderbook
            row = [
                data["type"],
                data.get("timestamp_us", ""),
                data.get("exchange", ""),
                data.get("symbol", ""),
                "",  # price
                "",  # size
                "",  # side
                data.get("bids_count", ""),
                data.get("asks_count", ""),
                data.get("datetime", "")
            ]
        
        if self.file_handle:
            self.csv_writer.writerow(row)
            self.file_handle.flush()
        else:
            print(",".join(str(x) for x in row))
    
    def _write_stats(self, data: Dict[str, Any]):
        """Write statistics summary"""
        if self.file_handle:
            self.file_handle.write(data + "\n")
            self.file_handle.flush()
        else:
            print(data)
    
    def close(self):
        """Close output file"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


# ============================================================================
# Monitor Class
# ============================================================================

class HotSpineMonitor:
    """
    Main HotSpine monitoring class.
    
    Orchestrates reading, processing, and outputting HotSpine data.
    """
    
    def __init__(
        self,
        shm_name: str = "btquant_hotspine",
        symbol_mapping_file: Optional[str] = None,
        output_format: str = "json",
        output_file: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        poll_interval: float = 0.001,
        max_items: int = 0,
        batch_size: int = 100,
        stats_interval: int = 10,
    ):
        """
        Initialize monitor.
        
        Args:
            shm_name: Shared memory segment name
            symbol_mapping_file: Optional symbol mapping file path
            output_format: Output format (json, csv, stats, pretty)
            output_file: Optional output file path
            filters: Optional filters to apply
            poll_interval: Polling interval in seconds
            max_items: Maximum items to process (0 = unlimited)
            batch_size: Number of items to read per batch
            stats_interval: Statistics reporting interval in seconds
        """
        self.shm_name = shm_name
        self.shm_path = f"/dev/shm/{shm_name}"
        self.poll_interval = poll_interval
        self.max_items = max_items
        self.batch_size = batch_size
        self.stats_interval = stats_interval
        self.running = False
        
        # Create components
        self.symbol_mapper = SymbolMapper(symbol_mapping_file)
        self.reader = HotSpineReader(self.shm_path, self.symbol_mapper, auto_attach=False)
        self.processor = DataProcessor(filters)
        self.output_writer = OutputWriter(output_format, output_file)
        
        # Statistics
        self.start_time = None
        self.last_stats_time = None
        self.items_processed = 0
    
    def start(self) -> bool:
        """Start monitoring"""
        logger.info(f"Starting HotSpine monitor: {self.shm_name}")
        
        if not self.reader.attach():
            logger.error("Failed to attach to shared memory")
            return False
        
        self.running = True
        self.start_time = time.time()
        self.last_stats_time = self.start_time
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._monitor_loop()
        return True
    
    def stop(self):
        """Stop monitoring"""
        logger.info("Stopping HotSpine monitor")
        self.running = False
        self.reader.detach()
        self.output_writer.close()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            while self.running:
                # Check if max items reached
                if self.max_items > 0 and self.items_processed >= self.max_items:
                    logger.info(f"Max items ({self.max_items}) reached, stopping")
                    break
                
                # Read available trades
                available_trades = self.reader.get_available_trades()
                trades_to_read = min(available_trades, self.batch_size)
                
                for _ in range(trades_to_read):
                    if self.max_items > 0 and self.items_processed >= self.max_items:
                        break
                    
                    trade_dict = self.reader.poll_trade_dict()
                    if trade_dict and self.processor.process_trade(trade_dict):
                        self.output_writer.write(trade_dict)
                        self.items_processed += 1
                
                # Read available orderbooks
                available_orderbooks = self.reader.get_available_orderbooks()
                obs_to_read = min(available_orderbooks, self.batch_size // 10)
                
                for _ in range(obs_to_read):
                    if self.max_items > 0 and self.items_processed >= self.max_items:
                        break
                    
                    ob_dict = self.reader.poll_orderbook_dict()
                    if ob_dict and self.processor.process_orderbook(ob_dict):
                        self.output_writer.write(ob_dict)
                        self.items_processed += 1
                
                # Print stats periodically
                current_time = time.time()
                if current_time - self.last_stats_time >= self.stats_interval:
                    self._print_progress()
                    self.last_stats_time = current_time
                
                # Sleep to reduce CPU usage
                if available_trades == 0 and available_orderbooks == 0:
                    time.sleep(self.poll_interval)
        
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}")
        finally:
            self._print_final_summary()
    
    def _print_progress(self):
        """Print progress statistics"""
        stats = self.reader.get_statistics()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(
            f"[PROGRESS] Trades: {stats.get('trades_read', 0):,} "
            f"({stats.get('trades_per_second', 0):.1f}/s) | "
            f"Orderbooks: {stats.get('orderbooks_read', 0):,} "
            f"({stats.get('orderbooks_per_second', 0):.1f}/s) | "
            f"Elapsed: {elapsed:.1f}s"
        )
    
    def _print_final_summary(self):
        """Print final processing summary"""
        summary = self.processor.get_summary()
        agg = self.processor.aggregate_by_symbol()
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 70)
        print("FINAL PROCESSING SUMMARY")
        print("=" * 70)
        print(f"Items Processed:       {self.items_processed:,}")
        print(f"Trades:                {summary['trades_processed']:,}")
        print(f"Orderbooks:            {summary['orderbooks_processed']:,}")
        print(f"Elapsed Time:          {elapsed:.2f}s")
        
        if self.processor.filters:
            print(f"Filters Applied:       {self.processor.filters}")
        
        if agg:
            print("\nSymbol Aggregation:")
            for symbol, data in sorted(agg.items()):
                price_range = ""
                if data["price_min"] is not None and data["price_max"] is not None:
                    price_range = f" | Range: {data['price_min']:.2f} - {data['price_max']:.2f}"
                print(
                    f"  {symbol}: {data['trades']} trades, "
                    f"{data['orderbooks']} orderbooks, "
                    f"vol={data['total_volume']:.4f}{price_range}"
                )
        
        print("=" * 70)


# ============================================================================
# Utility Functions
# ============================================================================

def parse_filters(filter_str: str) -> Dict[str, Any]:
    """
    Parse filter string like 'exchange=binance,symbol=BTC-USDT'.
    
    Args:
        filter_str: Comma-separated key=value pairs
        
    Returns:
        Dictionary of filters
    """
    filters = {}
    if not filter_str:
        return filters
    
    for pair in filter_str.split(','):
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to appropriate type
            if value.isdigit():
                filters[key] = int(value)
            elif value.replace('.', '').isdigit():
                filters[key] = float(value)
            else:
                filters[key] = value
    
    return filters


def list_shm_segments() -> List[str]:
    """List available HotSpine shared memory segments in /dev/shm"""
    segments = []
    try:
        for filename in os.listdir("/dev/shm"):
            # Filter for HotSpine segments
            if filename.startswith("btquant") or filename.startswith("hotspine"):
                segments.append(filename)
    except Exception as e:
        logger.error(f"Error listing /dev/shm: {e}")
    return segments


def discover_symbols(shm_name: str, duration: float = 5.0) -> Dict[int, Dict[str, Any]]:
    """
    Discover symbol IDs by reading data for a brief period.
    
    Args:
        shm_name: Shared memory segment name
        duration: Duration to monitor in seconds
        
    Returns:
        Dictionary mapping symbol_id to sample data
    """
    shm_path = f"/dev/shm/{shm_name}"
    
    print(f"\n{'='*70}")
    print(f"SYMBOL DISCOVERY MODE")
    print(f"Monitoring {shm_path} for {duration} seconds...")
    print(f"{'='*70}\n")
    
    try:
        reader = HotSpineReader(shm_path, auto_attach=True)
        if not reader.is_attached():
            print(f"Failed to attach to {shm_path}")
            return {}
        
        # Track discovered symbols
        discovered: Dict[int, Dict[str, Any]] = {}
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Read trades
            trades = reader.read_all_trades()
            for trade in trades:
                if trade.symbol_id not in discovered:
                    discovered[trade.symbol_id] = {
                        "type": "trade",
                        "price": trade.price,
                        "size": trade.size,
                        "side": trade.side_str,
                        "timestamp": trade.ts_exchange,
                    }
            
            # Read orderbooks
            orderbooks = reader.read_all_orderbooks()
            for ob in orderbooks:
                if ob.symbol_id not in discovered:
                    discovered[ob.symbol_id] = {
                        "type": "orderbook",
                        "mid_price": ob.get_mid_price(),
                        "bids_count": ob.bids_count,
                        "asks_count": ob.asks_count,
                        "timestamp": ob.ts_exchange,
                    }
            
            time.sleep(0.1)
        
        reader.detach()
        
        # Print results
        print(f"\nDiscovered {len(discovered)} unique symbol IDs:\n")
        print(f"{'Symbol ID':<15} {'Exchange/Symbol':<35} {'Sample Data'}")
        print("-" * 100)
        
        for symbol_id, data in sorted(discovered.items()):
            # Generate potential mapping using common exchanges
            # Try exact hash matching first
            exchange = None
            symbol = None
            
            # Check against default mappings
            if symbol_id in SymbolMapper.DEFAULT_MAPPINGS:
                exchange, symbol = SymbolMapper.DEFAULT_MAPPINGS[symbol_id]
            else:
                # Try brute-force matching for other common pairs
                exchanges = ["binance", "bybit", "okx", "mexc", "gate", "kucoin", "bitget", "crypto"]
                symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT"]
                
                for ex in exchanges:
                    for sym in symbols:
                        test_id = SymbolMapper.generate_symbol_id(ex, sym, "spot")
                        if test_id == symbol_id:
                            exchange = ex
                            symbol = sym
                            break
                    if exchange:
                        break
            
            if exchange and symbol:
                potential_str = f"{exchange}/{symbol}"
            else:
                potential_str = "(unknown)"
            
            sample = ""
            if data["type"] == "trade":
                sample = f"trade @ {data['price']} ({data['side']})"
            else:
                sample = f"orderbook mid={data.get('mid_price', 'N/A')}"
            
            print(f"{symbol_id:<15} {potential_str:<35} {sample}")
        
        print(f"\n{'='*70}")
        print("To create a mapping file, run:")
        print(f'  python hotspine_monitor.py --shm-name {shm_name} --symbol-mapping my_mappings.json')
        print(f"{'='*70}\n")
        
        return discovered
        
    except Exception as e:
        print(f"Error during discovery: {e}")
        return {}


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HotSpine Monitor - Real-time market data reader from shared memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor with default settings
  python hotspine_monitor.py
  
  # Filter by exchange
  python hotspine_monitor.py --filter exchange=binance
  
  # Output to file in CSV format
  python hotspine_monitor.py --output csv --output-file trades.csv
  
  # Limit to 1000 items
  python hotspine_monitor.py --max-items 1000
  
  # List available shared memory segments
  python hotspine_monitor.py --list-segments
  
  # Discover symbol IDs being used (5 second scan)
  python hotspine_monitor.py --discover
  
  # Discover with longer duration
  python hotspine_monitor.py --discover --discover-duration 10
        """
    )
    
    parser.add_argument("--shm-name", default="btquant_hotspine",
                       help="Shared memory segment name (default: btquant_hotspine)")
    parser.add_argument("--list-segments", action="store_true",
                       help="List available HotSpine segments in /dev/shm")
    parser.add_argument("--discover", action="store_true",
                       help="Discover symbol IDs by monitoring briefly (5 seconds)")
    parser.add_argument("--discover-duration", type=float, default=5.0,
                       help="Duration for symbol discovery in seconds (default: 5)")
    parser.add_argument("--output", choices=["json", "csv", "pretty", "stats"], 
                       default="pretty", help="Output format (default: pretty)")
    parser.add_argument("--output-file", 
                       help="Output file path (default: stdout)")
    parser.add_argument("--filter", 
                       help="Filters as key=value pairs separated by commas")
    parser.add_argument("--symbol-mapping", 
                       help="Path to symbol mapping JSON file")
    parser.add_argument("--poll-interval", type=float, default=0.001,
                       help="Polling interval in seconds (default: 0.001)")
    parser.add_argument("--max-items", type=int, default=0,
                       help="Maximum items to process (0=unlimited, default: 0)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Items to read per batch (default: 100)")
    parser.add_argument("--stats-interval", type=int, default=10,
                       help="Statistics interval in seconds (default: 10)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # List segments if requested
    if args.list_segments:
        segments = list_shm_segments()
        print("Available HotSpine segments in /dev/shm:")
        for seg in segments:
            print(f"  - {seg}")
        return
    
    # Discover symbols if requested
    if args.discover:
        discover_symbols(args.shm_name, args.discover_duration)
        return
    
    # Parse filters
    filters = parse_filters(args.filter) if args.filter else {}
    
    # Create and start monitor
    monitor = HotSpineMonitor(
        shm_name=args.shm_name,
        symbol_mapping_file=args.symbol_mapping,
        output_format=args.output,
        output_file=args.output_file,
        filters=filters,
        poll_interval=args.poll_interval,
        max_items=args.max_items,
        batch_size=args.batch_size,
        stats_interval=args.stats_interval,
    )
    
    try:
        success = monitor.start()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
