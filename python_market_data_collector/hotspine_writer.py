"""
HotSpine Writer - Python equivalent of HotSpine::HotSpineWriter

This module provides a Python implementation of the HotSpine writer
that can write trade data to shared memory for consumption by readers.
"""

import mmap
import os
import struct
import threading
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .market_data_types import Trade

logger = logging.getLogger(__name__)


@dataclass
class HotSpineTrade:
    """Binary representation of a trade for shared memory"""
    ts_exchange: int  # uint64_t - Exchange timestamp in microseconds
    ts_local: int    # uint64_t - Local receive timestamp in microseconds
    price: float     # double
    size: float      # double
    symbol_id: int   # uint32_t - Symbol ID (hash or mapping)
    side: int        # uint8_t - 0=buy, 1=sell
    market_type: int # uint8_t - 0=spot, 1=futures, 2=other

    @classmethod
    def from_trade(cls, trade: Trade, symbol_id: int = 0) -> 'HotSpineTrade':
        """Convert a Trade object to HotSpineTrade"""
        # Map market type string to int
        market_type_map = {
            'spot': 0,
            'futures': 1,
            'perpetual': 1,
            'other': 2
        }
        market_type_int = market_type_map.get(trade.market_type.lower(), 0)

        # Map side string to int
        side_int = 0 if trade.side.lower() == 'buy' else 1

        return cls(
            ts_exchange=trade.timestamp_us,
            ts_local=int(time.time() * 1_000_000),  # Current time in microseconds
            price=trade.price,
            size=trade.quantity,
            symbol_id=symbol_id,
            side=side_int,
            market_type=market_type_int
        )

    def to_bytes(self) -> bytes:
        """Convert to bytes for shared memory"""
        return struct.pack(
            '<QQddIIBB',  # Little-endian format
            self.ts_exchange,
            self.ts_local,
            self.price,
            self.size,
            self.symbol_id,
            self.side,
            self.market_type,
            0  # Padding byte
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'HotSpineTrade':
        """Create from bytes"""
        unpacked = struct.unpack('<QQddIIBB', data)
        return cls(
            ts_exchange=unpacked[0],
            ts_local=unpacked[1],
            price=unpacked[2],
            size=unpacked[3],
            symbol_id=unpacked[4],
            side=unpacked[5],
            market_type=unpacked[6]
        )


class HotSpineWriter:
    """
    High-performance HotSpine writer for shared memory trade data.

    This writer maintains a circular buffer in shared memory where trades
    can be written for consumption by HotSpine readers.
    """

    # Constants matching C++ implementation
    TRADE_SIZE = 48  # Size of HotSpineTrade in bytes
    HEADER_SIZE = 16  # Header with read/write positions
    DEFAULT_BUFFER_SIZE = 100000  # Default number of trades in buffer

    def __init__(self, shm_name: str = "/btquant_hotspine", buffer_size: int = DEFAULT_BUFFER_SIZE):
        """
        Initialize HotSpine writer.

        Args:
            shm_name: Shared memory segment name
            buffer_size: Number of trades to store in circular buffer
        """
        self.shm_name = shm_name
        self.buffer_size = buffer_size
        self.total_size = self.HEADER_SIZE + (self.buffer_size * self.TRADE_SIZE)

        self.shm_fd = None
        self.shm_mmap = None
        self.is_initialized = False

        # Batching
        self.batching_enabled = False
        self.batch_size = 1
        self.current_batch = []

        # Statistics
        self.trades_written = 0
        self.batches_flushed = 0
        self.write_errors = 0

        # Thread safety
        self.lock = threading.Lock()

        # Symbol mapping for consistent IDs
        self.symbol_to_id = {}
        self.next_symbol_id = 1

        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Initialize shared memory segment"""
        try:
            # Create shared memory file
            shm_path = f"/dev/shm{self.shm_name}"

            # Remove existing file if it exists
            if os.path.exists(shm_path):
                os.unlink(shm_path)

            # Create new shared memory file
            self.shm_fd = os.open(shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
            os.ftruncate(self.shm_fd, self.total_size)

            # Memory map the file
            self.shm_mmap = mmap.mmap(self.shm_fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

            # Initialize header (read_pos = 0, write_pos = 0)
            self.shm_mmap.seek(0)
            self.shm_mmap.write(struct.pack('<QQ', 0, 0))

            self.is_initialized = True
            logger.info(f"HotSpine writer initialized: {self.shm_name}, buffer size: {self.buffer_size}")

        except Exception as e:
            logger.error(f"Failed to initialize shared memory: {e}")
            self._cleanup()
            raise

    def _cleanup(self):
        """Clean up shared memory resources"""
        if self.shm_mmap:
            try:
                self.shm_mmap.close()
            except:
                pass
            self.shm_mmap = None

        if self.shm_fd:
            try:
                os.close(self.shm_fd)
            except:
                pass
            self.shm_fd = None

        # Clean up shared memory file
        try:
            shm_path = f"/dev/shm{self.shm_name}"
            if os.path.exists(shm_path):
                os.unlink(shm_path)
        except:
            pass

        self.is_initialized = False

    def __del__(self):
        """Destructor - clean up resources"""
        self._cleanup()

    def _get_symbol_id(self, symbol: str) -> int:
        """Get or create symbol ID for consistent mapping"""
        if symbol not in self.symbol_to_id:
            self.symbol_to_id[symbol] = self.next_symbol_id
            self.next_symbol_id += 1
        return self.symbol_to_id[symbol]

    def write_trade(self, trade: Trade) -> bool:
        """
        Write a trade to shared memory.

        Args:
            trade: Trade object to write

        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("HotSpine writer not initialized")
            return False

        try:
            # Convert to HotSpine format
            symbol_id = self._get_symbol_id(f"{trade.exchange}:{trade.symbol}")
            hotspine_trade = HotSpineTrade.from_trade(trade, symbol_id)

            # Add to batch if batching enabled
            if self.batching_enabled:
                self.current_batch.append(hotspine_trade)
                if len(self.current_batch) >= self.batch_size:
                    return self.flush_batch()
                return True

            # Write immediately
            return self._write_trade_to_shm(hotspine_trade)

        except Exception as e:
            logger.error(f"Error writing trade: {e}")
            self.write_errors += 1
            return False

    def _write_trade_to_shm(self, hotspine_trade: HotSpineTrade) -> bool:
        """Write a single trade to shared memory"""
        with self.lock:
            try:
                # Read current positions
                self.shm_mmap.seek(0)
                read_pos, write_pos = struct.unpack('<QQ', self.shm_mmap.read(16))

                # Calculate next write position (circular buffer)
                next_write_pos = (write_pos + 1) % self.buffer_size

                # Check if buffer is full (would overwrite unread data)
                if next_write_pos == read_pos:
                    logger.warning("HotSpine buffer full, dropping trade")
                    return False

                # Write trade data
                trade_offset = self.HEADER_SIZE + (write_pos * self.TRADE_SIZE)
                self.shm_mmap.seek(trade_offset)
                self.shm_mmap.write(hotspine_trade.to_bytes())

                # Update write position
                self.shm_mmap.seek(8)  # Position of write_pos
                self.shm_mmap.write(struct.pack('<Q', next_write_pos))

                self.trades_written += 1
                return True

            except Exception as e:
                logger.error(f"Error writing to shared memory: {e}")
                return False

    def flush_batch(self) -> bool:
        """
        Flush current batch to shared memory.

        Returns:
            True if successful, False otherwise
        """
        if not self.batching_enabled or not self.current_batch:
            return True

        success = True
        for trade in self.current_batch:
            if not self._write_trade_to_shm(trade):
                success = False

        self.current_batch.clear()
        self.batches_flushed += 1

        return success

    def set_batching_enabled(self, enabled: bool):
        """Enable or disable batching"""
        if self.batching_enabled and not enabled:
            # Flush any pending batch before disabling
            self.flush_batch()

        self.batching_enabled = enabled

    def set_batch_size(self, size: int):
        """Set batch size for batching"""
        if size < 1:
            size = 1
        self.batch_size = size

    def get_detailed_stats(self) -> str:
        """Get detailed statistics as formatted string"""
        stats = []
        stats.append(f"HotSpine Writer Statistics:")
        stats.append(f"  Shared Memory: {self.shm_name}")
        stats.append(f"  Buffer Size: {self.buffer_size}")
        stats.append(f"  Trades Written: {self.trades_written}")
        stats.append(f"  Batches Flushed: {self.batches_flushed}")
        stats.append(f"  Write Errors: {self.write_errors}")
        stats.append(f"  Batching: {'Enabled' if self.batching_enabled else 'Disabled'}")
        stats.append(f"  Batch Size: {self.batch_size}")
        stats.append(f"  Current Batch Size: {len(self.current_batch)}")
        stats.append(f"  Symbols Mapped: {len(self.symbol_to_id)}")

        return "\n".join(stats)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics as dictionary"""
        return {
            'shm_name': self.shm_name,
            'buffer_size': self.buffer_size,
            'trades_written': self.trades_written,
            'batches_flushed': self.batches_flushed,
            'write_errors': self.write_errors,
            'batching_enabled': self.batching_enabled,
            'batch_size': self.batch_size,
            'current_batch_size': len(self.current_batch),
            'symbols_mapped': len(self.symbol_to_id)
        }