#!/usr/bin/env python3
"""
HotSpine Monitor Prototype

A small, efficient Python prototype script that monitors /dev/shm for HotSpine
shared memory segments, reads and processes market data in real-time, and
provides basic querying/filtering/aggregation functionality.

Features:
- Monitors /dev/shm for new HotSpine shared memory segments
- Reads and parses binary trade and orderbook data from shared memory
- Provides basic querying, filtering, and aggregation by timestamps/keys
- Includes error handling for file access and parsing issues
- Minimal resource usage with efficient polling
- Outputs processed results to console or log file
- Command-line arguments for customization

Usage:
    python hotspine_monitor_prototype.py --shm-name btquant_hotspine --output json --filter exchange=binance

Author: PubBTQuant Prototype
"""

import os
import sys
import time
import json
import csv
import mmap
import ctypes
import argparse
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# HotSpine Data Structures (matching C++ layout)
# ============================================================================

class HotTrade(ctypes.Structure):
    """Python representation of HotSpine trade structure"""
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),
        ("ts_local", ctypes.c_uint64),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
        ("symbol_id", ctypes.c_uint32),
        ("side", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 3),
    ]

class HotOrderbookLevel(ctypes.Structure):
    """Single orderbook level"""
    _fields_ = [
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
    ]

class HotOrderbookSnapshot(ctypes.Structure):
    """Python representation of HotSpine orderbook snapshot"""
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),
        ("ts_local", ctypes.c_uint64),
        ("symbol_id", ctypes.c_uint32),
        ("bids_count", ctypes.c_uint8),
        ("asks_count", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 2),
        ("bids", HotOrderbookLevel * 20),
        ("asks", HotOrderbookLevel * 20),
    ]

class SharedMemoryHeader(ctypes.Structure):
    """Shared memory header structure"""
    _fields_ = [
        ("version", ctypes.c_uint64),
        ("capacity", ctypes.c_uint64),
        ("write_index", ctypes.c_uint64),
        ("read_index", ctypes.c_uint64),
        ("lost_count", ctypes.c_uint64),
        ("orderbook_write_index", ctypes.c_uint64),
        ("orderbook_read_index", ctypes.c_uint64),
        ("orderbook_lost_count", ctypes.c_uint64),
        ("orderbook_capacity", ctypes.c_uint64),
        ("padding", ctypes.c_uint8 * 8),
    ]

# ============================================================================
# Constants
# ============================================================================

HEADER_SIZE = 4096
TRADE_SIZE = ctypes.sizeof(HotTrade)
ORDERBOOK_SIZE = ctypes.sizeof(HotOrderbookSnapshot)
HOTSPINE_VERSION = 2

# ============================================================================
# Symbol Mapper (simplified)
# ============================================================================

class SimpleSymbolMapper:
    """Simple symbol ID to exchange/symbol mapping"""

    DEFAULT_MAPPINGS = {
        1: ("binance", "BTC-USDT"),
        2: ("binance", "ETH-USDT"),
        3: ("binance", "SOL-USDT"),
        101: ("bybit", "BTC-USDT"),
        102: ("bybit", "ETH-USDT"),
    }

    def __init__(self):
        self.mappings = self.DEFAULT_MAPPINGS.copy()

    def get_symbol_info(self, symbol_id: int) -> Tuple[str, str]:
        """Get exchange and symbol for symbol ID"""
        return self.mappings.get(symbol_id, (f"unknown_exchange_{symbol_id}", f"unknown_symbol_{symbol_id}"))

# ============================================================================
# HotSpine Reader
# ============================================================================

class HotSpineReader:
    """Simple HotSpine shared memory reader"""

    def __init__(self, shm_path: str):
        self.shm_path = shm_path
        self.fd = None
        self.mmap_obj = None
        self.header = None
        self.trades_buffer = None
        self.orderbooks_buffer = None
        self.symbol_mapper = SimpleSymbolMapper()
        self._attached = False

    def attach(self) -> bool:
        """Attach to shared memory segment"""
        try:
            if not os.path.exists(self.shm_path):
                logger.error(f"Shared memory file does not exist: {self.shm_path}")
                return False

            # Open file descriptor - use O_RDWR for ctypes.from_buffer to work
            # Even though we only read, ctypes requires writable buffer
            self.fd = os.open(self.shm_path, os.O_RDWR)
            file_size = os.fstat(self.fd).st_size

            # Memory map the file - use PROT_READ | PROT_WRITE for ctypes compatibility
            self.mmap_obj = mmap.mmap(self.fd, file_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

            # Create header pointer
            header_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mmap_obj, 0))
            self.header = ctypes.cast(ctypes.c_void_p(header_addr), ctypes.POINTER(SharedMemoryHeader)).contents

            # Validate version
            if self.header.version != HOTSPINE_VERSION:
                logger.error(f"Invalid HotSpine version: {self.header.version}, expected {HOTSPINE_VERSION}")
                return False

            # Calculate buffer pointers
            buffer_offset = HEADER_SIZE

            # Trade buffer
            if buffer_offset + (self.header.capacity * TRADE_SIZE) <= file_size:
                trades_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mmap_obj, buffer_offset))
                self.trades_buffer = ctypes.cast(ctypes.c_void_p(trades_addr), ctypes.POINTER(HotTrade))

            # Orderbook buffer
            trade_buffer_size = self.header.capacity * TRADE_SIZE
            orderbook_offset = buffer_offset + trade_buffer_size

            if orderbook_offset + (self.header.orderbook_capacity * ORDERBOOK_SIZE) <= file_size:
                orderbooks_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mmap_obj, orderbook_offset))
                self.orderbooks_buffer = ctypes.cast(ctypes.c_void_p(orderbooks_addr), ctypes.POINTER(HotOrderbookSnapshot))

            self._attached = True
            logger.info(f"Attached to HotSpine shared memory: {self.shm_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to attach to shared memory: {e}")
            self.detach()
            return False

    def detach(self):
        """Detach from shared memory"""
        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.header = None
        self.trades_buffer = None
        self.orderbooks_buffer = None
        self._attached = False

    def is_attached(self) -> bool:
        """Check if attached to shared memory"""
        return self._attached

    def get_available_trades(self) -> int:
        """Get number of available trades"""
        if not self._attached:
            return 0

        write_idx = self.header.write_index
        read_idx = self.header.read_index

        if write_idx >= read_idx:
            return write_idx - read_idx
        else:
            return self.header.capacity - read_idx + write_idx

    def get_available_orderbooks(self) -> int:
        """Get number of available orderbooks"""
        if not self._attached or not self.orderbooks_buffer:
            return 0

        write_idx = self.header.orderbook_write_index
        read_idx = self.header.orderbook_read_index

        if write_idx >= read_idx:
            return write_idx - read_idx
        else:
            return self.header.orderbook_capacity - read_idx + write_idx

    def read_trade(self, index: int) -> Optional[Dict[str, Any]]:
        """Read a single trade at the given index"""
        if not self._attached or not self.trades_buffer:
            return None

        try:
            trade = self.trades_buffer[index % self.header.capacity]
            exchange, symbol = self.symbol_mapper.get_symbol_info(trade.symbol_id)

            return {
                "type": "trade",
                "timestamp_exchange": trade.ts_exchange,
                "timestamp_local": trade.ts_local,
                "price": trade.price,
                "size": trade.size,
                "symbol_id": trade.symbol_id,
                "exchange": exchange,
                "symbol": symbol,
                "side": "BUY" if trade.side == 0 else "SELL",
                "datetime": datetime.utcfromtimestamp(trade.ts_exchange / 1_000_000).isoformat()
            }
        except Exception as e:
            logger.error(f"Error reading trade at index {index}: {e}")
            return None

    def read_orderbook(self, index: int) -> Optional[Dict[str, Any]]:
        """Read a single orderbook at the given index"""
        if not self._attached or not self.orderbooks_buffer:
            return None

        try:
            snapshot = self.orderbooks_buffer[index % self.header.orderbook_capacity]
            exchange, symbol = self.symbol_mapper.get_symbol_info(snapshot.symbol_id)

            bids = []
            for i in range(snapshot.bids_count):
                bids.append([snapshot.bids[i].price, snapshot.bids[i].size])

            asks = []
            for i in range(snapshot.asks_count):
                asks.append([snapshot.asks[i].price, snapshot.asks[i].size])

            return {
                "type": "orderbook",
                "timestamp_exchange": snapshot.ts_exchange,
                "timestamp_local": snapshot.ts_local,
                "symbol_id": snapshot.symbol_id,
                "exchange": exchange,
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "bids_count": snapshot.bids_count,
                "asks_count": snapshot.asks_count,
                "datetime": datetime.utcfromtimestamp(snapshot.ts_exchange / 1_000_000).isoformat()
            }
        except Exception as e:
            logger.error(f"Error reading orderbook at index {index}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self._attached:
            return {"attached": False}

        return {
            "attached": True,
            "version": self.header.version,
            "trade_capacity": self.header.capacity,
            "trade_available": self.get_available_trades(),
            "trade_write_index": self.header.write_index,
            "trade_read_index": self.header.read_index,
            "trade_lost": self.header.lost_count,
            "orderbook_capacity": getattr(self.header, 'orderbook_capacity', 0),
            "orderbook_available": self.get_available_orderbooks(),
            "orderbook_write_index": getattr(self.header, 'orderbook_write_index', 0),
            "orderbook_read_index": getattr(self.header, 'orderbook_read_index', 0),
            "orderbook_lost": getattr(self.header, 'orderbook_lost_count', 0),
        }

# ============================================================================
# Data Processor
# ============================================================================

class DataProcessor:
    """Processes and filters HotSpine data"""

    def __init__(self, filters: Dict[str, Any] = None):
        self.filters = filters or {}
        self.trade_count = 0
        self.orderbook_count = 0
        self.processed_data = []

    def process_trade(self, trade: Dict[str, Any]) -> bool:
        """Process a trade with filtering"""
        if not self._matches_filters(trade):
            return False

        self.trade_count += 1
        self.processed_data.append(trade)
        return True

    def process_orderbook(self, orderbook: Dict[str, Any]) -> bool:
        """Process an orderbook with filtering"""
        if not self._matches_filters(orderbook):
            return False

        self.orderbook_count += 1
        self.processed_data.append(orderbook)
        return True

    def _matches_filters(self, data: Dict[str, Any]) -> bool:
        """Check if data matches the configured filters"""
        for key, value in self.filters.items():
            if key not in data:
                continue
            if isinstance(value, str) and value not in str(data[key]):
                return False
            elif data[key] != value:
                return False
        return True

    def aggregate_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate data by symbol"""
        agg = {}
        for item in self.processed_data:
            symbol = f"{item['exchange']}/{item['symbol']}"
            if symbol not in agg:
                agg[symbol] = {
                    "symbol": symbol,
                    "trades": 0,
                    "orderbooks": 0,
                    "total_volume": 0.0,
                    "price_min": float('inf'),
                    "price_max": float('-inf'),
                    "last_price": None,
                    "last_update": None
                }

            if item["type"] == "trade":
                agg[symbol]["trades"] += 1
                agg[symbol]["total_volume"] += item["size"]
                agg[symbol]["price_min"] = min(agg[symbol]["price_min"], item["price"])
                agg[symbol]["price_max"] = max(agg[symbol]["price_max"], item["price"])
                agg[symbol]["last_price"] = item["price"]
                agg[symbol]["last_update"] = item["datetime"]
            elif item["type"] == "orderbook":
                agg[symbol]["orderbooks"] += 1
                if not agg[symbol]["last_update"] or item["datetime"] > agg[symbol]["last_update"]:
                    agg[symbol]["last_update"] = item["datetime"]

        # Clean up infinite values
        for symbol_data in agg.values():
            if symbol_data["price_min"] == float('inf'):
                symbol_data["price_min"] = None
            if symbol_data["price_max"] == float('-inf'):
                symbol_data["price_max"] = None

        return agg

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        return {
            "total_processed": len(self.processed_data),
            "trades_processed": self.trade_count,
            "orderbooks_processed": self.orderbook_count,
            "filters_applied": self.filters
        }

# ============================================================================
# Output Writers
# ============================================================================

class OutputWriter:
    """Handles output formatting and writing"""

    def __init__(self, output_format: str = "json", output_file: Optional[str] = None):
        self.output_format = output_format
        self.output_file = output_file
        self.file_handle = None

        if self.output_file:
            self.file_handle = open(self.output_file, 'w', encoding='utf-8')
            if self.output_format == "csv":
                self.csv_writer = csv.writer(self.file_handle)
                self.csv_writer.writerow([
                    "type", "timestamp_exchange", "timestamp_local", "exchange", "symbol",
                    "price", "size", "side", "bids_count", "asks_count", "datetime"
                ])

    def write_data(self, data: Dict[str, Any]):
        """Write a single data item"""
        if self.output_format == "json":
            self._write_json(data)
        elif self.output_format == "csv":
            self._write_csv(data)
        else:
            print(json.dumps(data, indent=2))

    def _write_json(self, data: Dict[str, Any]):
        """Write data as JSON"""
        output = json.dumps(data, indent=2)
        if self.file_handle:
            self.file_handle.write(output + "\n")
            self.file_handle.flush()
        else:
            print(output)

    def _write_csv(self, data: Dict[str, Any]):
        """Write data as CSV"""
        if data["type"] == "trade":
            row = [
                data["type"],
                data["timestamp_exchange"],
                data["timestamp_local"],
                data["exchange"],
                data["symbol"],
                data["price"],
                data["size"],
                data["side"],
                "",  # bids_count
                "",  # asks_count
                data["datetime"]
            ]
        elif data["type"] == "orderbook":
            row = [
                data["type"],
                data["timestamp_exchange"],
                data["timestamp_local"],
                data["exchange"],
                data["symbol"],
                "",  # price
                "",  # size
                "",  # side
                data["bids_count"],
                data["asks_count"],
                data["datetime"]
            ]
        else:
            return

        if self.file_handle:
            self.csv_writer.writerow(row)
            self.file_handle.flush()
        else:
            print(",".join(str(x) for x in row))

    def close(self):
        """Close output file"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

# ============================================================================
# Main Monitor Class
# ============================================================================

class HotSpineMonitor:
    """Main HotSpine monitoring class"""

    def __init__(self, shm_name: str = "btquant_hotspine", **kwargs):
        self.shm_name = shm_name
        self.shm_path = f"/dev/shm/{shm_name}"
        self.reader = HotSpineReader(self.shm_path)
        self.processor = DataProcessor(kwargs.get("filters", {}))
        self.output_writer = OutputWriter(
            kwargs.get("output_format", "json"),
            kwargs.get("output_file")
        )
        self.poll_interval = kwargs.get("poll_interval", 0.1)
        self.max_items = kwargs.get("max_items", 1000)
        self.running = False
        self.stats_interval = kwargs.get("stats_interval", 10)

    def start(self):
        """Start monitoring"""
        logger.info(f"Starting HotSpine monitor for {self.shm_name}")

        if not self.reader.attach():
            logger.error("Failed to attach to shared memory")
            return False

        self.running = True
        self._monitor_loop()
        return True

    def stop(self):
        """Stop monitoring"""
        logger.info("Stopping HotSpine monitor")
        self.running = False
        self.reader.detach()
        self.output_writer.close()

    def _monitor_loop(self):
        """Main monitoring loop"""
        last_stats_time = time.time()
        items_processed = 0

        try:
            while self.running and items_processed < self.max_items:
                # Read available trades
                available_trades = self.reader.get_available_trades()
                for i in range(min(available_trades, 100)):  # Process up to 100 at a time
                    trade = self.reader.read_trade(self.reader.header.read_index)
                    if trade and self.processor.process_trade(trade):
                        self.output_writer.write_data(trade)
                        items_processed += 1

                # Read available orderbooks
                available_orderbooks = self.reader.get_available_orderbooks()
                for i in range(min(available_orderbooks, 10)):  # Process up to 10 at a time
                    orderbook = self.reader.read_orderbook(self.reader.header.orderbook_read_index)
                    if orderbook and self.processor.process_orderbook(orderbook):
                        self.output_writer.write_data(orderbook)
                        items_processed += 1

                # Print stats periodically
                current_time = time.time()
                if current_time - last_stats_time >= self.stats_interval:
                    self._print_stats()
                    last_stats_time = current_time

                # Sleep to reduce CPU usage
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}")
        finally:
            self._print_final_summary()

    def _print_stats(self):
        """Print current statistics"""
        reader_stats = self.reader.get_stats()
        processor_stats = self.processor.get_summary()

        print("\n" + "="*50)
        print("HotSpine Monitor Statistics")
        print("="*50)
        print(f"Shared Memory: {reader_stats.get('attached', False)}")
        print(f"Trades Available: {reader_stats.get('trade_available', 0)}")
        print(f"Orderbooks Available: {reader_stats.get('orderbook_available', 0)}")
        print(f"Trades Processed: {processor_stats['trades_processed']}")
        print(f"Orderbooks Processed: {processor_stats['orderbooks_processed']}")
        print(f"Total Processed: {processor_stats['total_processed']}")
        print("="*50)

    def _print_final_summary(self):
        """Print final processing summary"""
        summary = self.processor.get_summary()
        agg = self.processor.aggregate_by_symbol()

        print("\n" + "="*60)
        print("FINAL PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Items Processed: {summary['total_processed']}")
        print(f"Trades: {summary['trades_processed']}")
        print(f"Orderbooks: {summary['orderbooks_processed']}")
        print(f"Filters Applied: {summary['filters_applied']}")

        if agg:
            print("\nSymbol Aggregation:")
            for symbol, data in agg.items():
                print(f"  {symbol}: {data['trades']} trades, {data['orderbooks']} orderbooks, vol={data['total_volume']:.2f}")

        print("="*60)

# ============================================================================
# Main Function
# ============================================================================

def parse_filters(filter_str: str) -> Dict[str, Any]:
    """Parse filter string like 'exchange=binance,symbol=BTC-USDT'"""
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

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="HotSpine Monitor Prototype")
    parser.add_argument("--shm-name", default="btquant_hotspine",
                       help="Shared memory segment name (default: btquant_hotspine)")
    parser.add_argument("--output", choices=["json", "csv"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--output-file", help="Output file path (default: stdout)")
    parser.add_argument("--filter", help="Filters as key=value pairs separated by commas")
    parser.add_argument("--max-items", type=int, default=1000,
                       help="Maximum number of items to process (default: 1000)")
    parser.add_argument("--poll-interval", type=float, default=0.1,
                       help="Polling interval in seconds (default: 0.1)")
    parser.add_argument("--stats-interval", type=int, default=10,
                       help="Statistics reporting interval in seconds (default: 10)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Parse filters
    filters = parse_filters(args.filter) if args.filter else {}

    # Create and start monitor
    monitor = HotSpineMonitor(
        shm_name=args.shm_name,
        filters=filters,
        output_format=args.output,
        output_file=args.output_file,
        max_items=args.max_items,
        poll_interval=args.poll_interval,
        stats_interval=args.stats_interval
    )

    try:
        success = monitor.start()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()