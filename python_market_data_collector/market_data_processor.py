"""
Market Data Processor - Python equivalent of market_data_processor.cpp

This module processes incoming market data from exchanges, manages buffering,
and coordinates data flow to storage systems.
"""

import threading
import time
import queue
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from .market_data_types import Trade, OHLCV, OrderbookSnapshot
from .candle_aggregator import CandleAggregator
from .hotspine_writer import HotSpineWriter
from .config_types import DebugConfig
from .utilities import get_current_timestamp, now_micros


@dataclass
class ProcessorStats:
    """Statistics for the market data processor"""
    trades_received: int = 0
    trades_inserted: int = 0
    candles_generated: int = 0
    candles_inserted: int = 0
    orderbooks_received: int = 0
    orderbooks_inserted: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    trades_per_sec: float = 0.0
    orderbooks_per_sec: float = 0.0


class MarketDataProcessor:
    """
    Processes market data from exchanges and manages data flow.

    This class handles trade and orderbook data, performs candle aggregation,
    manages buffering, and coordinates with storage systems.
    """

    def __init__(self,
                 candle_agg: Optional[CandleAggregator] = None,
                 hotspine_writer: Optional[HotSpineWriter] = None,
                 enable_exclusive_hotspine: bool = False,
                 debug_config: Optional[DebugConfig] = None):
        """
        Initialize the market data processor.

        Args:
            candle_agg: Candle aggregator for OHLCV generation
            hotspine_writer: HotSpine writer for shared memory storage
            enable_exclusive_hotspine: Whether to use HotSpine exclusively
            debug_config: Debug configuration
        """
        self.candle_agg = candle_agg
        self.hotspine_writer = hotspine_writer
        self.enable_exclusive_hotspine = enable_exclusive_hotspine
        self.debug_config = debug_config or DebugConfig()

        # Buffers
        self.trade_buffer: List[Trade] = []
        self.candle_buffer: List[OHLCV] = []
        self.orderbook_buffer: List[OrderbookSnapshot] = []

        # Buffer limits
        self.max_trade_buffer_size = 1000
        self.max_candle_buffer_size = 200
        self.max_orderbook_buffer_size = 100

        # Threading
        self.buffer_mutex = threading.Lock()
        self.running = False

        # Statistics
        self.stats = ProcessorStats()
        self.active_pairs: set = set()

        # Performance tracking
        self.last_stats_ts_us = 0
        self.trades_last_window = 0
        self.orderbooks_last_window = 0

        # Callbacks for external processing
        self.trade_callbacks: List[Callable[[Trade], None]] = []
        self.orderbook_callbacks: List[Callable[[OrderbookSnapshot], None]] = []

        print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: Initialized")

    def set_buffer_limits(self, max_trades: int, max_candles: int, max_orderbooks: int) -> None:
        """Set buffer size limits"""
        self.max_trade_buffer_size = max_trades
        self.max_candle_buffer_size = max_candles
        self.max_orderbook_buffer_size = max_orderbooks

        print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: Buffer limits set - Trades: {max_trades}, Candles: {max_candles}, Orderbooks: {max_orderbooks}")

    def add_trade_callback(self, callback: Callable[[Trade], None]) -> None:
        """Add a callback for trade processing"""
        self.trade_callbacks.append(callback)

    def add_orderbook_callback(self, callback: Callable[[OrderbookSnapshot], None]) -> None:
        """Add a callback for orderbook processing"""
        self.orderbook_callbacks.append(callback)

    def process_trade(self, trade: Trade) -> None:
        """Process an incoming trade"""
        if self.debug_config.enabled and self.debug_config.verbose_logging:
            print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: Processing trade - {trade.exchange}:{trade.symbol} @ {trade.price}")

        # Update statistics
        with self.buffer_mutex:
            self.trade_buffer.append(trade)
            self.active_pairs.add(f"{trade.exchange}:{trade.symbol}:{trade.market_type}")
            self.stats.trades_received += 1

        # Process trade through candle aggregator
        if self.candle_agg:
            self.candle_agg.process_trade(trade)

        # Write to HotSpine if available
        if self.hotspine_writer:
            if not self.hotspine_writer.write_trade(trade):
                print(f"[{get_current_timestamp()}][ERROR] MarketDataProcessor: Failed to write trade to HotSpine")

        # Call external callbacks
        for callback in self.trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                print(f"[{get_current_timestamp()}][ERROR] MarketDataProcessor: Trade callback error: {e}")

        # Calculate latency (if not in exclusive HotSpine mode)
        if not self.enable_exclusive_hotspine:
            latency_ms = (now_micros() - trade.timestamp_us) / 1000.0
            self.stats.avg_latency_ms = 0.99 * self.stats.avg_latency_ms + 0.01 * latency_ms

        # Check if we need to flush trades
        self.flush_trades_if_needed()

    def process_orderbook(self, orderbook: OrderbookSnapshot) -> None:
        """Process an incoming orderbook snapshot"""
        if self.debug_config.enabled and self.debug_config.verbose_logging:
            print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: Processing orderbook - {orderbook.exchange}:{orderbook.symbol}")

        # Update statistics
        with self.buffer_mutex:
            self.orderbook_buffer.append(orderbook)
            self.active_pairs.add(f"{orderbook.exchange}:{orderbook.symbol}:{orderbook.market_type}")
            self.stats.orderbooks_received += 1

        # Call external callbacks
        for callback in self.orderbook_callbacks:
            try:
                callback(orderbook)
            except Exception as e:
                print(f"[{get_current_timestamp()}][ERROR] MarketDataProcessor: Orderbook callback error: {e}")

        # Check if we need to flush orderbooks
        self.flush_orderbooks_if_needed()

    def flush_trades_if_needed(self, force: bool = False) -> None:
        """Flush trades to database if buffer is full or forced"""
        if self.enable_exclusive_hotspine:
            if self.debug_config.enabled and self.debug_config.verbose_logging:
                print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: Skipping trade flush in exclusive HotSpine mode")
            return

        batch = []
        with self.buffer_mutex:
            if self.debug_config.enabled and self.debug_config.buffer_debug:
                print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: Checking trade buffer - Size: {len(self.trade_buffer)}, Max: {self.max_trade_buffer_size}")

            if not force and len(self.trade_buffer) < self.max_trade_buffer_size:
                return

            batch = self.trade_buffer[:]
            self.trade_buffer.clear()

        if not batch:
            return

        if self.debug_config.enabled and self.debug_config.verbose_logging:
            print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: Flushing {len(batch)} trades")

        # In a real implementation, this would insert to database
        # For now, just update statistics
        self.stats.trades_inserted += len(batch)

    def flush_candles_if_needed(self, force: bool = False) -> None:
        """Flush completed candles to database"""
        if self.enable_exclusive_hotspine or not self.candle_agg:
            return

        # Get newly completed candles
        newly_completed = self.candle_agg.get_all_completed_candles()
        self.stats.candles_generated += len(newly_completed)

        with self.buffer_mutex:
            self.candle_buffer.extend(newly_completed)

            if not force and len(self.candle_buffer) < self.max_candle_buffer_size:
                return

            if not self.candle_buffer:
                return

            # Group by table (symbol)
            by_table: Dict[str, List[OHLCV]] = {}
            for candle in self.candle_buffer:
                table_name = candle.get_table_name()
                if table_name not in by_table:
                    by_table[table_name] = []
                by_table[table_name].append(candle)

            self.candle_buffer.clear()

        # In a real implementation, this would bulk insert to database
        for table_name, candles in by_table.items():
            if self.debug_config.enabled and self.debug_config.verbose_logging:
                print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: Would insert {len(candles)} candles into {table_name}")
            self.stats.candles_inserted += len(candles)

    def flush_orderbooks_if_needed(self, force: bool = False) -> None:
        """Flush orderbooks to database if buffer is full or forced"""
        if self.enable_exclusive_hotspine:
            if self.debug_config.enabled and self.debug_config.verbose_logging:
                print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: Skipping orderbook flush in exclusive HotSpine mode")
            return

        batch = []
        with self.buffer_mutex:
            if self.debug_config.enabled and self.debug_config.buffer_debug:
                print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: Checking orderbook buffer - Size: {len(self.orderbook_buffer)}, Max: {self.max_orderbook_buffer_size}")

            if not force and len(self.orderbook_buffer) < self.max_orderbook_buffer_size:
                return

            batch = self.orderbook_buffer[:]
            self.orderbook_buffer.clear()

        if not batch:
            return

        if self.debug_config.enabled and self.debug_config.verbose_logging:
            print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: Flushing {len(batch)} orderbooks")

        # In a real implementation, this would insert to database
        self.stats.orderbooks_inserted += len(batch)

    def flush_buffers(self) -> None:
        """Force flush all buffers"""
        if self.debug_config.enabled and self.debug_config.flush_debug:
            print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: Force flushing all buffers")

        if self.enable_exclusive_hotspine:
            if self.hotspine_writer:
                self.hotspine_writer.flush_batch()
        else:
            self.flush_trades_if_needed(True)
            self.flush_candles_if_needed(True)
            self.flush_orderbooks_if_needed(True)

    def get_stats(self) -> ProcessorStats:
        """Get current statistics with rate calculations"""
        stats = ProcessorStats()
        stats.trades_received = self.stats.trades_received
        stats.trades_inserted = self.stats.trades_inserted
        stats.candles_generated = self.stats.candles_generated
        stats.candles_inserted = self.stats.candles_inserted
        stats.orderbooks_received = self.stats.orderbooks_received
        stats.orderbooks_inserted = self.stats.orderbooks_inserted
        stats.errors = self.stats.errors
        stats.avg_latency_ms = self.stats.avg_latency_ms

        # Calculate rates
        now_us = now_micros()
        last_us = self.last_stats_ts_us
        if last_us > 0:
            dt_sec = (now_us - last_us) / 1_000_000.0
            if dt_sec > 0.1:
                tr_prev = self.trades_last_window
                ob_prev = self.orderbooks_last_window

                tr_rate = (stats.trades_received - tr_prev) / dt_sec
                ob_rate = (stats.orderbooks_received - ob_prev) / dt_sec

                stats.trades_per_sec = tr_rate
                stats.orderbooks_per_sec = ob_rate

                self.trades_last_window = stats.trades_received
                self.orderbooks_last_window = stats.orderbooks_received

        self.last_stats_ts_us = now_us

        return stats

    def get_pair_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics per trading pair"""
        # This would track trades and orderbooks per pair
        # For now, return a simple structure
        return {}

    def get_stats_json(self) -> str:
        """Get statistics as JSON string"""
        import json
        stats = self.get_stats()
        pair_stats = self.get_pair_stats()

        data = {
            "trades_received": stats.trades_received,
            "trades_inserted": stats.trades_inserted,
            "candles_generated": stats.candles_generated,
            "candles_inserted": stats.candles_inserted,
            "orderbooks_received": stats.orderbooks_received,
            "orderbooks_inserted": stats.orderbooks_inserted,
            "errors": stats.errors,
            "avg_latency_ms": stats.avg_latency_ms,
            "trades_per_sec": stats.trades_per_sec,
            "orderbooks_per_sec": stats.orderbooks_per_sec,
            "pairs": pair_stats
        }

        return json.dumps(data)

    def log_web_socket_data_flow_stats(self) -> None:
        """Log WebSocket data flow statistics"""
        stats = self.get_stats()

        print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: WebSocket Data Flow Statistics:")
        print(f"[{get_current_timestamp()}][INFO]   Data Reception Rates:")
        print(f"[{get_current_timestamp()}][INFO]     Trades received: {stats.trades_received} ({stats.trades_per_sec:.1f} /sec)")
        print(f"[{get_current_timestamp()}][INFO]     Orderbooks received: {stats.orderbooks_received} ({stats.orderbooks_per_sec:.1f} /sec)")
        print(f"[{get_current_timestamp()}][INFO]   Data Processing Status:")
        print(f"[{get_current_timestamp()}][INFO]     Trades inserted: {stats.trades_inserted}")
        print(f"[{get_current_timestamp()}][INFO]     Orderbooks inserted: {stats.orderbooks_inserted}")
        print(f"[{get_current_timestamp()}][INFO]     Candles generated: {stats.candles_generated}")
        print(f"[{get_current_timestamp()}][INFO]     Candles inserted: {stats.candles_inserted}")
        print(f"[{get_current_timestamp()}][INFO]   Error Statistics:")
        print(f"[{get_current_timestamp()}][INFO]     Total errors: {stats.errors}")
        print(f"[{get_current_timestamp()}][INFO]     Average latency: {stats.avg_latency_ms:.2f} ms")

        if stats.trades_per_sec > 0 or stats.orderbooks_per_sec > 0:
            print(f"[{get_current_timestamp()}][INFO]     WebSocket connection: HEALTHY (receiving data)")
        else:
            print(f"[{get_current_timestamp()}][WARNING]     WebSocket connection: NO DATA RECEIVED")

    def validate_web_socket_data_flow(self) -> None:
        """Validate WebSocket data flow"""
        stats = self.get_stats()

        if stats.trades_received == 0 and stats.orderbooks_received == 0:
            print(f"[{get_current_timestamp()}][ERROR] MarketDataProcessor: WebSocket Data Flow Validation FAILED!")
            print(f"[{get_current_timestamp()}][ERROR]   No data received from WebSocket connection!")
            return

        if stats.trades_per_sec < 0.1 and stats.orderbooks_per_sec < 0.1:
            print(f"[{get_current_timestamp()}][WARNING] MarketDataProcessor: Low data reception rates detected!")
            print(f"[{get_current_timestamp()}][WARNING]   Trades: {stats.trades_per_sec:.1f} /sec, Orderbooks: {stats.orderbooks_per_sec:.1f} /sec")

        if stats.errors > 0:
            print(f"[{get_current_timestamp()}][WARNING] MarketDataProcessor: Data processing errors detected: {stats.errors}")

        print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: WebSocket Data Flow Validation PASSED!")
        print(f"[{get_current_timestamp()}][INFO]   Healthy data reception detected:")
        print(f"[{get_current_timestamp()}][INFO]     Trades: {stats.trades_received} received, {stats.trades_per_sec:.1f} /sec")
        print(f"[{get_current_timestamp()}][INFO]     Orderbooks: {stats.orderbooks_received} received, {stats.orderbooks_per_sec:.1f} /sec")

    def validate_candle_aggregation(self) -> None:
        """Validate candle aggregation state"""
        if self.candle_agg:
            self.candle_agg.validate_candle_aggregation()

    def add_web_socket_debugging(self) -> None:
        """Add comprehensive WebSocket debugging"""
        stats = self.get_stats()
        pair_stats = self.get_pair_stats()

        print(f"[{get_current_timestamp()}][DEBUG] MarketDataProcessor: WebSocket Debugging Information:")
        print(f"[{get_current_timestamp()}][DEBUG]   Data Reception Status:")
        print(f"[{get_current_timestamp()}][DEBUG]     Trades received: {stats.trades_received} ({stats.trades_per_sec:.1f}/sec)")
        print(f"[{get_current_timestamp()}][DEBUG]     Orderbooks received: {stats.orderbooks_received} ({stats.orderbooks_per_sec:.1f}/sec)")
        print(f"[{get_current_timestamp()}][DEBUG]     Errors: {stats.errors}")
        print(f"[{get_current_timestamp()}][DEBUG]     Average latency: {stats.avg_latency_ms:.2f} ms")
        print(f"[{get_current_timestamp()}][DEBUG]   Active Trading Pairs: {len(pair_stats)}")

        print(f"[{get_current_timestamp()}][DEBUG]   Buffer Status:")
        with self.buffer_mutex:
            print(f"[{get_current_timestamp()}][DEBUG]     Trade buffer: {len(self.trade_buffer)}/{self.max_trade_buffer_size}")
            print(f"[{get_current_timestamp()}][DEBUG]     Candle buffer: {len(self.candle_buffer)}/{self.max_candle_buffer_size}")
            print(f"[{get_current_timestamp()}][DEBUG]     Orderbook buffer: {len(self.orderbook_buffer)}/{self.max_orderbook_buffer_size}")

        print(f"[{get_current_timestamp()}][DEBUG]   HotSpine Status:")
        if self.hotspine_writer:
            hs_stats = self.hotspine_writer.get_stats()
            print(f"[{get_current_timestamp()}][DEBUG]     Trades written: {hs_stats['trades_written']}")
            print(f"[{get_current_timestamp()}][DEBUG]     Batches flushed: {hs_stats['batches_flushed']}")
            print(f"[{get_current_timestamp()}][DEBUG]     Current batch: {hs_stats['current_batch_size']}/{hs_stats['batch_size']}")

    def start(self) -> None:
        """Start the processor"""
        self.running = True
        print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: Started")

    def stop(self) -> None:
        """Stop the processor and flush remaining data"""
        self.running = False
        self.flush_buffers()
        print(f"[{get_current_timestamp()}][INFO] MarketDataProcessor: Stopped")