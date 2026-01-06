"""
Candle Aggregator - Python equivalent of candle_aggregator.h/.cpp

This module provides OHLCV candle aggregation from trade data,
equivalent to the C++ CandleAggregator class.
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .market_data_types import Trade, OHLCV


def get_current_timestamp() -> str:
    """Get current timestamp formatted for logging"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond:06d}"


@dataclass
class CandleState:
    """Internal state for tracking candle aggregation"""
    open_time_ms: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    market_type: str = ""
    is_initialized: bool = False


class CandleAggregator:
    """
    Aggregates trade data into OHLCV candles for multiple timeframes.

    This class maintains active candle states for each exchange/symbol/timeframe
    combination and generates completed candles when timeframes advance.
    """

    def __init__(self, timeframes: List[str]):
        """
        Initialize the candle aggregator.

        Args:
            timeframes: List of timeframe strings (e.g., ["1m", "5m", "1h"])
        """
        self.timeframes = timeframes
        self.active_candles: Dict[str, CandleState] = {}  # key -> CandleState
        self.completed_candles: List[OHLCV] = []
        self.mutex = threading.Lock()

    @staticmethod
    def make_key(exchange: str, symbol: str, timeframe: str) -> str:
        """Create a unique key for candle state lookup"""
        return f"{exchange}:{symbol}:{timeframe}"

    def timeframe_millis(self, tf: str) -> int:
        """
        Convert timeframe string to milliseconds.

        Args:
            tf: Timeframe string (e.g., "1m", "5m", "1h", "1d")

        Returns:
            Timeframe duration in microseconds
        """
        if not tf:
            return 60_000_000  # Default 60 seconds in microseconds

        unit = tf[-1].lower()
        try:
            val = int(tf[:-1])
        except ValueError:
            return 60_000_000  # Default on parse error

        # Convert to microseconds
        if unit == 's':
            return val * 1_000_000
        elif unit == 'm':
            return val * 60 * 1_000_000
        elif unit == 'h':
            return val * 60 * 60 * 1_000_000
        elif unit == 'd':
            return val * 24 * 60 * 60 * 1_000_000
        else:
            return val * 60 * 1_000_000  # Default to minutes

    def align_timestamp(self, timestamp_us: int, timeframe: str) -> int:
        """
        Align timestamp to the start of the timeframe bucket.

        Args:
            timestamp_us: Timestamp in microseconds
            timeframe: Timeframe string

        Returns:
            Aligned timestamp in microseconds
        """
        tf_us = self.timeframe_millis(timeframe)
        return (timestamp_us // tf_us) * tf_us

    def process_trade(self, trade: Trade) -> None:
        """
        Process a trade and update candle aggregations.

        Args:
            trade: Trade data to process
        """
        with self.mutex:
            print(f"[{get_current_timestamp()}][DEBUG] CandleAggregator: Processing trade for aggregation")
            print(f"[{get_current_timestamp()}][DEBUG]   Exchange: {trade.exchange}, Symbol: {trade.symbol}, Price: {trade.price}, Quantity: {trade.quantity}, Timestamp: {trade.timestamp_us}")

            for tf in self.timeframes:
                bucket = self.align_timestamp(trade.timestamp_us, tf)
                key = self.make_key(trade.exchange, trade.symbol, tf)

                print(f"[{get_current_timestamp()}][DEBUG]   Timeframe: {tf}, Bucket: {bucket}, Key: {key}")

                cs = self.active_candles.get(key)
                if cs is None:
                    cs = CandleState()
                    self.active_candles[key] = cs

                # New candle or first trade in this timeframe
                if not cs.is_initialized:
                    print(f"[{get_current_timestamp()}][DEBUG]   Creating new candle for timeframe {tf}")
                    cs.open_time_ms = bucket
                    cs.open = trade.price
                    cs.high = trade.price
                    cs.low = trade.price
                    cs.close = trade.price
                    cs.volume = trade.quantity
                    cs.market_type = trade.market_type
                    cs.is_initialized = True
                    continue

                # Bucket advanced => close old candle, start a new one
                if bucket > cs.open_time_ms:
                    print(f"[{get_current_timestamp()}][DEBUG]   Closing completed candle for timeframe {tf}")
                    print(f"[{get_current_timestamp()}][DEBUG]     Open: {cs.open}, High: {cs.high}, Low: {cs.low}, Close: {cs.close}, Volume: {cs.volume}")

                    c = OHLCV(
                        timestamp_us=cs.open_time_ms,
                        exchange=trade.exchange,
                        symbol=trade.symbol,
                        market_type=cs.market_type,
                        timeframe=tf,
                        open=cs.open,
                        high=cs.high,
                        low=cs.low,
                        close=cs.close,
                        volume=cs.volume
                    )
                    self.completed_candles.append(c)

                    print(f"[{get_current_timestamp()}][DEBUG]   Starting new candle for timeframe {tf}")
                    cs.open_time_ms = bucket
                    cs.open = trade.price
                    cs.high = trade.price
                    cs.low = trade.price
                    cs.close = trade.price
                    cs.volume = trade.quantity
                else:
                    # Same bucket: update OHLCV
                    print(f"[{get_current_timestamp()}][DEBUG]   Updating existing candle for timeframe {tf}")
                    print(f"[{get_current_timestamp()}][DEBUG]     Current High: {cs.high} -> {max(cs.high, trade.price)}")
                    print(f"[{get_current_timestamp()}][DEBUG]     Current Low: {cs.low} -> {min(cs.low, trade.price)}")
                    print(f"[{get_current_timestamp()}][DEBUG]     Current Volume: {cs.volume} -> {cs.volume + trade.quantity}")

                    cs.high = max(cs.high, trade.price)
                    cs.low = min(cs.low, trade.price)
                    cs.close = trade.price
                    cs.volume += trade.quantity

            print(f"[{get_current_timestamp()}][DEBUG] CandleAggregator: Completed trade processing for aggregation")

    def get_all_completed_candles(self) -> List[OHLCV]:
        """
        Get all completed candles and clear the buffer.

        Returns:
            List of completed OHLCV candles
        """
        with self.mutex:
            result = self.completed_candles[:]
            self.completed_candles.clear()
            return result

    def flush_all(self) -> List[OHLCV]:
        """
        Force close all active candles and return them.

        Returns:
            List of all candles (completed + active)
        """
        with self.mutex:
            print(f"[{get_current_timestamp()}][DEBUG] CandleAggregator: Flushing all active candles")
            print(f"[{get_current_timestamp()}][DEBUG]   Active candles count: {len(self.active_candles)}")

            for key, cs in self.active_candles.items():
                if not cs.is_initialized:
                    continue

                print(f"[{get_current_timestamp()}][DEBUG]   Flushing candle: {key}")
                print(f"[{get_current_timestamp()}][DEBUG]     Open: {cs.open}, High: {cs.high}, Low: {cs.low}, Close: {cs.close}, Volume: {cs.volume}")

                # Extract components from key ("ex:sym:tf")
                parts = key.split(':')
                if len(parts) != 3:
                    continue

                exchange, symbol, tf = parts

                c = OHLCV(
                    timestamp_us=cs.open_time_ms,
                    exchange=exchange,
                    symbol=symbol,
                    market_type=cs.market_type,
                    timeframe=tf,
                    open=cs.open,
                    high=cs.high,
                    low=cs.low,
                    close=cs.close,
                    volume=cs.volume
                )
                self.completed_candles.append(c)

            print(f"[{get_current_timestamp()}][DEBUG]   Completed candles after flush: {len(self.completed_candles)}")

            self.active_candles.clear()

            result = self.completed_candles[:]
            self.completed_candles.clear()

            print(f"[{get_current_timestamp()}][DEBUG] CandleAggregator: Flush completed. Returning {len(result)} candles")

            return result

    def validate_candle_aggregation(self) -> None:
        """Validate the current state of candle aggregation"""
        with self.mutex:
            print(f"[{get_current_timestamp()}][INFO] CandleAggregator: Validating candle aggregation state")
            print(f"[{get_current_timestamp()}][INFO]   Active candles: {len(self.active_candles)}")
            print(f"[{get_current_timestamp()}][INFO]   Completed candles: {len(self.completed_candles)}")

            if not self.active_candles:
                print(f"[{get_current_timestamp()}][WARNING]   No active candles found")
                print(f"[{get_current_timestamp()}][WARNING]   This could indicate:")
                print(f"[{get_current_timestamp()}][WARNING]     - No trades processed yet")
                print(f"[{get_current_timestamp()}][WARNING]     - Trades not reaching candle aggregator")
                print(f"[{get_current_timestamp()}][WARNING]     - Data flow disruption")
            else:
                print(f"[{get_current_timestamp()}][INFO]   Active candle details:")
                for key, cs in self.active_candles.items():
                    print(f"[{get_current_timestamp()}][INFO]     {key}")
                    print(f"[{get_current_timestamp()}][INFO]       Open: {cs.open}, High: {cs.high}, Low: {cs.low}, Close: {cs.close}, Volume: {cs.volume}, Initialized: {'YES' if cs.is_initialized else 'NO'}")

            print(f"[{get_current_timestamp()}][INFO]   Timeframes configured: {len(self.timeframes)}")
            for tf in self.timeframes:
                print(f"[{get_current_timestamp()}][INFO]     - {tf}")

            print(f"[{get_current_timestamp()}][INFO] CandleAggregator: Validation completed")