"""
Market Data Types - Python equivalent of market_data_types.h

This module defines the core data structures for market data processing,
equivalent to the C++ market_data_types.h file.
"""

import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


def format_timestamp_micros(timestamp_us: int) -> str:
    """Format timestamp in microseconds to string format"""
    # Convert microseconds to seconds
    timestamp_s = timestamp_us / 1_000_000.0

    # Create datetime object (UTC)
    dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)

    # Format as string with microseconds
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp_us: int  # Exchange timestamp in microseconds (UTC)
    exchange: str
    symbol: str
    market_type: str  # "spot", "perpetual", "futures", etc.
    trade_id: str
    price: float
    quantity: float
    side: str  # "buy" or "sell"
    is_buyer_maker: Optional[bool] = None

    def to_timestamp(self) -> str:
        """Convert timestamp to formatted string"""
        return format_timestamp_micros(self.timestamp_us)

    def to_sql_values(self) -> List[Any]:
        """Convert to SQL insert values"""
        return [
            self.to_timestamp(),
            self.exchange,
            self.symbol,
            self.market_type,
            self.trade_id,
            self.price,
            self.quantity,
            self.side,
            1 if self.is_buyer_maker else 0
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp_us": self.timestamp_us,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "market_type": self.market_type,
            "trade_id": self.trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "side": self.side,
            "is_buyer_maker": self.is_buyer_maker
        }


@dataclass
class OHLCV:
    """Represents an OHLCV candle"""
    timestamp_us: int  # Open time in microseconds (UTC)
    exchange: str
    symbol: str
    market_type: str
    timeframe: str  # "1m", "5m", "1h", etc.
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_timestamp(self) -> str:
        """Convert timestamp to formatted string"""
        return format_timestamp_micros(self.timestamp_us)

    def to_sql_values(self) -> List[Any]:
        """Convert to SQL insert values"""
        return [
            self.to_timestamp(),
            self.exchange,
            self.symbol,
            self.market_type,
            self.timeframe,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume
        ]

    def get_table_name(self) -> str:
        """Get the database table name for this symbol"""
        return f"{self.symbol}_klines"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp_us": self.timestamp_us,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "market_type": self.market_type,
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }


@dataclass
class OrderbookSnapshot:
    """Represents an orderbook snapshot"""
    timestamp_us: int  # Timestamp in microseconds (UTC)
    exchange: str
    symbol: str
    market_type: str
    bids_json: str  # JSON string of bids [[price, qty], ...]
    asks_json: str  # JSON string of asks [[price, qty], ...]
    checksum: Optional[str] = None

    def to_sql_values(self) -> List[Any]:
        """Convert to SQL insert values"""
        return [
            format_timestamp_micros(self.timestamp_us),
            self.exchange,
            self.symbol,
            self.market_type,
            self.bids_json,
            self.asks_json,
            self.checksum or ""
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp_us": self.timestamp_us,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "market_type": self.market_type,
            "bids_json": self.bids_json,
            "asks_json": self.asks_json,
            "checksum": self.checksum
        }


# Type aliases for convenience
TradeList = List[Trade]
OHLCVList = List[OHLCV]
OrderbookList = List[OrderbookSnapshot]