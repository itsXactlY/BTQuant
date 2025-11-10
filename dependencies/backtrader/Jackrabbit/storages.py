"""
jrr_websocket/storage.py

Async market data storage with PostgreSQL backend.
Uses asyncpg for high-performance database operations.
"""

import asyncpg
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Storage configuration"""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'jrr_marketdata'
    user: str = 'postgres'
    password: str = ''
    min_size: int = 5  # Min connection pool size
    max_size: int = 20  # Max connection pool size
    timeout: float = 10.0


class MarketDataStorage:
    """
    High-performance async market data storage.

    Supports:
    - OHLCV (candlestick) data
    - Trade execution stream (orderflow)
    - Orderbook snapshots (L2 depth)
    - Efficient batch operations
    - Connection pooling
    """

    def __init__(self, config: StorageConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pool: Optional[asyncpg.Pool] = None
        self.write_buffer: Dict[str, List[Dict]] = {
            'ohlcv': [],
            'trades': [],
            'orderbooks': [],
        }
        self.buffer_flush_interval = 5  # Flush every 5 seconds
        self.buffer_max_size = 10000  # Flush when buffer reaches 10k items

    async def connect(self) -> None:
        """Establish connection pool"""
        try:
            dsn = f"postgresql://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"

            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                timeout=self.config.timeout,
            )

            self.logger.info(f"Connected to {self.config.database}")

            # Initialize tables if needed
            await self._initialize_schema()

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Disconnected from database")

    async def _initialize_schema(self) -> None:
        """Create tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # OHLCV table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    exchange VARCHAR(50) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                    timeframe VARCHAR(10) NOT NULL,
                    open NUMERIC(20, 8) NOT NULL,
                    high NUMERIC(20, 8) NOT NULL,
                    low NUMERIC(20, 8) NOT NULL,
                    close NUMERIC(20, 8) NOT NULL,
                    volume NUMERIC(30, 8) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(timestamp, exchange, symbol, market_type, timeframe)
                );
            """)

            # Trade table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    exchange VARCHAR(50) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                    trade_id VARCHAR(100),
                    price NUMERIC(20, 8) NOT NULL,
                    quantity NUMERIC(30, 8) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    is_buyer_maker BOOLEAN,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Orderbook snapshot table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    exchange VARCHAR(50) NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                    bids JSONB NOT NULL,
                    asks JSONB NOT NULL,
                    checksum VARCHAR(64),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup 
                ON ohlcv(exchange, symbol, market_type, timeframe, timestamp DESC);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_lookup 
                ON trades(exchange, symbol, market_type, timestamp DESC);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orderbook_lookup 
                ON orderbook_snapshots(exchange, symbol, market_type, timestamp DESC);
            """)

            self.logger.info("Schema initialized")

    async def store_ohlcv(self, data: Dict[str, Any]) -> bool:
        """
        Store OHLCV candle data.
        Uses UPSERT to handle duplicate timestamps.
        """
        query = """
            INSERT INTO ohlcv (timestamp, exchange, symbol, market_type, timeframe,
                               open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (timestamp, exchange, symbol, market_type, timeframe)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume;
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    datetime.fromtimestamp(data['timestamp'] / 1000),
                    data['exchange'],
                    data['symbol'],
                    data.get('market_type', 'spot'),
                    data['timeframe'],
                    float(data['open']),
                    float(data['high']),
                    float(data['low']),
                    float(data['close']),
                    float(data['volume']),
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to store OHLCV: {e}")
            return False

    async def store_trade(self, data: Dict[str, Any]) -> bool:
        """Store individual trade execution"""
        query = """
            INSERT INTO trades (timestamp, exchange, symbol, market_type,
                               trade_id, price, quantity, side, is_buyer_maker)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9);
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    datetime.fromtimestamp(data['timestamp'] / 1000),
                    data['exchange'],
                    data['symbol'],
                    data.get('market_type', 'spot'),
                    data.get('trade_id'),
                    float(data['price']),
                    float(data['quantity']),
                    data.get('side', 'unknown'),
                    data.get('is_buyer_maker'),
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to store trade: {e}")
            return False

    async def store_orderbook(self, data: Dict[str, Any]) -> bool:
        """Store orderbook snapshot"""
        import json

        query = """
            INSERT INTO orderbook_snapshots (timestamp, exchange, symbol, market_type,
                                             bids, asks, checksum)
            VALUES ($1, $2, $3, $4, $5, $6, $7);
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    datetime.fromtimestamp(data['timestamp'] / 1000),
                    data['exchange'],
                    data['symbol'],
                    data.get('market_type', 'spot'),
                    json.dumps(data.get('bids', [])),
                    json.dumps(data.get('asks', [])),
                    data.get('checksum'),
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to store orderbook: {e}")
            return False

    async def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Retrieve historical OHLCV data"""

        if end is None:
            end = datetime.utcnow()

        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE exchange = $1 AND symbol = $2 AND timeframe = $3
              AND timestamp >= $4 AND timestamp < $5
            ORDER BY timestamp ASC
        """

        params = [exchange, symbol, timeframe, start, end]

        if limit:
            query += " LIMIT $6"
            params.append(limit)

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to retrieve OHLCV: {e}")
            return []

    async def get_trades(
        self,
        exchange: str,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Retrieve historical trades"""

        if end is None:
            end = datetime.utcnow()

        query = """
            SELECT timestamp, trade_id, price, quantity, side, is_buyer_maker
            FROM trades
            WHERE exchange = $1 AND symbol = $2
              AND timestamp >= $3 AND timestamp < $4
            ORDER BY timestamp ASC
        """

        params = [exchange, symbol, start, end]

        if limit:
            query += " LIMIT $5"
            params.append(limit)

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to retrieve trades: {e}")
            return []

    async def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get latest close price for symbol"""
        query = """
            SELECT close FROM ohlcv
            WHERE exchange = $1 AND symbol = $2 AND timeframe = '1m'
            ORDER BY timestamp DESC
            LIMIT 1;
        """

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, exchange, symbol)
                return float(row['close']) if row else None
        except Exception as e:
            self.logger.error(f"Failed to get latest price: {e}")
            return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.pool.acquire() as conn:
                ohlcv_count = await conn.fetchval("SELECT COUNT(*) FROM ohlcv;")
                trades_count = await conn.fetchval("SELECT COUNT(*) FROM trades;")
                orderbook_count = await conn.fetchval("SELECT COUNT(*) FROM orderbook_snapshots;")

                return {
                    'ohlcv_records': ohlcv_count,
                    'trades_records': trades_count,
                    'orderbook_records': orderbook_count,
                    'pool_size': self.pool.get_size(),
                    'pool_free_size': self.pool.get_idle_size(),
                }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}
