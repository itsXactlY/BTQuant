"""
jrr_websocket/storage_mssql.py

Synchronous market data storage with SQL Server backend.
Uses custom fast_mssql C++ driver for maximum performance.

Updated design:
- Dedicated database for BTQuant live data (default: BTQ_MarketData)
- Table-per-market sharding for OHLCV:
    {exchange}_{symbol}_klines
  e.g. binance_btcusdt_klines
- Trades and orderbook_snapshots remain in shared tables for now.
"""

import fast_mssql
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
import json
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MSSQLConfig:
    """SQL Server configuration"""
    server: str = "localhost"
    # Separate DB for live market data; keep BinanceData for historical/backtests
    database: str = "BTQ_MarketData"
    username: str = "SA"
    password: str = ""
    driver: str = "{ODBC Driver 18 for SQL Server}"
    trust_server_certificate: bool = True

    def get_connection_string(self) -> str:
        """Build ODBC connection string"""
        trust = "yes" if self.trust_server_certificate else "no"
        return (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate={trust};"
        )


class MarketDataStorage:
    """
    High-performance synchronous SQL Server market data storage.

    Uses custom C++ ODBC driver (fast_mssql) for:
    - Direct ODBC calls (no Python DB layer overhead)
    - Connection pooling at C++ level
    - Bulk insert operations
    - Thread-safe operations

    Supports:
    - OHLCV (candlestick) data, sharded per exchange+symbol table
    - Trade execution stream (orderflow)
    - Orderbook snapshots (L2 depth)
    - Efficient batch operations
    """

    def __init__(self, config: MSSQLConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.connection_string = config.get_connection_string()
        self._lock = Lock()
        self._initialized = False

        # Cache of OHLCV table names that are already created
        self._ohlcv_tables: set[str] = set()

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Initialize connection and schema"""
        try:
            # Test connection (this also seeds the connection pool)
            result = fast_mssql.fetch_data_from_db(
                self.connection_string, "SELECT @@VERSION"
            )
            self.logger.info(
                "Connected to SQL Server: %s (database=%s)", result, self.config.database
            )

            # Initialize shared schema (trades, orderbook)
            self._initialize_schema()
            self._initialized = True

        except Exception as e:
            self.logger.error("Connection failed: %s", e)
            raise

    def disconnect(self) -> None:
        """Remove connection from pool"""
        try:
            fast_mssql.remove_connection(self.connection_string)
            self.logger.info("Disconnected from SQL Server")
        except Exception as e:
            self.logger.warning("Disconnect warning: %s", e)

    def _initialize_schema(self) -> None:
        """
        Create shared tables if they don't exist.

        OHLCV is sharded per market, so we only create:
        - trades
        - orderbook_snapshots
        """
        # Trade table
        fast_mssql.execute_non_query(
            self.connection_string,
            """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='trades' AND xtype='U')
            CREATE TABLE trades (
                id BIGINT IDENTITY(1,1) PRIMARY KEY,
                timestamp DATETIME2 NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                trade_id VARCHAR(100),
                price DECIMAL(20, 8) NOT NULL,
                quantity DECIMAL(30, 8) NOT NULL,
                side VARCHAR(10) NOT NULL,
                is_buyer_maker BIT,
                created_at DATETIME2 DEFAULT GETDATE()
            );
        """,
        )

        # Orderbook snapshot table
        fast_mssql.execute_non_query(
            self.connection_string,
            """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='orderbook_snapshots' AND xtype='U')
            CREATE TABLE orderbook_snapshots (
                id BIGINT IDENTITY(1,1) PRIMARY KEY,
                timestamp DATETIME2 NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                bids NVARCHAR(MAX) NOT NULL,
                asks NVARCHAR(MAX) NOT NULL,
                checksum VARCHAR(64),
                created_at DATETIME2 DEFAULT GETDATE()
            );
        """,
        )

        # Create indexes
        fast_mssql.execute_non_query(
            self.connection_string,
            """
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_trades_lookup')
            CREATE INDEX idx_trades_lookup 
            ON trades(exchange, symbol, market_type, timestamp DESC);
        """,
        )

        fast_mssql.execute_non_query(
            self.connection_string,
            """
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_orderbook_lookup')
            CREATE INDEX idx_orderbook_lookup 
            ON orderbook_snapshots(exchange, symbol, market_type, timestamp DESC);
        """,
        )

        self.logger.info("Shared schema (trades/orderbook) initialized")

    # ------------------------------------------------------------------
    # OHLCV table-per-market helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_identifier(value: str) -> str:
        """
        Normalize strings to safe SQL identifiers:
        - lower case
        - non-alphanumeric -> '_'
        - avoid leading digits
        """
        value = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
        if not value:
            value = "unknown"
        if value[0].isdigit():
            value = "_" + value
        return value

    def _ohlcv_table_name(self, exchange: str, symbol: str) -> str:
        """
        Map (exchange, symbol) → table name.

        Example:
            ('binance', 'btcusdt') -> 'binance_btcusdt_klines'
        """
        ex = self._normalize_identifier(exchange)
        sym = self._normalize_identifier(symbol)
        return f"{ex}_{sym}_klines"

    def _ensure_ohlcv_table(self, exchange: str, symbol: str) -> str:
        """Create per-market OHLCV table if it doesn't exist"""
        table = self._ohlcv_table_name(exchange, symbol)
        if table in self._ohlcv_tables:
            return table

        with self._lock:
            if table in self._ohlcv_tables:
                return table

            create_sql = f"""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table}' AND xtype='U')
            CREATE TABLE [{table}] (
                id BIGINT IDENTITY(1,1) PRIMARY KEY,
                timestamp DATETIME2 NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                timeframe VARCHAR(10) NOT NULL,
                [open] DECIMAL(20, 8) NOT NULL,
                high DECIMAL(20, 8) NOT NULL,
                low DECIMAL(20, 8) NOT NULL,
                [close] DECIMAL(20, 8) NOT NULL,
                volume DECIMAL(30, 8) NOT NULL,
                created_at DATETIME2 DEFAULT GETDATE(),
                CONSTRAINT UQ_{table}_ohlcv UNIQUE(timestamp, exchange, symbol, market_type, timeframe)
            );
            """
            fast_mssql.execute_non_query(self.connection_string, create_sql)

            index_sql = f"""
            IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_{table}_lookup')
            CREATE INDEX idx_{table}_lookup 
            ON [{table}](exchange, symbol, market_type, timeframe, timestamp DESC);
            """
            fast_mssql.execute_non_query(self.connection_string, index_sql)

            self._ohlcv_tables.add(table)
            self.logger.info("OHLCV table ready: %s", table)
            return table

    # ------------------------------------------------------------------
    # Store operations
    # ------------------------------------------------------------------
    def store_ohlcv(self, data: Dict[str, Any]) -> bool:
        """
        Store a single OHLCV candle.

        Expected keys:
            exchange, symbol, timestamp(ms), timeframe, open, high, low, close, volume
            market_type (optional, default 'spot')
        """
        try:
            table = self._ensure_ohlcv_table(data["exchange"], data["symbol"])
            ts = datetime.fromtimestamp(data["timestamp"] / 1000.0).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

            query = f"""
                MERGE INTO [{table}] AS target
                USING (SELECT ? AS timestamp, ? AS exchange, ? AS symbol, ? AS market_type, 
                              ? AS timeframe, ? AS [open], ? AS high, ? AS low, 
                              ? AS [close], ? AS volume) AS source
                ON (target.timestamp = source.timestamp 
                    AND target.exchange = source.exchange 
                    AND target.symbol = source.symbol
                    AND target.market_type = source.market_type
                    AND target.timeframe = source.timeframe)
                WHEN MATCHED THEN
                    UPDATE SET 
                        [open] = source.[open],
                        high = source.high,
                        low = source.low,
                        [close] = source.[close],
                        volume = source.volume
                WHEN NOT MATCHED THEN
                    INSERT (timestamp, exchange, symbol, market_type, timeframe,
                            [open], high, low, [close], volume)
                    VALUES (source.timestamp, source.exchange, source.symbol, source.market_type,
                            source.timeframe, source.[open], source.high, source.low,
                            source.[close], source.volume);
            """

            row = [
                ts,
                data["exchange"],
                data["symbol"],
                data.get("market_type", "spot"),
                data["timeframe"],
                str(float(data["open"])),
                str(float(data["high"])),
                str(float(data["low"])),
                str(float(data["close"])),
                str(float(data["volume"])),
            ]

            with self._lock:
                fast_mssql.bulk_insert(self.connection_string, query, [row])
            return True

        except Exception as e:
            self.logger.error("Failed to store OHLCV: %s", e)
            return False

    def store_trade(self, data: Dict[str, Any]) -> bool:
        """Store individual trade execution into shared trades table"""
        query = """
            INSERT INTO trades (timestamp, exchange, symbol, market_type,
                               trade_id, price, quantity, side, is_buyer_maker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        try:
            ts = datetime.fromtimestamp(data["timestamp"] / 1000.0).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

            row = [
                ts,
                data["exchange"],
                data["symbol"],
                data.get("market_type", "spot"),
                str(data.get("trade_id", "")),
                str(float(data["price"])),
                str(float(data["quantity"])),
                data.get("side", "unknown"),
                "1" if data.get("is_buyer_maker") else "0",
            ]

            with self._lock:
                fast_mssql.bulk_insert(self.connection_string, query, [row])
            return True

        except Exception as e:
            self.logger.error("Failed to store trade: %s", e)
            return False

    def store_orderbook(self, data: Dict[str, Any]) -> bool:
        """Store orderbook snapshot into shared table"""
        query = """
            INSERT INTO orderbook_snapshots (timestamp, exchange, symbol, market_type,
                                             bids, asks, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """

        try:
            ts = datetime.fromtimestamp(data["timestamp"] / 1000.0).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

            row = [
                ts,
                data["exchange"],
                data["symbol"],
                data.get("market_type", "spot"),
                json.dumps(data.get("bids", [])),
                json.dumps(data.get("asks", [])),
                data.get("checksum"),
            ]

            with self._lock:
                fast_mssql.bulk_insert(self.connection_string, query, [row])
            return True

        except Exception as e:
            self.logger.error("Failed to store orderbook: %s", e)
            return False

    def bulk_store_ohlcv(self, data_list: List[Dict[str, Any]]) -> int:
        """
        Bulk store OHLCV candles.

        Groups by (exchange, symbol) so each table gets its own MERGE.
        Returns number of rows successfully sent to fast_mssql.
        """
        if not data_list:
            return 0

        grouped: Dict[str, List[List[str]]] = defaultdict(list)

        try:
            # Prepare rows grouped by table
            for data in data_list:
                table = self._ensure_ohlcv_table(data["exchange"], data["symbol"])
                ts = datetime.fromtimestamp(data["timestamp"] / 1000.0).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )
                row = [
                    ts,
                    data["exchange"],
                    data["symbol"],
                    data.get("market_type", "spot"),
                    data["timeframe"],
                    str(float(data["open"])),
                    str(float(data["high"])),
                    str(float(data["low"])),
                    str(float(data["close"])),
                    str(float(data["volume"])),
                ]
                grouped[table].append(row)

            total = 0
            with self._lock:
                for table, rows in grouped.items():
                    query = f"""
                        MERGE INTO [{table}] AS target
                        USING (SELECT ? AS timestamp, ? AS exchange, ? AS symbol, ? AS market_type, 
                                      ? AS timeframe, ? AS [open], ? AS high, ? AS low, 
                                      ? AS [close], ? AS volume) AS source
                        ON (target.timestamp = source.timestamp 
                            AND target.exchange = source.exchange 
                            AND target.symbol = source.symbol
                            AND target.market_type = source.market_type
                            AND target.timeframe = source.timeframe)
                        WHEN MATCHED THEN
                            UPDATE SET 
                                [open] = source.[open],
                                high = source.high,
                                low = source.low,
                                [close] = source.[close],
                                volume = source.volume
                        WHEN NOT MATCHED THEN
                            INSERT (timestamp, exchange, symbol, market_type, timeframe,
                                    [open], high, low, [close], volume)
                            VALUES (source.timestamp, source.exchange, source.symbol, source.market_type,
                                    source.timeframe, source.[open], source.high, source.low,
                                    source.[close], source.volume);
                    """
                    fast_mssql.bulk_insert(self.connection_string, query, rows)
                    total += len(rows)

            return total

        except Exception as e:
            self.logger.error("Failed bulk OHLCV insert: %s", e)
            return 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical OHLCV data from per-market table"""

        if end is None:
            end = datetime.utcnow()

        table = self._ensure_ohlcv_table(exchange, symbol)

        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

        top_clause = f"TOP {limit} " if limit else ""

        query = f"""
            SELECT {top_clause}
                   timestamp, [open], high, low, [close], volume
            FROM [{table}]
            WHERE exchange = '{exchange}' AND symbol = '{symbol}' AND timeframe = '{timeframe}'
              AND timestamp >= '{start_str}' AND timestamp < '{end_str}'
            ORDER BY timestamp ASC;
        """

        try:
            rows = fast_mssql.fetch_data_from_db(self.connection_string, query)
            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            self.logger.error("Failed to retrieve OHLCV: %s", e)
            return []

    def get_trades(
        self,
        exchange: str,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve historical trades from shared trades table"""

        if end is None:
            end = datetime.utcnow()

        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

        top_clause = f"TOP {limit} " if limit else ""

        query = f"""
            SELECT {top_clause}
                   timestamp, trade_id, price, quantity, side, is_buyer_maker
            FROM trades
            WHERE exchange = '{exchange}' AND symbol = '{symbol}'
              AND timestamp >= '{start_str}' AND timestamp < '{end_str}'
            ORDER BY timestamp ASC;
        """

        try:
            rows = fast_mssql.fetch_data_from_db(self.connection_string, query)
            columns = ["timestamp", "trade_id", "price", "quantity", "side", "is_buyer_maker"]
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            self.logger.error("Failed to retrieve trades: %s", e)
            return []

    def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
        """Get latest close price for symbol from per-market 1m table"""
        try:
            table = self._ensure_ohlcv_table(exchange, symbol)
        except Exception as e:
            self.logger.error("Failed to ensure OHLCV table for latest_price: %s", e)
            return None

        query = f"""
            SELECT TOP 1 [close] FROM [{table}]
            WHERE exchange = '{exchange}' AND symbol = '{symbol}' AND timeframe = '1m'
            ORDER BY timestamp DESC;
        """

        try:
            rows = fast_mssql.fetch_data_from_db(self.connection_string, query)
            return float(rows[0][0]) if rows and rows[0][0] != "NULL" else None

        except Exception as e:
            self.logger.error("Failed to get latest price: %s", e)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        For OHLCV, aggregate rows across all *_klines tables.
        """
        stats: Dict[str, Any] = {
            "database": self.config.database,
            "server": self.config.server,
            "pool_size": None,
            "ohlcv_tables": 0,
            "ohlcv_records": 0,
            "trades_records": 0,
            "orderbook_records": 0,
        }

        try:
            # OHLCV: all *_klines tables
            ohlcv_stats_query = """
                SELECT t.name, SUM(p.rows) AS row_count
                FROM sys.tables t
                JOIN sys.partitions p ON t.object_id = p.object_id
                WHERE t.name LIKE '%_klines' AND p.index_id IN (0,1)
                GROUP BY t.name;
            """
            rows = fast_mssql.fetch_data_from_db(self.connection_string, ohlcv_stats_query)
            total_rows = 0
            for name, row_count in rows:
                try:
                    total_rows += int(row_count)
                except Exception:
                    continue
            stats["ohlcv_tables"] = len(rows)
            stats["ohlcv_records"] = total_rows

        except Exception as e:
            self.logger.error("Failed to get OHLCV stats: %s", e)

        try:
            trades_count = fast_mssql.fetch_data_from_db(
                self.connection_string, "SELECT COUNT(*) FROM trades;"
            )[0][0]
            stats["trades_records"] = int(trades_count)
        except Exception as e:
            self.logger.error("Failed to get trades stats: %s", e)

        try:
            orderbook_count = fast_mssql.fetch_data_from_db(
                self.connection_string, "SELECT COUNT(*) FROM orderbook_snapshots;"
            )[0][0]
            stats["orderbook_records"] = int(orderbook_count)
        except Exception as e:
            self.logger.error("Failed to get orderbook stats: %s", e)

        try:
            stats["pool_size"] = fast_mssql.get_pool_size()
        except Exception:
            # older driver versions might not expose this
            pass

        return stats

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def close_all_connections():
        """Close all connections in the pool"""
        fast_mssql.close_all_connections()
