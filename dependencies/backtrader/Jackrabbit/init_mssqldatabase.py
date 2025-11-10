"""
scripts/init_mssql_databases.py

Initialize SQL Server databases for BTQuant market data and Optuna.

- creates database (if missing)
- creates global tables: trades, orderbook_snapshots
- optionally:
    * a single global OHLCV table  (mode="global")
    * OR per-pair OHLCV tables     (mode="per_pair")

IMPORTANT:
This script only creates structure (DDL). It does NOT insert any data.
Your collectors / ccapi / C++ processes are responsible for filling the tables.
"""

import pyodbc
import logging
from typing import Optional, Iterable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class SQLServerInitializer:
    """Initialize SQL Server databases for BTQ market data"""

    def __init__(
        self,
        server: str = "localhost",
        username: str = "SA",
        password: str = "",
        driver: str = "{ODBC Driver 18 for SQL Server}",
    ):
        self.server = server
        self.username = username
        self.password = password
        self.driver = driver

        # master connection string (for creating databases)
        self.master_conn_string = (
            f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE=master;"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_connection_string(self, database: str) -> str:
        """Build connection string for a specific database"""
        return (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"
        )

    # ------------------------------------------------------------------
    # Database creation
    # ------------------------------------------------------------------
    def create_database(self, database_name: str) -> bool:
        """Create database if it doesn't exist"""
        try:
            conn = pyodbc.connect(self.master_conn_string)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sys.databases WHERE name = ?", database_name)
            exists = cursor.fetchone() is not None
            cursor.close()
            conn.close()

            if exists:
                logger.info(f"✓ Database '{database_name}' already exists")
                return True

            conn = pyodbc.connect(self.master_conn_string)
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE [{database_name}]")
            cursor.close()
            conn.close()

            logger.info(f"✓ Database '{database_name}' created")
            return True

        except Exception as e:
            logger.error(f"Failed to create database '{database_name}': {e}")
            return False

    # ------------------------------------------------------------------
    # Schema creation
    # ------------------------------------------------------------------
    def initialize_market_data_schema(
        self,
        database: str = "BTQ_MarketData",
        mode: str = "per_pair",
        table_pattern: str = "{symbol}_klines",
        create_global_ohlcv: bool = False,
    ) -> bool:
        """
        Initialize base schema for market data.

        :param mode: "global" or "per_pair"
        :param table_pattern: pattern for per-pair OHLCV tables
                              placeholders: {exchange}, {symbol}
        :param create_global_ohlcv:
            If True and mode="global", create 'ohlcv' table.
            If False, only trades + orderbook tables are created.
        """
        try:
            conn_string = self._get_connection_string(database)
            conn = pyodbc.connect(conn_string, autocommit=False)
            cursor = conn.cursor()

            logger.info(f"Initializing schema in '{database}' (mode={mode})...")

            # optionally global OHLCV table
            if mode == "global" and create_global_ohlcv:
                logger.info("  Creating global ohlcv table...")
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ohlcv' AND xtype='U')
                    CREATE TABLE ohlcv (
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
                        created_at DATETIME2 DEFAULT SYSUTCDATETIME(),
                        CONSTRAINT UQ_ohlcv UNIQUE(timestamp, exchange, symbol, market_type, timeframe)
                    );
                    """
                )

            # Trades table
            logger.info("  Creating trades table...")
            cursor.execute(
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
                    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
                );
                """
            )

            # Orderbook snapshots table
            logger.info("  Creating orderbook_snapshots table...")
            cursor.execute(
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
                    created_at DATETIME2 DEFAULT SYSUTCDATETIME()
                );
                """
            )

            conn.commit()

            # Indexes
            logger.info("  Creating indexes...")

            if mode == "global" and create_global_ohlcv:
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_ohlcv_lookup')
                    CREATE INDEX idx_ohlcv_lookup 
                    ON ohlcv(exchange, symbol, market_type, timeframe, timestamp DESC);
                    """
                )

            cursor.execute(
                """
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_trades_lookup')
                CREATE INDEX idx_trades_lookup 
                ON trades(exchange, symbol, market_type, timestamp DESC);
                """
            )

            cursor.execute(
                """
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_orderbook_lookup')
                CREATE INDEX idx_orderbook_lookup 
                ON orderbook_snapshots(exchange, symbol, market_type, timestamp DESC);
                """
            )

            conn.commit()
            conn.close()

            logger.info(f"✓ Base schema initialized in '{database}'")
            logger.info(f"  (per-pair OHLCV tables follow pattern: {table_pattern})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False

    def create_ohlcv_table_for_pair(
        self,
        database: str,
        exchange: str,
        symbol: str,
        table_pattern: str = "{symbol}_klines",
        schema: str = "dbo",
    ) -> bool:
        """
        Create OHLCV table for a specific (exchange, symbol) according to pattern.

        Example pattern:
            "{symbol}_klines"              -> "BTCUSDT_klines"
            "{exchange}_{symbol}_klines"   -> "binance_BTCUSDT_klines"
        """
        try:
            conn_string = self._get_connection_string(database)
            conn = pyodbc.connect(conn_string, autocommit=False)
            cursor = conn.cursor()

            clean_symbol = symbol.replace("/", "").replace("-", "").replace("_", "")
            ex = exchange.lower()

            table_name = table_pattern.format(exchange=ex, symbol=clean_symbol)
            full_name = f"{schema}.[{table_name}]"

            logger.info(f"  Ensuring OHLCV table {full_name} ...")

            cursor.execute(
                f"""
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
                BEGIN
                    CREATE TABLE {full_name} (
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
                        created_at DATETIME2 DEFAULT SYSUTCDATETIME(),
                        CONSTRAINT UQ_{table_name} UNIQUE(timestamp, exchange, symbol, market_type, timeframe)
                    );

                    CREATE INDEX IX_{table_name}_lookup
                        ON {full_name}(exchange, symbol, market_type, timeframe, timestamp DESC);
                END
                """
            )

            conn.commit()
            conn.close()

            logger.info(f"  ✓ OHLCV table ready: {full_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create OHLCV table for {exchange} {symbol}: {e}")
            return False

    # ------------------------------------------------------------------
    # Optuna DB (unverändert)
    # ------------------------------------------------------------------
    def initialize_optuna_database(self, database: str = "OptunaBT") -> bool:
        """
        Initialize Optuna database.
        Optuna will create its own tables on first use.
        """
        try:
            if not self.create_database(database):
                return False

            optuna_url = (
                f"mssql+pyodbc://{self.username}:{self.password}"
                f"@{self.server}/{database}"
                f"?driver=ODBC+Driver+18+for+SQL+Server"
                f"&TrustServerCertificate=yes"
            )

            logger.info(f"✓ Optuna database '{database}' ready")
            logger.info(f"  Optuna storage URL: {optuna_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Optuna database: {e}")
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def verify_connection(self) -> bool:
        """Test connection to SQL Server"""
        try:
            conn = pyodbc.connect(self.master_conn_string)
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            logger.info("✓ Connected to SQL Server")
            logger.info(f"  Version: {version.splitlines()[0]}")
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def get_database_info(
        self,
        database: str,
        klines_suffix: str = "_klines",
    ) -> Optional[dict]:
        """
        Get counts for trades / orderbook and a summary of *_klines tables.
        """
        try:
            conn_string = self._get_connection_string(database)
            conn = pyodbc.connect(conn_string)
            cursor = conn.cursor()

            info = {"database": database, "tables": {}}

            # trades / orderbook
            for table in ("trades", "orderbook_snapshots", "ohlcv"):
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    info["tables"][table] = count
                except Exception:
                    info["tables"][table] = "N/A"

            # all *_klines tables
            cursor.execute(
                """
                SELECT name FROM sys.tables
                WHERE name LIKE ?
                """,
                f"%{klines_suffix}",
            )
            kline_tables = [row[0] for row in cursor.fetchall()]
            info["kline_tables"] = kline_tables

            conn.close()
            return info

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return None


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main():
    """Main initialization routine"""
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║      SQL Server Database Initialization                      ║")
    print("║      BTQ Market Data (read-only feed for BTQuant)           ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()

    # === CONFIG =======================================================
    SERVER = "localhost"
    USERNAME = "SA"
    PASSWORD = "q?}33YIToo:H%xue$Kr*"
    DRIVER = "{ODBC Driver 18 for SQL Server}"

    CANDLE_DATABASE = "BTQ_MarketData"

    # DB-Layout für OHLCV:
    MODE = "per_pair"           # "global" oder "per_pair"
    TABLE_PATTERN = "{symbol}_klines"
    # Beispiel: für MODE="per_pair" & TABLE_PATTERN="{symbol}_klines"
    # wird BTCUSDT -> "BTCUSDT_klines"

    # Liste von Paaren, für die du schon beim Init Tabellen erzeugen willst
    DEFAULT_EXCHANGE = "binance"
    PAIRS: Iterable[str] = [
        "btcusdt",
        "ethusdt",
        # weitere Paare hier ergänzen ...
    ]
    # ================================================================

    initializer = SQLServerInitializer(
        server=SERVER,
        username=USERNAME,
        password=PASSWORD,
        driver=DRIVER,
    )

    # 1) Verbindung testen
    print("[1/4] Verifying SQL Server connection...")
    if not initializer.verify_connection():
        print("❌ Connection failed. Please check your credentials and server.")
        return
    print()

    # 2) Datenbank anlegen
    print(f"[2/4] Creating database '{CANDLE_DATABASE}'...")
    if not initializer.create_database(CANDLE_DATABASE):
        print("❌ Failed to create database")
        return
    print()

    # 3) Basisschema initialisieren
    print(f"[3/4] Initializing schema in '{CANDLE_DATABASE}' (mode={MODE})...")
    if not initializer.initialize_market_data_schema(
        CANDLE_DATABASE,
        mode=MODE,
        table_pattern=TABLE_PATTERN,
        create_global_ohlcv=(MODE == "global"),
    ):
        print("❌ Failed to initialize schema")
        return
    print()

    # 4) Per-Pair OHLCV-Tabellen (nur wenn MODE = per_pair)
    if MODE == "per_pair":
        print("[4/4] Creating per-pair OHLCV tables...")
        for sym in PAIRS:
            initializer.create_ohlcv_table_for_pair(
                database=CANDLE_DATABASE,
                exchange=DEFAULT_EXCHANGE,
                symbol=sym,
                table_pattern=TABLE_PATTERN,
            )
        print()

    # Übersicht
    info = initializer.get_database_info(CANDLE_DATABASE)
    if info:
        print("Database info:")
        print(f"  Database: {info['database']}")
        print("  Tables:")
        for table, count in info["tables"].items():
            print(f"    - {table}: {count}")
        print("  OHLCV per-pair tables:")
        for tbl in info.get("kline_tables", []):
            print(f"    - {tbl}")
    print()

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  SETUP COMPLETE                                               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("# Market Data (BTQ_MarketData) connection string:")
    print("connection_string = (")
    print(f"    'DRIVER={DRIVER};'")
    print(f"    'SERVER={SERVER};'")
    print(f"    'DATABASE={CANDLE_DATABASE};'")
    print(f"    'UID={USERNAME};'")
    print(f"    'PWD={PASSWORD};'")
    print("    'TrustServerCertificate=yes;'")
    print(")")
    print()


if __name__ == "__main__":
    main()
