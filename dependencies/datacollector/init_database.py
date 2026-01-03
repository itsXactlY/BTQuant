#!/usr/bin/env python3
"""
BTQ_MarketData - MICROSECOND LATENCY EDITION
============================================
Zero staging. Direct writes. Sub-millisecond inserts.
"""

import pyodbc
import sys

SERVER = "localhost"
DB = "BTQ_MarketData"
USER = "SA"
PASS = "q?}33YIToo:H%xue$Kr*"

def get_conn(database="master", autocommit=True):
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={SERVER};DATABASE={database};UID={USER};PWD={PASS};"
        f"TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str, autocommit=autocommit)

def setup():
    print("🔧 BTQ_MarketData MICROSECOND LATENCY Init\n")
    
    # 1. DROP
    print("1/4 Dropping old database...", end=" ", flush=True)
    try:
        conn = get_conn(autocommit=True)
        cursor = conn.cursor()
        cursor.execute(f"IF DB_ID('{DB}') IS NOT NULL ALTER DATABASE [{DB}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE")
        cursor.execute(f"IF DB_ID('{DB}') IS NOT NULL DROP DATABASE [{DB}]")
        conn.close()
        print("✅")
    except Exception as e:
        print(f"⚠️  {e}")
    
    # 2. CREATE
    print("2/4 Creating database...", end=" ", flush=True)
    try:
        conn = get_conn(autocommit=True)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE DATABASE [{DB}]
            ON PRIMARY (
                NAME = '{DB}_data',
                FILENAME = '/var/opt/mssql/data/{DB}.mdf',
                SIZE = 50GB,
                FILEGROWTH = 10GB,
                MAXSIZE = 500GB
            )
            LOG ON (
                NAME = '{DB}_log',
                FILENAME = '/var/opt/mssql/data/{DB}_log.ldf',
                SIZE = 10GB,
                FILEGROWTH = 2GB,
                MAXSIZE = 100GB
            )
        """)
        conn.close()
        print("✅")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # 3. CONFIGURE
    print("3/4 Configuring performance...", end=" ", flush=True)
    try:
        conn = get_conn(autocommit=True)
        cursor = conn.cursor()
        
        cursor.execute(f"ALTER DATABASE [{DB}] SET DELAYED_DURABILITY = FORCED")
        cursor.execute(f"ALTER DATABASE [{DB}] SET RECOVERY SIMPLE")
        cursor.execute(f"ALTER DATABASE [{DB}] SET AUTO_CREATE_STATISTICS ON")
        cursor.execute(f"ALTER DATABASE [{DB}] SET AUTO_UPDATE_STATISTICS_ASYNC ON")
        cursor.execute(f"ALTER DATABASE [{DB}] SET TARGET_RECOVERY_TIME = 60 SECONDS")
        cursor.execute(f"ALTER DATABASE [{DB}] SET AUTO_SHRINK OFF")
        conn.close()
        
        # MAXDOP (must connect to DB)
        conn = get_conn(database=DB, autocommit=True)
        cursor = conn.cursor()
        cursor.execute("ALTER DATABASE SCOPED CONFIGURATION SET MAXDOP = 2")
        conn.close()
        
        print("✅")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # 4. CREATE TABLES
    print("4/4 Creating tables...", end=" ", flush=True)
    try:
        conn = get_conn(database=DB, autocommit=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE dbo.trades (
                id BIGINT IDENTITY(1,1) PRIMARY KEY NONCLUSTERED,
                [timestamp] DATETIME2(6) NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                trade_id VARCHAR(200) NULL,
                price DECIMAL(20,8) NOT NULL,
                quantity DECIMAL(30,8) NOT NULL,
                side VARCHAR(10) NOT NULL,
                is_buyer_maker BIT NULL,
                created_at DATETIME2(6) DEFAULT SYSUTCDATETIME()
            )
        """)
        
        cursor.execute("""
            CREATE CLUSTERED INDEX CIX_trades_time 
            ON dbo.trades([timestamp] DESC)
            WITH (DATA_COMPRESSION = PAGE, FILLFACTOR = 90)
        """)
        
        cursor.execute("""
            CREATE NONCLUSTERED INDEX IX_trades_lookup
            ON dbo.trades(exchange, symbol, market_type, [timestamp] DESC)
            INCLUDE (price, quantity, side)
            WITH (DATA_COMPRESSION = PAGE, FILLFACTOR = 90)
        """)
        
        cursor.execute("""
            CREATE TABLE dbo.orderbook_snapshots (
                id BIGINT IDENTITY(1,1) PRIMARY KEY NONCLUSTERED,
                [timestamp] DATETIME2(6) NOT NULL,
                exchange VARCHAR(50) NOT NULL,
                symbol VARCHAR(50) NOT NULL,
                market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
                bids VARCHAR(MAX) NOT NULL,
                asks VARCHAR(MAX) NOT NULL,
                checksum VARCHAR(64) NULL,
                created_at DATETIME2(6) DEFAULT SYSUTCDATETIME()
            )
        """)
        
        cursor.execute("""
            CREATE CLUSTERED INDEX CIX_orderbook_time
            ON dbo.orderbook_snapshots([timestamp] DESC)
            WITH (DATA_COMPRESSION = PAGE, FILLFACTOR = 90)
        """)
        
        cursor.execute("""
            CREATE NONCLUSTERED INDEX IX_orderbook_lookup
            ON dbo.orderbook_snapshots(exchange, symbol, market_type, [timestamp] DESC)
            WITH (DATA_COMPRESSION = PAGE, FILLFACTOR = 90)
        """)
        
        cursor.execute("""
            CREATE TABLE dbo.orders (
                [timestamp] DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME(),
                id VARCHAR(100) NOT NULL,
                exchange VARCHAR(20) NOT NULL,
                account VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side CHAR(4) NOT NULL,
                type VARCHAR(10) NOT NULL,
                price DECIMAL(18,8) NULL,
                amount DECIMAL(18,8) NOT NULL,
                filled DECIMAL(18,8) NOT NULL DEFAULT 0,
                status VARCHAR(20) NOT NULL,
                client_order_id VARCHAR(100) NULL
            )
        """)
        
        cursor.execute("""
            CREATE CLUSTERED INDEX CIX_orders 
            ON dbo.orders([timestamp] DESC) 
            WITH (DATA_COMPRESSION = PAGE)
        """)
        
        cursor.execute("""
            CREATE UNIQUE NONCLUSTERED INDEX IX_orders_id 
            ON dbo.orders(id) 
            WITH (DATA_COMPRESSION = PAGE)
        """)
        
        cursor.execute("""
            CREATE PROCEDURE dbo.sp_purge AS
            BEGIN
                SET NOCOUNT ON;
                DECLARE @cutoff DATETIME2 = DATEADD(HOUR, -48, SYSUTCDATETIME());
                DECLARE @deleted INT;
                
                WHILE 1=1
                BEGIN
                    DELETE TOP (100000) FROM dbo.trades WITH (ROWLOCK)
                    WHERE [timestamp] < @cutoff;
                    SET @deleted = @@ROWCOUNT;
                    IF @deleted = 0 BREAK;
                    WAITFOR DELAY '00:00:00.200';
                END;
                
                WHILE 1=1
                BEGIN
                    DELETE TOP (100000) FROM dbo.orderbook_snapshots WITH (ROWLOCK)
                    WHERE [timestamp] < @cutoff;
                    SET @deleted = @@ROWCOUNT;
                    IF @deleted = 0 BREAK;
                    WAITFOR DELAY '00:00:00.200';
                END;
                
                ALTER INDEX CIX_trades_time ON dbo.trades REORGANIZE;
                ALTER INDEX CIX_orderbook_time ON dbo.orderbook_snapshots REORGANIZE;
            END
        """)
        
        conn.commit()
        conn.close()
        print("✅")
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"""
✅ DONE

Your C++ bulk inserter is ready to rock at 30k+ inserts/sec.
""")

if __name__ == "__main__":
    setup()