# BigBrainCentral Data Spine

BigBrainCentral is BTQuant's institutional-grade market data infrastructure, providing a complete data pipeline from exchange APIs to strategy execution with microsecond precision and enterprise reliability.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Ingestion](#data-ingestion)
- [Storage Layer](#storage-layer)
- [Data Access](#data-access)
- [Performance Characteristics](#performance-characteristics)
- [Operational Considerations](#operational-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

BigBrainCentral revolutionizes quantitative trading data management by implementing a true institutional data spine:

### Key Differentiators

- **Single Source of Truth**: All data flows through SQL Server with ACID guarantees
- **Microstructure Data**: Trades, orderbooks, and candles with microsecond precision
- **C++ Hot Path**: Ultra-low latency ingestion without Python GIL limitations
- **Research/Live Parity**: Backtests use identical data to live trading
- **Enterprise Scale**: Handles billions of records with optimized queries

### Architecture Overview

```
Exchange APIs → C++ Collectors → SQL Server → Python Adapters → Strategies/Analytics
```

## Architecture

### Component Breakdown

#### 1. C++ Ingestion Layer

**ExchangeConnectionManager**
- Manages WebSocket connections to multiple exchanges
- Correlation ID encoding: `exchange:symbol:market_type`
- Thread pool management for concurrent data streams

**MarketDataProcessor**
- Parses ccapi events (trades, orderbooks)
- Normalizes heterogeneous exchange payloads
- Enriches data with timestamps and metadata
- Decodes correlation IDs for proper classification

**CandleAggregator**
- Maintains per-symbol timeframe state
- Supports arbitrary resolutions (1s, 15s, 1m, 5m, 1h, etc.)
- Aligns timestamps to bucket boundaries
- Flushes completed candles for bulk insertion

**MSSQLBulkInserter**
- ODBC connection management with prepared statements
- Column-wise parameter binding for maximum throughput
- Three insertion paths:
  - `dbo.trades`: Raw trade data
  - `dbo.orderbook_snapshots`: Bid/ask depth
  - `{exchange}_{symbol}_klines`: Per-symbol OHLCV

#### 2. SQL Server Storage

**Schema Design**
- **Trades Table**:
  ```sql
  CREATE TABLE dbo.trades (
      id BIGINT IDENTITY PRIMARY KEY,
      exchange VARCHAR(50) NOT NULL,
      symbol VARCHAR(20) NOT NULL,
      market_type VARCHAR(20) NOT NULL,
      price DECIMAL(20, 8) NOT NULL,
      size DECIMAL(20, 8) NOT NULL,
      side TINYINT NOT NULL,  -- 0=BUY, 1=SELL
      aggressor_flag BIT,
      ts_exchange DATETIME2(6) NOT NULL,
      ts_local DATETIME2(6) NOT NULL,
      created_at DATETIME2(6) DEFAULT GETUTCDATE()
  );
  ```

- **Orderbook Snapshots Table**:
  ```sql
  CREATE TABLE dbo.orderbook_snapshots (
      id BIGINT IDENTITY PRIMARY KEY,
      exchange VARCHAR(50) NOT NULL,
      symbol VARCHAR(20) NOT NULL,
      market_type VARCHAR(20) NOT NULL,
      bids NVARCHAR(MAX),  -- JSON array of [price, size]
      asks NVARCHAR(MAX),  -- JSON array of [price, size]
      ts_exchange DATETIME2(6) NOT NULL,
      ts_local DATETIME2(6) NOT NULL,
      checksum BINARY(32),
      created_at DATETIME2(6) DEFAULT GETUTCDATE()
  );
  ```

- **OHLCV Tables** (per symbol):
  ```sql
  CREATE TABLE binance_btcusdt_klines (
      id BIGINT IDENTITY PRIMARY KEY,
      open_time DATETIME2(6) NOT NULL,
      open DECIMAL(20, 8) NOT NULL,
      high DECIMAL(20, 8) NOT NULL,
      low DECIMAL(20, 8) NOT NULL,
      close DECIMAL(20, 8) NOT NULL,
      volume DECIMAL(20, 8) NOT NULL,
      close_time DATETIME2(6) NOT NULL,
      quote_volume DECIMAL(20, 8),
      count BIGINT,
      taker_buy_volume DECIMAL(20, 8),
      taker_buy_quote_volume DECIMAL(20, 8),
      created_at DATETIME2(6) DEFAULT GETUTCDATE()
  );
  ```

#### 3. Python Access Layer

**ReadOnlyOHLCV**
- SELECT-only access for research
- Supports global and per-pair table modes
- Automatic query optimization

**DatabaseOHLCVData**
- Backtrader feed implementation
- Polls for new data in live mode
- Handles timestamp conversion

**MarketDataStorage**
- JackRabbitRelay integration
- Bulk data operations
- Health monitoring

## Data Ingestion

### Exchange Support

BigBrainCentral supports major cryptocurrency exchanges:

| Exchange | Status | Features |
|----------|--------|----------|
| Binance | ✅ Production | Spot, Futures, Options |
| Bitget | ✅ Production | Spot, Futures |
| MEXC | ✅ Production | Spot |
| OKX | ✅ Production | Spot, Futures, Options |
| Bybit | 🚧 Planned | Spot, Futures |
| KuCoin | 🚧 Planned | Spot, Futures |
| Gate.io | 🚧 Planned | Spot, Futures |

### Data Types

#### Trade Data
- **Price**: DECIMAL(20, 8) for precision
- **Size**: DECIMAL(20, 8) for full volume representation
- **Side**: BUY/SELL classification
- **Aggressor Flag**: Identifies market taker
- **Timestamps**: Microsecond precision (DATETIME2(6))

#### Orderbook Data
- **Bids/Asks**: JSON arrays of [price, size] pairs
- **Depth**: Configurable levels (default: top 20)
- **Checksums**: Data integrity validation
- **Update Frequency**: Real-time snapshots

#### Candle Data
- **Timeframes**: 1s to 1M+ intervals
- **OHLCV**: Standard price/volume data
- **Additional Fields**: Quote volume, trade count, taker metrics
- **Alignment**: Exchange-specific bucket boundaries

### Ingestion Pipeline

#### 1. Connection Establishment
```cpp
// Correlation ID encoding
std::string correlation_id = exchange + ":" + symbol + ":" + market_type;

// WebSocket subscription
session->subscribe({
    {"exchange", exchange},
    {"symbol", symbol},
    {"market_type", market_type}
});
```

#### 2. Data Processing
```cpp
void MarketDataProcessor::processTrade(const ccapi::Event& event) {
    // Parse ccapi event
    auto trade = parseTradeEvent(event);

    // Normalize data
    HotTrade normalized_trade = normalizeTrade(trade);

    // Enrich with metadata
    normalized_trade.ts_local = getCurrentTimestamp();

    // Queue for bulk insertion
    trade_buffer.push_back(normalized_trade);
}
```

#### 3. Bulk Insertion
```cpp
void MSSQLBulkInserter::flushTrades() {
    // Prepare statement
    SQLPrepare(stmt, "INSERT INTO dbo.trades (...) VALUES (?, ?, ...)", SQL_NTS);

    // Bind parameters column-wise
    SQLBindParameter(stmt, 1, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR,
                    0, 0, (SQLPOINTER)exchange_buffer.data(), 0, nullptr);

    // Execute bulk insert
    SQLExecute(stmt);

    // Commit transaction
    SQLTransact(env, conn, SQL_COMMIT);
}
```

## Storage Layer

### Database Configuration

#### Optimal Settings
```sql
-- Enable advanced features
EXEC sp_configure 'show advanced options', 1;
RECONFIGURE;

-- Memory optimization
EXEC sp_configure 'max server memory (MB)', 8192;  -- 8GB for 16GB system
RECONFIGURE;

-- Query optimization
EXEC sp_configure 'cost threshold for parallelism', 50;
EXEC sp_configure 'max degree of parallelism', 4;
RECONFIGURE;
```

#### Index Strategy
```sql
-- Trades table indexes
CREATE CLUSTERED INDEX IX_trades_ts_exchange
ON dbo.trades (exchange, symbol, ts_exchange);

CREATE NONCLUSTERED INDEX IX_trades_symbol_ts
ON dbo.trades (symbol, ts_exchange) INCLUDE (price, size, side);

-- OHLCV table indexes
CREATE CLUSTERED INDEX IX_klines_open_time
ON binance_btcusdt_klines (open_time);

CREATE NONCLUSTERED INDEX IX_klines_symbol_time
ON binance_btcusdt_klines (symbol, open_time) INCLUDE (open, high, low, close, volume);
```

#### Partitioning Strategy
```sql
-- Create partition function
CREATE PARTITION FUNCTION pf_trades_monthly (DATETIME2(6))
AS RANGE RIGHT FOR VALUES (
    '2024-01-01', '2024-02-01', '2024-03-01', -- Monthly partitions
    '2024-04-01', '2024-05-01', '2024-06-01'
);

-- Create partition scheme
CREATE PARTITION SCHEME ps_trades_monthly
AS PARTITION pf_trades_monthly
TO (fg_2024_01, fg_2024_02, fg_2024_03,
    fg_2024_04, fg_2024_05, fg_2024_06, fg_future);

-- Apply to table
ALTER TABLE dbo.trades
ADD CONSTRAINT PK_trades PRIMARY KEY NONCLUSTERED (id)
ON ps_trades_monthly (ts_exchange);
```

### Data Retention

#### Automated Cleanup
```sql
-- Create cleanup procedure
CREATE PROCEDURE sp_cleanup_old_data
    @retention_days INT = 365
AS
BEGIN
    DECLARE @cutoff_date DATETIME2(6) = DATEADD(DAY, -@retention_days, GETUTCDATE());

    -- Delete old trades
    DELETE FROM dbo.trades
    WHERE ts_exchange < @cutoff_date;

    -- Delete old orderbooks
    DELETE FROM dbo.orderbook_snapshots
    WHERE ts_exchange < @cutoff_date;

    -- Log cleanup
    INSERT INTO dbo.cleanup_log (table_name, records_deleted, cutoff_date)
    VALUES ('trades', @@ROWCOUNT, @cutoff_date);
END;
```

#### Archival Strategy
```sql
-- Archive to separate database
INSERT INTO archive_db.dbo.trades_archived
SELECT * FROM dbo.trades
WHERE ts_exchange < DATEADD(MONTH, -12, GETUTCDATE());

-- Compress archived data
ALTER INDEX ALL ON archive_db.dbo.trades_archived
REBUILD WITH (DATA_COMPRESSION = PAGE);
```

## Data Access

### Python Integration

#### Backtrader Feeds
```python
from backtrader.feeds import DatabaseOHLCVData

# Create feed
data = DatabaseOHLCVData(
    exchange='binance',
    symbol='BTCUSDT',
    timeframe='1h',
    fromdate=datetime(2024, 1, 1),
    todate=datetime(2024, 12, 31)
)

# Add to cerebro
cerebro.adddata(data)
```

#### Direct SQL Access
```python
import pyodbc

# Connection
conn = pyodbc.connect(connection_string)

# Query trades
cursor = conn.cursor()
cursor.execute("""
    SELECT ts_exchange, price, size, side
    FROM dbo.trades
    WHERE exchange = ? AND symbol = ?
    AND ts_exchange BETWEEN ? AND ?
    ORDER BY ts_exchange
""", ('binance', 'BTCUSDT', start_date, end_date))

trades = cursor.fetchall()
```

#### Analytics Integration
```python
import pandas as pd

# Load data for analysis
query = """
SELECT
    DATEPART(HOUR, ts_exchange) as hour,
    AVG(price) as avg_price,
    SUM(size) as total_volume,
    COUNT(*) as trade_count
FROM dbo.trades
WHERE exchange = 'binance' AND symbol = 'BTCUSDT'
AND ts_exchange >= DATEADD(DAY, -30, GETUTCDATE())
GROUP BY DATEPART(HOUR, ts_exchange)
ORDER BY hour
"""

df = pd.read_sql(query, conn)
```

### Research Applications

#### Microstructure Analysis
```python
# Order flow analysis
query = """
SELECT
    ts_exchange,
    price,
    size,
    side,
    SUM(CASE WHEN side = 0 THEN size ELSE -size END) OVER
        (ORDER BY ts_exchange ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as order_imbalance
FROM dbo.trades
WHERE exchange = 'binance' AND symbol = 'BTCUSDT'
ORDER BY ts_exchange
"""

order_flow = pd.read_sql(query, conn)
```

#### Liquidity Analysis
```python
# Orderbook depth analysis
query = """
SELECT
    ts_exchange,
    JSON_VALUE(bids, '$[0][0]') as best_bid,
    JSON_VALUE(asks, '$[0][0]') as best_ask,
    CAST(JSON_VALUE(bids, '$[0][1]') AS FLOAT) as bid_size,
    CAST(JSON_VALUE(asks, '$[0][1]') AS FLOAT) as ask_size
FROM dbo.orderbook_snapshots
WHERE exchange = 'binance' AND symbol = 'BTCUSDT'
ORDER BY ts_exchange DESC
"""

liquidity = pd.read_sql(query, conn)
```

## Performance Characteristics

### Ingestion Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Trades/Second** | 270,000 | Peak observed |
| **Orderbooks/Second** | 800 | Limited by exchange APIs |
| **Latency** | <130ms | End-to-end from exchange to SQL |
| **CPU Usage** | <5% | C++ optimized |
| **Memory Usage** | 100MB | Buffer management |

### Query Performance

| Query Type | Latency | Throughput |
|------------|---------|------------|
| **Single Symbol Trades** | 50ms | 20K queries/sec |
| **Time Range Scan** | 200ms | 5K queries/sec |
| **OHLCV Aggregation** | 100ms | 10K queries/sec |
| **Orderbook Lookup** | 25ms | 40K queries/sec |

### Storage Efficiency

| Data Type | Size/Record | Daily Volume | Monthly Storage |
|-----------|-------------|--------------|-----------------|
| **Trades** | 80 bytes | 10M | 2.4GB |
| **Orderbooks** | 2KB | 100K | 20GB |
| **1m Candles** | 120 bytes | 1440 | 170KB |
| **1h Candles** | 120 bytes | 24 | 3KB |

## Operational Considerations

### Monitoring

#### Key Metrics
```sql
-- Ingestion health check
SELECT
    exchange,
    symbol,
    COUNT(*) as trades_last_hour,
    MAX(ts_exchange) as latest_trade,
    DATEDIFF(MINUTE, MAX(ts_exchange), GETUTCDATE()) as minutes_behind
FROM dbo.trades
WHERE ts_exchange >= DATEADD(HOUR, -1, GETUTCDATE())
GROUP BY exchange, symbol;
```

#### Alert Conditions
- Ingestion lag > 5 minutes
- Error rate > 1%
- Buffer utilization > 90%
- Query latency > 1 second

### Backup and Recovery

#### Backup Strategy
```sql
-- Full backup weekly
BACKUP DATABASE BigBrainCentral
TO DISK = 'D:\backups\bb_weekly.bak'
WITH COMPRESSION, CHECKSUM;

-- Differential daily
BACKUP DATABASE BigBrainCentral
TO DISK = 'D:\backups\bb_daily.diff'
WITH DIFFERENTIAL, COMPRESSION;

-- Transaction log hourly
BACKUP LOG BigBrainCentral
TO DISK = 'D:\backups\bb_log.trn'
WITH COMPRESSION;
```

#### Point-in-Time Recovery
```sql
-- Restore sequence
RESTORE DATABASE BigBrainCentral
FROM DISK = 'D:\backups\bb_weekly.bak'
WITH NORECOVERY;

RESTORE DATABASE BigBrainCentral
FROM DISK = 'D:\backups\bb_daily.diff'
WITH NORECOVERY;

RESTORE LOG BigBrainCentral
FROM DISK = 'D:\backups\bb_log.trn'
WITH RECOVERY;
```

### Scaling

#### Vertical Scaling
- Increase SQL Server memory allocation
- Add more CPU cores
- Use faster storage (NVMe SSDs)
- Optimize tempdb configuration

#### Horizontal Scaling
- Read replicas for analytics
- Sharded databases by exchange
- Distributed ingestion workers
- Load-balanced query routing

## Troubleshooting

### Common Issues

#### 1. Ingestion Lag
**Symptoms**: Data appears delayed in queries
**Causes**:
- Network connectivity issues
- Exchange API rate limits
- SQL Server performance problems

**Solutions**:
```sql
-- Check ingestion status
SELECT
    exchange,
    symbol,
    MAX(ts_exchange) as latest_data,
    DATEDIFF(SECOND, MAX(ts_exchange), GETUTCDATE()) as lag_seconds
FROM dbo.trades
GROUP BY exchange, symbol;
```

#### 2. Connection Failures
**Symptoms**: ODBC connection errors
**Causes**:
- SQL Server service down
- Network firewall issues
- Authentication problems

**Solutions**:
```bash
# Test ODBC connection
isql -v "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=master;UID=sa;PWD=YourPassword;TrustServerCertificate=yes"
```

#### 3. Performance Degradation
**Symptoms**: Queries slow down over time
**Causes**:
- Index fragmentation
- Statistics out of date
- Tempdb full

**Solutions**:
```sql
-- Rebuild indexes
ALTER INDEX ALL ON dbo.trades REBUILD;

-- Update statistics
UPDATE STATISTICS dbo.trades;

-- Check tempdb usage
SELECT
    name,
    size_mb = size * 8.0 / 1024,
    used_mb = (size - available) * 8.0 / 1024
FROM (
    SELECT
        name,
        size = SUM(size),
        available = SUM(available)
    FROM tempdb.sys.database_files
    GROUP BY name
) t;
```

#### 4. Data Quality Issues
**Symptoms**: Missing or incorrect data
**Causes**:
- Exchange API changes
- Parsing errors
- Data corruption

**Solutions**:
```sql
-- Data validation queries
SELECT
    exchange,
    symbol,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN price <= 0 THEN 1 END) as invalid_prices,
    COUNT(CASE WHEN size <= 0 THEN 1 END) as invalid_sizes,
    MIN(ts_exchange) as earliest_trade,
    MAX(ts_exchange) as latest_trade
FROM dbo.trades
GROUP BY exchange, symbol;
```

BigBrainCentral represents the state-of-the-art in quantitative trading data infrastructure, providing institutional-grade data management with the transparency and performance required for serious algorithmic trading.