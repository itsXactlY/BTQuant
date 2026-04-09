# BigBrainCentral Technical Documentation

## Overview

BigBrainCentral is BTQuant's SQL Server-based market data storage system. It uses a custom C++ ODBC driver (`fast_mssql`) for high-performance database operations.

**Key Features:**
- Custom C++ ODBC driver (no Python DB layer overhead)
- Connection pooling at C++ level
- Bulk insert operations
- Thread-safe operations
- Per-market OHLCV table sharding

## Components

### 1. MSSQLConfig (`bigbraincentral/storage_mssql.py`)

Configuration dataclass for SQL Server connections:

```python
@dataclass
class MSSQLConfig:
    server: str = "localhost"
    database: str = "BTQ_MarketData"
    username: str = "SA"
    password: str = ""
    driver: str = "{ODBC Driver 18 for SQL Server}"
    trust_server_certificate: bool = True
    
    def get_connection_string(self) -> str:
        """Build ODBC connection string"""
```

**Example Connection String:**
```
DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=BTQ_MarketData;UID=SA;PWD=YourPassword;TrustServerCertificate=yes;
```

### 2. MarketDataStorage Class

High-performance synchronous SQL Server storage.

```python
class MarketDataStorage:
    def __init__(self, config: MSSQLConfig, logger: Optional[logging.Logger] = None):
    
    # Connection management
    def connect(self) -> None:
    def disconnect(self) -> None:
    
    # Store operations
    def store_ohlcv(self, data: Dict[str, Any]) -> bool:
    def store_trade(self, data: Dict[str, Any]) -> bool:
    def store_orderbook(self, data: Dict[str, Any]) -> bool:
    def bulk_store_ohlcv(self, data_list: List[Dict[str, Any]]) -> int:
    
    # Read operations
    def get_ohlcv(self, exchange: str, symbol: str, timeframe: str,
                  start: datetime, end: Optional[datetime] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
    def get_trades(self, exchange: str, symbol: str,
                   start: datetime, end: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
    def get_stats(self) -> Dict[str, Any]:
```

### 3. Database Schema

#### Trades Table

```sql
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

CREATE INDEX idx_trades_lookup 
ON trades(exchange, symbol, market_type, timestamp DESC);
```

#### Orderbook Snapshots Table

```sql
CREATE TABLE orderbook_snapshots (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    timestamp DATETIME2 NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    market_type VARCHAR(20) NOT NULL DEFAULT 'spot',
    bids NVARCHAR(MAX) NOT NULL,    -- JSON array
    asks NVARCHAR(MAX) NOT NULL,    -- JSON array
    checksum VARCHAR(64),
    created_at DATETIME2 DEFAULT GETDATE()
);

CREATE INDEX idx_orderbook_lookup 
ON orderbook_snapshots(exchange, symbol, market_type, timestamp DESC);
```

#### OHLCV Tables (Per-Market Sharding)

Each exchange+symbol combination gets its own table:

```sql
-- Table name format: {exchange}_{symbol}_klines
-- Example: binance_btcusdt_klines

CREATE TABLE [binance_btcusdt_klines] (
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
    CONSTRAINT UQ_btcusdt_klines_ohlcv 
        UNIQUE(timestamp, exchange, symbol, market_type, timeframe)
);

CREATE INDEX idx_binance_btcusdt_klines_lookup 
ON [binance_btcusdt_klines](exchange, symbol, market_type, timeframe, timestamp DESC);
```

### 4. Data Formats

#### OHLCV Data Input

```python
ohlcv_data = {
    "exchange": "binance",          # str - Exchange name
    "symbol": "BTCUSDT",            # str - Symbol
    "timestamp": 1704067200000,     # int - Milliseconds since epoch
    "timeframe": "1m",              # str - Timeframe (1m, 5m, 1h, etc.)
    "open": 42000.50,               # float - Open price
    "high": 42100.75,               # float - High price
    "low": 41950.25,                # float - Low price
    "close": 42050.00,              # float - Close price
    "volume": 123.456,              # float - Volume
    "market_type": "spot"           # str (optional) - Market type
}

storage.store_ohlcv(ohlcv_data)
```

#### Trade Data Input

```python
trade_data = {
    "timestamp": 1704067200000,     # int - Milliseconds since epoch
    "exchange": "binance",          # str - Exchange name
    "symbol": "BTCUSDT",            # str - Symbol
    "trade_id": "123456789",        # str - Trade ID
    "price": 42000.50,              # float - Trade price
    "quantity": 0.123,              # float - Trade quantity
    "side": "buy",                  # str - "buy" or "sell"
    "is_buyer_maker": True,         # bool - Is buyer the maker
    "market_type": "spot"           # str (optional) - Market type
}

storage.store_trade(trade_data)
```

#### Orderbook Data Input

```python
orderbook_data = {
    "timestamp": 1704067200000,     # int - Milliseconds since epoch
    "exchange": "binance",          # str - Exchange name
    "symbol": "BTCUSDT",            # str - Symbol
    "bids": [                       # list - Bid levels [(price, size), ...]
        [42000.00, 1.5],
        [41999.00, 2.3],
    ],
    "asks": [                       # list - Ask levels [(price, size), ...]
        [42001.00, 0.8],
        [42002.00, 1.2],
    ],
    "checksum": "abc123",           # str (optional) - Orderbook checksum
    "market_type": "spot"           # str (optional) - Market type
}

storage.store_orderbook(orderbook_data)
```

### 5. ReadOnlyOHLCV Class (`feeds/db_ohlcv_mssql.py`)

Read-only OHLCV data access for backtesting and live feeds:

```python
@dataclass
class MSSQLFeedConfig:
    server: str = "localhost"
    database: str = "BTQ_MarketData"
    username: str = "SA"
    password: str = ""
    driver: str = "{ODBC Driver 18 for SQL Server}"
    trust_server_certificate: bool = True
    
    def connection_string(self) -> str:

class ReadOnlyOHLCV:
    def __init__(self,
                 config: MSSQLFeedConfig,
                 mode: str = "global",          # "global" or "per_pair"
                 global_table: str = "ohlcv",
                 schema: str = "dbo",
                 table_pattern: str = "{symbol}_klines"):
    
    def get_ohlcv(self,
                  exchange: str,
                  symbol: str,
                  timeframe: str,
                  start: datetime,
                  end: Optional[datetime] = None,
                  limit: Optional[int] = None,
                  strict_gt: bool = False) -> List[dict]:
```

**Modes:**
- `"global"`: Single table with exchange/symbol columns
- `"per_pair"`: Separate table per exchange+symbol

### 6. ReadOnlyTradesAgg Class

Extends ReadOnlyOHLCV with tick-level trade access:

```python
class ReadOnlyTradesAgg(ReadOnlyOHLCV):
    def get_ticks_by_id(self, 
                        exchange: str, 
                        symbol: str, 
                        last_id: int, 
                        limit: int = 1000) -> list[dict]:
```

**Output Format:**
```python
[
    {
        "id": 12345,                    # int - Trade ID (monotonic)
        "timestamp": datetime,          # datetime - Trade timestamp
        "open": 42000.50,              # float - Trade price (same as close)
        "high": 42000.50,              # float - Trade price
        "low": 42000.50,               # float - Trade price
        "close": 42000.50,             # float - Trade price
        "volume": 0.123                # float - Trade quantity
    },
    ...
]
```

**Note:** Tick data uses OHLCV format where O=H=L=C=price and V=quantity for compatibility with Backtrader.

### 7. DatabaseOHLCVData Feed

Backtrader-compatible data feed reading from SQL Server:

```python
class DatabaseOHLCVData(DataBase):
    params = (
        ("db_config", None),            # MSSQLFeedConfig
        ("exchange", None),             # "binance", "okx", etc.
        ("symbol", None),               # "BTC-USDT", etc.
        ("timeframe", TimeFrame.Seconds),
        ("compression", 1),
        ("fromdate", None),             # datetime
        ("todate", None),               # datetime (optional)
        ("live", True),                 # Enable live polling
        ("poll_interval", 0.10),        # Base poll interval
        ("mode", "global"),             # "global" or "per_pair"
        ("global_table", "ohlcv"),
        ("schema", "dbo"),
        ("table_pattern", "{symbol}_klines"),
        ("source", "auto"),             # "auto", "klines", "trades"
        ("ticks", True),                # Raw per-trade mode
        ("tick_batch_limit", 1000),     # Batch size per poll
        ("min_poll", 0.05),             # Fastest poll interval
        ("max_poll", 0.50),             # Slowest poll interval
        ("debug", False),
    )
```

**Convenience Subclasses:**
```python
class BinanceDBData(DatabaseOHLCVData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("exchange", "binance")

class OkxDBData(DatabaseOHLCVData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("exchange", "okx")
```

## Usage Examples

### Basic Storage Setup

```python
from backtrader.bigbraincentral.storage_mssql import MarketDataStorage, MSSQLConfig

config = MSSQLConfig(
    server="localhost",
    database="BTQ_MarketData",
    username="SA",
    password="YourStrong!Passw0rd"
)

storage = MarketDataStorage(config)
storage.connect()

# Store OHLCV
storage.store_ohlcv({
    "exchange": "binance",
    "symbol": "BTCUSDT",
    "timestamp": 1704067200000,
    "timeframe": "1m",
    "open": 42000.0,
    "high": 42100.0,
    "low": 41900.0,
    "close": 42050.0,
    "volume": 100.0
})

# Query OHLCV
data = storage.get_ohlcv(
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1m",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 2)
)

storage.disconnect()
```

### Live Feed from Database

```python
import backtrader as bt
from backtrader.feeds.db_ohlcv_mssql import BinanceDBData, MSSQLFeedConfig
from datetime import datetime

config = MSSQLFeedConfig(
    server="localhost",
    database="BTQ_MarketData",
    username="SA",
    password="YourPassword"
)

cerebro = bt.Cerebro()

data = BinanceDBData(
    db_config=config,
    symbol="BTC-USDT",
    timeframe=bt.TimeFrame.Minutes,
    compression=1,
    fromdate=datetime(2024, 1, 1),
    live=True,
    ticks=True
)

cerebro.adddata(data)
cerebro.addstrategy(MyStrategy)
cerebro.run()
```

### Bulk OHLCV Storage

```python
ohlcv_list = [
    {"exchange": "binance", "symbol": "BTCUSDT", "timestamp": 1704067200000,
     "timeframe": "1m", "open": 42000.0, "high": 42100.0, 
     "low": 41900.0, "close": 42050.0, "volume": 100.0},
    # ... more rows
]

count = storage.bulk_store_ohlcv(ohlcv_list)
print(f"Stored {count} rows")
```
