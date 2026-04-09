# HotSpine Technical Documentation

## Overview

HotSpine is a high-performance shared memory market data system that enables ultra-low latency trade and orderbook reading for live trading. It provides a bridge between C++ market data collectors and Python trading strategies.

**Architecture:**
```
C++ Collector --> HotSpine Writer --> Shared Memory --> Python Reader --> Strategy
```

## Core Components

### 1. HotSpineReader (`hotspine/reader.py`)

The main Python interface for reading market data from shared memory.

**Key Features:**
- Low-latency trade and orderbook reading
- Symbol ID to exchange/symbol mapping
- Buffer monitoring and health checking
- Real-time statistics and performance metrics

### 2. Data Structures

#### HotTrade (in `reader.py`)

C-compatible structure matching C++ HotSpine layout:

```python
class HotTrade(ctypes.Structure):
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),   # Exchange timestamp in microseconds (UTC)
        ("ts_local", ctypes.c_uint64),      # Local receive timestamp in microseconds
        ("price", ctypes.c_double),         # Trade price
        ("size", ctypes.c_double),          # Trade size (quantity)
        ("symbol_id", ctypes.c_uint32),     # Symbol ID (hash or mapping)
        ("side", ctypes.c_uint8),           # 0=BUY, 1=SELL
        ("padding", ctypes.c_uint8 * 3),    # Alignment padding
    ]
```

**Properties:**
- `side_enum -> Side`: Returns `Side.BUY` or `Side.SELL`
- `to_dict() -> Dict`: Converts to dictionary
- `to_trade_data() -> TradeData`: Converts to high-level TradeData

#### HotTrade (in `sql_integration.py`)

SQL integration variant with market type field:

```python
class HotTrade(ctypes.Structure):
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),   # Exchange timestamp in microseconds
        ("ts_local", ctypes.c_uint64),      # Local receive timestamp in microseconds
        ("price", ctypes.c_double),         # Trade price
        ("size", ctypes.c_double),          # Trade size
        ("symbol_id", ctypes.c_uint32),     # Symbol ID (hash or mapping)
        ("side", ctypes.c_uint8),           # 0=buy, 1=sell
        ("market_type", ctypes.c_uint8),    # 0=spot, 1=futures, 2=other
    ]
```

**Note:** The `sql_integration.py` version includes the `market_type` field which is NOT present in the `reader.py` version.

#### HotOrderbookLevel

Single orderbook level structure:

```python
class HotOrderbookLevel(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),  # Level price
        ("size", ctypes.c_double),   # Level size (quantity)
    ]
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (price, size) tuple"""
```

#### HotOrderbookSnapshot

Full orderbook snapshot structure:

```python
class HotOrderbookSnapshot(ctypes.Structure):
    _fields_ = [
        ("ts_exchange", ctypes.c_uint64),           # Exchange timestamp (microseconds)
        ("ts_local", ctypes.c_uint64),              # Local receive timestamp
        ("symbol_id", ctypes.c_uint32),             # Symbol ID
        ("bids_count", ctypes.c_uint8),             # Number of bid levels
        ("asks_count", ctypes.c_uint8),             # Number of ask levels
        ("padding", ctypes.c_uint8 * 2),            # Alignment
        ("bids", HotOrderbookLevel * 20),           # Max 20 bid levels
        ("asks", HotOrderbookLevel * 20),           # Max 20 ask levels
    ]
    
    def get_bids(self) -> List[Tuple[float, float]]:
    def get_asks(self) -> List[Tuple[float, float]]:
    def to_dict(self) -> Dict[str, Any]:
```

#### SharedMemoryHeader

Header structure for the shared memory segment:

```python
class SharedMemoryHeader(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint64),                  # Protocol version
        ("capacity", ctypes.c_uint64),                 # Buffer capacity (trades)
        ("write_index", ctypes.c_uint64),              # Writer position
        ("read_index", ctypes.c_uint64),               # Reader position
        ("lost_count", ctypes.c_uint64),               # Trades lost to overflow
        ("orderbook_write_index", ctypes.c_uint64),    # Orderbook writer position
        ("orderbook_read_index", ctypes.c_uint64),     # Orderbook reader position
        ("orderbook_lost_count", ctypes.c_uint64),     # Orderbooks lost to overflow
        ("orderbook_capacity", ctypes.c_uint64),       # Orderbook buffer capacity
        ("padding", ctypes.c_uint8 * 8),               # Alignment
    ]
```

### 3. High-Level Data Classes

#### TradeData

High-level trade representation with resolved symbol info:

```python
@dataclass
class TradeData:
    timestamp_us: int           # Exchange timestamp (microseconds)
    local_timestamp_us: int     # Local timestamp (microseconds)
    price: float                # Trade price
    size: float                 # Trade size
    symbol_id: int              # Symbol ID
    side: Side                  # BUY or SELL
    exchange: str = ""          # Resolved exchange name
    symbol: str = ""            # Resolved symbol name
    
    @property
    def timestamp(self) -> datetime:
    def to_dict(self) -> Dict[str, Any]:
```

#### OrderbookData

High-level orderbook representation:

```python
@dataclass
class OrderbookData:
    timestamp_us: int                       # Exchange timestamp
    local_timestamp_us: int                 # Local timestamp
    symbol_id: int                          # Symbol ID
    bids: List[Tuple[float, float]]         # [(price, size), ...]
    asks: List[Tuple[float, float]]         # [(price, size), ...]
    exchange: str = ""                      # Resolved exchange
    symbol: str = ""                        # Resolved symbol
    
    @property
    def timestamp(self) -> datetime:
    def get_mid_price(self) -> Optional[float]:
    def get_spread(self) -> Optional[float]:
    def to_dict(self) -> Dict[str, Any]:
```

### 4. SymbolMapper

Maps symbol IDs to exchange/symbol pairs:

```python
class SymbolMapper:
    DEFAULT_MAPPINGS = {
        # Binance spot
        1: ("binance", "BTC-USDT"),
        2: ("binance", "ETH-USDT"),
        3: ("binance", "SOL-USDT"),
        # Bybit
        101: ("bybit", "BTC-USDT"),
        102: ("bybit", "ETH-USDT"),
        # OKX
        201: ("okx", "BTC-USDT"),
        202: ("okx", "ETH-USDT"),
    }
    
    def __init__(self, mapping_file: Optional[str] = None):
    def load_from_file(self, filepath: str): ...
    def save_to_file(self, filepath: str): ...
    def get_symbol_info(self, symbol_id: int) -> Tuple[str, str]:
    def get_symbol_id(self, exchange: str, symbol: str) -> int:
    def register_symbol(self, exchange: str, symbol: str, 
                        symbol_id: Optional[int] = None) -> int:
```

**Auto-loading:** The mapper automatically loads from `/dev/shm/btquant_symbols.json` if it exists (written by C++ HotSpineWriter).

### 5. HotSpineConfig (`hotspine/config.py`)

Centralized configuration with environment variable support:

```python
@dataclass
class HotSpineConfig:
    # Shared memory configuration
    shm_name: str = "/btquant_hotspine"
    shm_capacity: int = 1000000
    
    # Reader configuration
    poll_interval: float = 0.0001      # 100 microseconds
    batch_mode: bool = False
    max_batch_size: int = 1000
    
    # SQL integration configuration
    enable_sql_storage: bool = True
    sql_batch_size: int = 100
    sql_queue_size: int = 10000
    
    # Performance monitoring
    enable_monitoring: bool = True
    metrics_interval: float = 1.0
    
    # Error handling
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    
    # Symbol management
    symbol_mapping: Optional[Dict[int, Dict[str, str]]] = None
    
    # Market type filtering
    market_type_filter: str = "all"    # "spot", "futures", or "all"
    
    # Symbol whitelisting
    symbol_whitelist: Optional[List[str]] = None
```

**Environment Variables:**

| Variable | Description |
|----------|-------------|
| `HOTSPINE_SHM_NAME` | Shared memory segment name |
| `HOTSPINE_SHM_CAPACITY` | Buffer capacity |
| `HOTSPINE_POLL_INTERVAL` | Polling interval (seconds) |
| `HOTSPINE_BATCH_MODE` | Enable batch mode (true/1/yes) |
| `HOTSPINE_MAX_BATCH_SIZE` | Maximum batch size |
| `HOTSPINE_ENABLE_SQL_STORAGE` | Enable SQL storage |
| `HOTSPINE_SQL_BATCH_SIZE` | SQL batch size |
| `HOTSPINE_SQL_QUEUE_SIZE` | SQL queue size |
| `HOTSPINE_ENABLE_MONITORING` | Enable health monitoring |
| `HOTSPINE_METRICS_INTERVAL` | Metrics interval (seconds) |
| `HOTSPINE_MAX_RECONNECT_ATTEMPTS` | Max reconnect attempts |
| `HOTSPINE_RECONNECT_DELAY` | Reconnect delay (seconds) |
| `HOTSPINE_SYMBOL_MAPPING` | JSON symbol mapping |
| `HOTSPINE_MARKET_TYPE_FILTER` | Market type filter |
| `HOTSPINE_SYMBOL_WHITELIST` | JSON symbol whitelist |

### 6. HotSpineData Feed (`feeds/hotspine_feed.py`)

Backtrader-compatible data feed for HotSpine:

```python
class HotSpineData(DataBase):
    params = (
        ('shm_name', '/btquant_hotspine'),  # Shared memory segment
        ('symbol_id', None),                # Symbol ID to filter
        ('symbol', ''),                     # Symbol string
        ('timeframe', bt.TimeFrame.Ticks),  # Timeframe (ticks for live)
        ('compression', 1),                 # Compression factor
        ('batch_mode', False),              # Batch reading mode
        ('poll_interval', 0.0001),          # Polling interval
    )
    
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def islive(self) -> bool: True
    def haslivedata(self) -> bool: True
    def get_feed_metrics(self) -> Dict[str, Any]:
    def get_health_status(self) -> Dict[str, Any]:
```

**Factory Functions:**
```python
def create_hotspine_data_feed(
    symbol_id: int,
    symbol: str = "",
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001
) -> HotSpineData: ...

def create_hotspine_feed(
    shm_name: str = "/btquant_hotspine",
    symbol_id: int = None,
    symbol: str = "",
    batch_mode: bool = False,
    poll_interval: float = 0.0001
) -> HotSpineFeed: ...
```

### 7. HotSpineSQLIntegration (`hotspine/sql_integration.py`)

Asynchronous SQL storage for HotSpine data:

```python
class HotSpineSQLIntegration:
    def __init__(self, 
                 sql_config: Optional[MSSQLConfig] = None,
                 config: Optional[HotSpineConfig] = None):
    
    # Storage operations
    def start_async_storage(self): ...
    def stop_async_storage(self): ...
    def store_trade_async(self, trade: HotTrade) -> bool:
    def store_ohlcv_async(self, ohlcv_data: Dict[str, Any]) -> bool:
    
    # Replay and analytics
    def get_historical_trades(self, exchange: str, symbol: str,
                             start: datetime, end: Optional[datetime] = None,
                             limit: Optional[int] = None) -> List[Dict]:
    def get_historical_ohlcv(self, exchange: str, symbol: str, timeframe: str,
                            start: datetime, end: Optional[datetime] = None,
                            limit: Optional[int] = None) -> List[Dict]:
    def get_latest_price(self, exchange: str, symbol: str) -> Optional[float]:
    
    # Monitoring
    def is_healthy(self) -> bool:
    def get_database_stats(self) -> Dict[str, Any]:
    def get_storage_stats(self) -> Dict[str, Any]:
```

**Storage Stats Output:**
```python
{
    'trades_stored': int,
    'storage_errors': int,
    'queue_size': int,
    'last_storage_time_ms': float,
    'running': bool,
    'healthy': bool,
    'reconnect_attempts': int,
    'avg_storage_latency': float
}
```

## Usage Examples

### Live Trading with HotSpine

```python
from backtrader.livetrading import livetrade_hotspine

livetrade_hotspine(
    symbol_id=1,                          # BTC-USDT on Binance
    strategy_class=MyStrategy,
    shm_name="/btquant_hotspine",
    batch_mode=False,
    poll_interval=0.0001,
    take_profit=2.0,
    stop_loss=1.0
)
```

### Multi-Symbol Trading

```python
from backtrader.livetrading import livetrade_hotspine_multi_symbol

livetrade_hotspine_multi_symbol(
    symbol_ids=[1, 2, 3],                 # BTC, ETH, SOL
    strategy=MyStrategy,
    shm_name="/btquant_hotspine"
)
```

### Using HotSpineData Feed Directly

```python
import backtrader as bt
from backtrader.feeds.hotspine_feed import HotSpineData

cerebro = bt.Cerebro()

data = HotSpineData(
    symbol_id=1,
    shm_name="/btquant_hotspine",
    batch_mode=False,
    poll_interval=0.0001
)

cerebro.adddata(data)
cerebro.addstrategy(MyStrategy, backtest=False)
cerebro.run(live=True, runonce=False)
```

### SQL Integration

```python
from backtrader.hotspine.sql_integration import HotSpineSQLIntegration
from backtrader.bigbraincentral.storage_mssql import MSSQLConfig

config = MSSQLConfig(server="localhost", database="BTQ_MarketData")
integration = HotSpineSQLIntegration(sql_config=config)

integration.start_async_storage()

# Store trade asynchronously
integration.store_trade_async(trade)

# Query historical data
trades = integration.get_historical_trades(
    exchange="binance",
    symbol="BTC-USDT",
    start=datetime(2024, 1, 1)
)

integration.stop_async_storage()
```

## Performance Considerations

- **Poll Interval**: Default 100 microseconds (`0.0001s`). Lower values increase CPU usage but reduce latency.
- **Batch Mode**: Enable for higher throughput at slight latency cost.
- **SQL Queue**: Size 10000 by default. Trades are dropped if queue fills.
- **Shared Memory**: Default capacity 1,000,000 trades. C++ collector must match.
