from backtrader.Jackrabbit.storage_mssql import MarketDataStorage, MSSQLConfig
import fast_mssql
from datetime import datetime

# Direct usage - properly defines timestamp_ms
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

# Your exact connection format
config = MSSQLConfig(
    server='localhost',
    database='BinanceData',
    username='SA',
    password='q?}33YIToo:H%xue$Kr*',
    driver='{ODBC Driver 18 for SQL Server}',
    trust_server_certificate=True
)

# Initialize storage
storage = MarketDataStorage(config)
storage.connect()

# Store OHLCV (synchronous)
storage.store_ohlcv({
    'exchange': 'binance',
    'symbol': 'btcusdt',
    'timestamp': timestamp,
    'timeframe': '1m',
    'open': 50000.0,
    'high': 50100.0,
    'low': 49900.0,
    'close': 50050.0,
    'volume': 10.5,
})

# Bulk insert (uses C++ bulk operations)
candles = [...]  # List of candle dicts
count = storage.bulk_store_ohlcv(candles)

# Query (direct C++ ODBC)
candles = storage.get_ohlcv('binance', 'btcusdt', '1m', start, end)

# Connection pool managed at C++ level
print(f"Pool size: {fast_mssql.get_pool_size()}")