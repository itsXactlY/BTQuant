from backtrader.Jackrabbit.storage_mssql import MarketDataStorage, MSSQLConfig
from datetime import datetime

# Configure
config = MSSQLConfig(
    server='localhost',
    database='BinanceData',
    username='SA',
    password='q?}33YIToo:H%xue$Kr*',
)

# Connect
storage = MarketDataStorage(config)
storage.connect()

# Store OHLCV
timestamp_ms = int(datetime.now().timestamp() * 1000)

storage.store_ohlcv({
    'exchange': 'binance',
    'symbol': 'btcusdt',
    'timestamp': timestamp_ms,
    'timeframe': '1m',
    'open': 50000.0,
    'high': 50100.0,
    'low': 49900.0,
    'close': 50050.0,
    'volume': 10.5,
})

# Store trade
storage.store_trade({
    'exchange': 'binance',
    'symbol': 'btcusdt',
    'timestamp': timestamp_ms,
    'trade_id': '12345',
    'price': 50000.0,
    'quantity': 0.1,
    'side': 'buy',
    'is_buyer_maker': True,
})

# Get latest price
price = storage.get_latest_price('binance', 'btcusdt')
print(f"Latest BTC price: ${price:.2f}")

# Get statistics
stats = storage.get_stats()
print(f"OHLCV records: {stats['ohlcv_records']:,}")
print(f"Trade records: {stats['trades_records']:,}")

# Disconnect
storage.disconnect()

from datetime import datetime

# Generate multiple candles
candles = []
base_time = int(datetime.now().timestamp() * 1000)

for i in range(100):
    timestamp_ms = base_time + (i * 60000)  # 1 minute apart

    candles.append({
        'exchange': 'binance',
        'symbol': 'btcusdt',
        'timestamp': timestamp_ms,
        'timeframe': '1m',
        'open': 50000.0 + i,
        'high': 50100.0 + i,
        'low': 49900.0 + i,
        'close': 50050.0 + i,
        'volume': 100.0,
    })

# Bulk insert
count = storage.bulk_store_ohlcv(candles)
print(f"Stored {count} candles")

from datetime import datetime, timedelta

# Get last hour of 1m candles
end = datetime.now()
start = end - timedelta(hours=1)

candles = storage.get_ohlcv(
    exchange='binance',
    symbol='btcusdt',
    timeframe='1m',
    start=start,
    end=end,
    limit=60
)

for candle in candles:
    print(f"{candle['timestamp']} | OHLC: {candle['open']}/{candle['high']}/{candle['low']}/{candle['close']}")
