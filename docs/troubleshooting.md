# BTQuant Troubleshooting Guide

This guide helps diagnose and resolve common issues with BTQuant.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [Data Feed Issues](#data-feed-issues)
- [Broker Issues](#broker-issues)
- [HotSpine Issues](#hotspine-issues)
- [Database Issues](#database-issues)
- [Strategy Issues](#strategy-issues)
- [Live Trading Issues](#live-trading-issues)

## Installation Issues

### Python Version Problems

**Symptoms:** Import errors, syntax errors, compatibility issues.

```bash
# Check Python version (requires 3.12+)
python3 --version

# Recreate virtual environment
rm -rf .btq
python3 -m venv .btq
source .btq/bin/activate
pip install --upgrade pip setuptools wheel
```

### Dependency Installation Failures

**Symptoms:** Pip errors, missing packages, compilation failures.

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential python3-dev unixodbc-dev git

# Clear pip cache and reinstall
pip cache purge
cd dependencies
pip install .
```

## Configuration Issues

### Missing API Keys

**Symptoms:** `RuntimeError: Missing API key/secret for {exchange}`

**Solutions:**

1. **Create config file:**
```bash
mkdir -p .btq/ccxt
cat > .btq/ccxt/binance_main.json << 'EOF'
{
    "apiKey": "your-api-key",
    "secret": "your-secret"
}
EOF
```

2. **Or set environment variables:**
```bash
export BTQ_BINANCE_MAIN_API_KEY="your-api-key"
export BTQ_BINANCE_MAIN_API_SECRET="your-secret"
```

### dontcommit.py Import Errors

**Symptoms:** `ImportError: cannot import name 'identify' from 'backtrader.dontcommit'`

**Solution:** Ensure `dontcommit.py` exists with all required fields:
```python
# dependencies/backtrader/dontcommit.py
identify = ""
jrr_webhook_url = "http://127.0.0.1:80"
discord_webhook_url = ''
telegram_api_id = 1111111
telegram_api_hash = ""
telegram_session_file = ".base.session"
telegram_channel = -100
# ... other fields
```

## Data Feed Issues

### CCXT Data Retrieval Failures

**Symptoms:** Empty data, network errors, rate limits.

```python
# Test basic connectivity
import ccxt
exchange = ccxt.binance()
print("Markets:", len(exchange.load_markets()))
ticker = exchange.fetch_ticker('BTC/USDT')
print("Price:", ticker['last'])
```

**Solutions:**
- Check internet connectivity: `ping -c 4 api.binance.com`
- Use different exchange: Add `exchange='coinbase'` to backtest()
- Increase retries: `CCXT(..., retries=5)`

### DatabaseOHLCV No Data

**Symptoms:** Feed returns no data, `None` from `_load()`.

**Diagnostics:**
```python
from backtrader.feeds.db_ohlcv_mssql import ReadOnlyOHLCV, MSSQLFeedConfig

config = MSSQLFeedConfig(server="localhost", database="BTQ_MarketData")
reader = ReadOnlyOHLCV(config, mode="global")

# Test query
data = reader.get_ohlcv("binance", "BTCUSDT", "1m", 
                        start=datetime(2024, 1, 1))
print(f"Rows: {len(data)}")
```

**Solutions:**
- Verify table exists: `SELECT * FROM sys.tables WHERE name='ohlcv'`
- Check symbol format: Try `BTCUSDT`, `BTC-USDT`, `BTC_USDT`
- Verify timeframe: Use `1m`, `5m`, `1h`, `1d`

### HotSpineData Connection Failure

**Symptoms:** `Failed to initialize HotSpine reader`

```bash
# Check shared memory exists
ls -la /dev/shm/ | grep hotspine

# Verify C++ writer is running
ps aux | grep hotspine
```

## Broker Issues

### JrrBroker Webhook Errors

**Symptoms:** Orders not executing, `Connection refused`

**Diagnostics:**
```bash
# Test JRR webhook
curl -X POST http://127.0.0.1:80 \
  -H "Content-Type: application/json" \
  -d '{"Exchange":"mimic","Market":"spot","Account":"default","Action":"Ping"}'
```

**Solutions:**
- Verify JackRabbitRelay is running
- Check `jrr_webhook_url` in `dontcommit.py`
- Verify `identify` string matches JRR setup

### CCXTBroker Order Failures

**Symptoms:** Orders rejected, `Insufficient balance`

```python
# Test exchange connectivity
broker = CCXTBroker("binance", "USDT", config)
print("Cash:", broker.getcash())
print("Position:", broker.get_position_info(data))
```

**Solutions:**
- Verify API key has trading permissions
- Check sufficient balance for order size
- Verify symbol format: `BTC/USDT` not `BTCUSDT`

## HotSpine Issues

### Shared Memory Not Found

**Symptoms:** `FileNotFoundError: [Errno 2] No such file or directory: '/dev/shm/btquant_hotspine'`

**Diagnostics:**
```bash
ls -la /dev/shm/
# Should show btquant_hotspine if C++ writer is running
```

**Solutions:**
- Start C++ HotSpine collector first
- Verify `shm_name` matches writer configuration
- Check permissions: `chmod 666 /dev/shm/btquant_hotspine`

### Buffer Overflow (Lost Trades)

**Symptoms:** `lost_count` increasing, trades being dropped.

```python
from backtrader.hotspine.reader import HotSpineReader

reader = HotSpineReader("/btquant_hotspine")
print("Lost trades:", reader.get_lost_count())
```

**Solutions:**
- Increase `shm_capacity` in HotSpineConfig (default: 1,000,000)
- Enable batch mode: `HotSpineData(batch_mode=True)`
- Reduce `poll_interval` for faster reading

### HotSpine SQL Integration Errors

**Symptoms:** `Failed to connect to SQL Server`

```python
from backtrader.hotspine.sql_integration import HotSpineSQLIntegration
from backtrader.bigbraincentral.storage_mssql import MSSQLConfig

config = MSSQLConfig(server="localhost", database="BTQ_MarketData")
integration = HotSpineSQLIntegration(sql_config=config)
print("Healthy:", integration.is_healthy())
```

**Solutions:**
- Verify SQL Server is running: `sudo systemctl status mssql-server`
- Test ODBC connection: `isql -v "DRIVER={ODBC Driver 18 for SQL Server};..."`
- Check `sql_queue_size` (default: 10000) - trades dropped if full

## Database Issues

### Connection Timeouts

**Symptoms:** `pyodbc.OperationalError: Login timeout expired`

```bash
# Test ODBC connection
isql -v "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=master;UID=SA;PWD=YourPassword;TrustServerCertificate=yes"

# Check SQL Server status
sudo systemctl status mssql-server
```

### Deadlock Errors

**Symptoms:** `Transaction was deadlocked`

The `ReadOnlyTradesAgg` class includes automatic deadlock retry:

```python
# Automatic retry with exponential backoff (up to 4 retries)
rows = trades_reader.get_ticks_by_id(exchange, symbol, last_id)
```

If deadlocks persist:
- Reduce concurrent queries
- Add proper indexes
- Check `READPAST` and `ROWLOCK` hints in queries

### OHLCV Table Not Found

**Symptoms:** `Invalid object name 'binance_btcusdt_klines'`

```sql
-- Check if table exists
SELECT * FROM sys.tables WHERE name LIKE '%btcusdt%'

-- Create manually if needed
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
    created_at DATETIME2 DEFAULT GETDATE()
);
```

## Strategy Issues

### Indicator NaN Values

**Symptoms:** Strategy produces unexpected results, NaN indicators.

```python
from backtrader import transparencypatch

# Enable transparency patch for debugging
patch = transparencypatch.TransparencyPatch()
patch.debug = True
patch.apply_indicator_patch()
```

### No Trades Executed

**Symptoms:** Strategy runs but never enters positions.

**Debugging:**
```python
class DebugStrategy(BaseStrategy):
    def buy_or_short_condition(self):
        print(f"Checking buy: close={self.data.close[0]}, buy_executed={self.buy_executed}")
        # Your logic here
        return False
```

**Common Causes:**
- `buy_executed` already True (position open)
- `percent_sizer` not set (position size = 0)
- Indicator warmup period not passed

### OrderTracker Issues

**Symptoms:** Position tracking incorrect after restart.

```python
# Verify active orders loaded from CSV
print(f"Active orders: {len(self.active_orders)}")
for order in self.active_orders:
    print(f"  {order.order_type} {order.size} @ {order.entry_price}")
```

## Live Trading Issues

### Exchange Connection Failures

**Symptoms:** `CCXT Network Error`, `Connection refused`

```python
import ccxt
exchange = ccxt.binance({'enableRateLimit': True})
try:
    exchange.load_markets()
    print("Connected")
except Exception as e:
    print(f"Failed: {e}")
```

### Synchronization Issues

**Symptoms:** Strategy running behind, data delays.

```bash
# Check system time
date
timedatectl status

# Test exchange latency
ping -c 4 api.binance.com
```

### Telegram Alert Failures

**Symptoms:** Alerts not sending, `Telegram client not authorized`

```python
from backtrader.brokers.jrrbroker import TelegramService

service = TelegramService(
    api_id=telegram_api_id,
    api_hash=telegram_api_hash,
    session_file=".base.session",
    channel_id=telegram_channel
)
# Must run async initialization
await service.initialize()
```

**Solutions:**
- Ensure `.base.session` file exists and is valid
- Re-authenticate if session expired
- Check `telegram_channel` is correct (negative for groups)

## System Diagnostics

### Comprehensive Health Check

```python
def system_health_check():
    health = {}
    
    # Check backtrader
    try:
        import backtrader as bt
        health['backtrader'] = 'OK'
    except Exception as e:
        health['backtrader'] = str(e)
    
    # Check database
    try:
        import fast_mssql
        health['fast_mssql'] = 'OK'
    except Exception as e:
        health['fast_mssql'] = str(e)
    
    # Check HotSpine
    try:
        from backtrader.hotspine.reader import HotSpineReader
        reader = HotSpineReader("/btquant_hotspine")
        health['hotspine'] = f'Connected, lost: {reader.get_lost_count()}'
    except Exception as e:
        health['hotspine'] = str(e)
    
    # Check system resources
    try:
        import psutil
        health['memory'] = f'{psutil.virtual_memory().percent:.1f}%'
        health['cpu'] = f'{psutil.cpu_percent():.1f}%'
    except:
        pass
    
    return health

# Run
for component, status in system_health_check().items():
    print(f"{component}: {status}")
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_backtest():
    pr = cProfile.Profile()
    pr.enable()
    
    result = backtest(MyStrategy, coin="BTC", interval="1h")
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
```
