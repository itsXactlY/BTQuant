# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with BTQuant. Follow the systematic approach for efficient problem resolution.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Issues](#data-issues)
- [Strategy Issues](#strategy-issues)
- [Performance Issues](#performance-issues)
- [Live Trading Issues](#live-trading-issues)
- [Database Issues](#database-issues)
- [HotSpine Issues](#hotspine-issues)
- [System Diagnostics](#system-diagnostics)

## Installation Issues

### Python Version Problems

**Symptoms:**
- Import errors
- Syntax errors
- Compatibility issues

**Solutions:**
```bash
# Check Python version
python3 --version
# Should be 3.12 or 3.13

# Check virtual environment
which python3
# Should point to .btq/bin/python3

# Recreate virtual environment
rm -rf .btq
python3 -m venv .btq
source .btq/bin/activate
pip install --upgrade pip setuptools wheel
```

### Dependency Installation Failures

**Symptoms:**
- Pip install errors
- Missing packages
- Compilation errors

**Solutions:**
```bash
# Update pip and tools
pip install --upgrade pip setuptools wheel

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential python3-dev unixodbc-dev git

# Clear pip cache
pip cache purge

# Reinstall dependencies
cd dependencies
pip install .
```

### Permission Errors

**Symptoms:**
- Access denied errors
- File creation failures
- Directory permission issues

**Solutions:**
```bash
# Fix file permissions
chmod +x Installers/install.sh
chmod 755 dependencies/
chmod 644 dependencies/setup.py

# Check directory ownership
ls -la BTQuant/
# Should be owned by your user

# Fix ownership if needed
sudo chown -R $USER:$USER BTQuant/
```

## Data Issues

### CCXT Data Retrieval Failures

**Symptoms:**
- Empty dataframes
- Network errors
- Rate limit errors

**Diagnostics:**
```python
# Test basic connectivity
import ccxt
exchange = ccxt.binance()
print("Available markets:", len(exchange.load_markets()))

# Test specific symbol
try:
    ticker = exchange.fetch_ticker('BTC/USDT')
    print("BTC/USDT ticker:", ticker['last'])
except Exception as e:
    print("Error:", e)
```

**Solutions:**
```python
# Check internet connectivity
ping -c 4 8.8.8.8

# Test exchange availability
curl -I https://api.binance.com/api/v3/ping

# Use different exchange
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'coinbase')

# Add retry logic
import time
for attempt in range(3):
    try:
        data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(5)
```

### Data Validation Errors

**Symptoms:**
- NaN values in data
- Invalid price/volume data
- Missing timestamps

**Validation Script:**
```python
def validate_data_comprehensive(data):
    """Comprehensive data validation"""
    issues = []

    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Check for NaN values
    nan_counts = data.isnull().sum()
    if nan_counts.any():
        issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

    # Check price validity
    if (data['close'] <= 0).any():
        issues.append("Invalid closing prices (zero or negative)")

    if (data['volume'] < 0).any():
        issues.append("Negative volumes found")

    # Check OHLC relationships
    invalid_ohlc = data[(data['high'] < data['low']) |
                       (data['high'] < data['open']) |
                       (data['high'] < data['close']) |
                       (data['low'] > data['open']) |
                       (data['low'] > data['close'])]
    if not invalid_ohlc.empty:
        issues.append(f"Invalid OHLC relationships in {len(invalid_ohlc)} rows")

    # Check timestamp ordering
    if not data.index.is_monotonic_increasing:
        issues.append("Timestamps not in chronological order")

    return issues

# Usage
issues = validate_data_comprehensive(data)
if issues:
    print("Data validation issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("Data validation passed")
```

### Database Connection Issues

**Symptoms:**
- ODBC errors
- Connection timeouts
- Authentication failures

**Diagnostics:**
```bash
# Test ODBC connection
isql -v "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=master;UID=SA;PWD=YourPassword;TrustServerCertificate=yes"

# Check SQL Server status
sudo systemctl status mssql-server

# Test Python connection
python3 -c "
import pyodbc
try:
    conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=master;UID=SA;PWD=YourPassword;TrustServerCertificate=yes')
    print('Connection successful')
    conn.close()
except Exception as e:
    print(f'Connection failed: {e}')
"
```

## Strategy Issues

### Indicator Calculation Errors

**Symptoms:**
- NaN indicator values
- Unexpected indicator behavior
- Performance issues

**Debugging:**
```python
# Enable transparency
from backtrader import transparencypatch

patch = transparencypatch.TransparencyPatch()
patch.debug = True
patch.apply_indicator_patch()

# Monitor calculations
class DebugStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close)

    def next(self):
        # Log indicator values
        self.log(f"RSI: {self.rsi[0]:.4f}")
        self.log(f"MACD: {self.macd.macd[0]:.6f}")
        self.log(f"Data close: {self.data.close[0]:.4f}")

        # Check for NaN
        if math.isnan(self.rsi[0]):
            self.log("ERROR: RSI is NaN!")
        if math.isnan(self.macd.macd[0]):
            self.log("ERROR: MACD is NaN!")
```

### Strategy Logic Errors

**Symptoms:**
- No trades executed
- Unexpected trade timing
- Wrong position sizing

**Debugging Template:**
```python
class DebugStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.sma_fast = bt.indicators.SMA(self.data.close, period=10)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=30)

    def next(self):
        # Log conditions
        rsi_value = self.rsi[0]
        fast_sma = self.sma_fast[0]
        slow_sma = self.sma_slow[0]
        close_price = self.data.close[0]

        self.log(f"RSI: {rsi_value:.2f}, Fast SMA: {fast_sma:.2f}, Slow SMA: {slow_sma:.2f}, Close: {close_price:.2f}")

        # Check entry conditions
        if not self.buy_executed:
            rsi_condition = rsi_value < 30
            sma_condition = fast_sma > slow_sma

            self.log(f"Entry conditions - RSI < 30: {rsi_condition}, Fast > Slow: {sma_condition}")

            if rsi_condition and sma_condition:
                self.log("BUY SIGNAL - Creating order")
                self.create_order('BUY')
            else:
                self.log("No buy signal")

        # Check exit conditions
        elif self.buy_executed:
            rsi_condition = rsi_value > 70
            sma_condition = fast_sma < slow_sma

            self.log(f"Exit conditions - RSI > 70: {rsi_condition}, Fast < Slow: {sma_condition}")

            if rsi_condition or sma_condition:
                self.log("SELL SIGNAL - Closing position")
                self.close_all_positions()
            else:
                self.log("Holding position")
```

### Parameter Optimization Issues

**Symptoms:**
- Optimization not converging
- Poor out-of-sample performance
- Overfitting

**Solutions:**
```python
# Use walk-forward analysis instead of simple optimization
from backtrader.utils.backtest import walk_forward_analysis

results = walk_forward_analysis(
    strategy_class=MyStrategy,
    data=data,
    train_window=252,  # 1 year training
    test_window=21     # 1 month testing
)

# Analyze walk-forward efficiency
def analyze_walk_forward(results):
    returns = [r['return'] for r in results]
    avg_return = sum(returns) / len(returns)
    positive_periods = sum(1 for r in returns if r > 0)
    win_rate = positive_periods / len(returns)

    print(f"Average return: {avg_return:.2f}%")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Total periods: {len(results)}")

analyze_walk_forward(results)
```

## Performance Issues

### Memory Problems

**Symptoms:**
- Out of memory errors
- Slow performance
- System freezing

**Solutions:**
```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

    if memory_mb > 2000:  # 2GB threshold
        print("WARNING: High memory usage!")

# Optimize data types
data = data.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'float32'
})

# Process in chunks for large datasets
chunk_size = 10000
for i in range(0, len(data), chunk_size):
    chunk = data.iloc[i:i+chunk_size]
    # Process chunk
    process_data_chunk(chunk)
```

### CPU Performance Issues

**Symptoms:**
- Slow backtests
- High CPU usage
- System unresponsiveness

**Solutions:**
```python
# Profile performance
import cProfile
import pstats

def profile_strategy():
    pr = cProfile.Profile()
    pr.enable()

    # Run backtest
    result = backtest(strategy=MyStrategy, data=data)

    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

profile_strategy()

# Optimize indicator calculations
class OptimizedStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

        # Calculate indicators once
        self.rsi = bt.indicators.RSI(self.data.close, period=14, plot=False)
        self.macd = bt.indicators.MACD(self.data.close, plot=False)

        # Cache expensive calculations
        self.rsi_cache = {}
        self.macd_cache = {}
```

### Database Performance Issues

**Symptoms:**
- Slow queries
- Timeout errors
- High resource usage

**Solutions:**
```sql
-- Analyze slow queries
SELECT
    qs.sql_handle,
    qs.execution_count,
    qs.total_worker_time / qs.execution_count as avg_cpu_time,
    qs.total_elapsed_time / qs.execution_count as avg_elapsed_time,
    SUBSTRING(qt.text, (qs.statement_start_offset/2)+1,
        ((CASE qs.statement_end_offset
            WHEN -1 THEN DATALENGTH(qt.text)
            ELSE qs.statement_end_offset
        END - qs.statement_start_offset)/2)+1) as query_text
FROM sys.dm_exec_query_stats qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) qt
ORDER BY qs.total_worker_time / qs.execution_count DESC;

-- Check index usage
SELECT
    OBJECT_NAME(s.object_id) as table_name,
    i.name as index_name,
    s.user_seeks,
    s.user_scans,
    s.user_lookups
FROM sys.dm_db_index_usage_stats s
JOIN sys.indexes i ON s.object_id = i.object_id AND s.index_id = i.index_id
WHERE s.database_id = DB_ID() AND OBJECT_NAME(s.object_id) LIKE '%trades%';
```

## Live Trading Issues

### Order Execution Failures

**Symptoms:**
- Orders not executing
- Rejection errors
- Slippage issues

**Diagnostics:**
```python
# Test exchange connectivity
import ccxt

exchange = ccxt.binance({
    'apiKey': 'your_api_key',
    'secret': 'your_secret'
})

try:
    # Test balance
    balance = exchange.fetch_balance()
    print("Balance:", balance['BTC'])

    # Test ticker
    ticker = exchange.fetch_ticker('BTC/USDT')
    print("BTC/USDT price:", ticker['last'])

    # Test order (paper trade)
    order = exchange.create_order(
        symbol='BTC/USDT',
        type='limit',
        side='buy',
        amount=0.001,
        price=ticker['last'] * 0.99  # 1% below market
    )
    print("Test order:", order)

except Exception as e:
    print(f"Exchange error: {e}")
```

### Synchronization Issues

**Symptoms:**
- Strategy running behind
- Data delays
- Timing mismatches

**Solutions:**
```python
# Check system time synchronization
date
timedatectl status

# Test latency to exchanges
ping -c 4 api.binance.com

# Monitor strategy timing
class TimingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.last_bar_time = None

    def next(self):
        current_time = self.data.datetime[0]
        if self.last_bar_time:
            time_diff = current_time - self.last_bar_time
            self.log(f"Bar interval: {time_diff} seconds")

            if time_diff > 400:  # More than 6.67 minutes for 1h bars
                self.log("WARNING: Data delay detected!")

        self.last_bar_time = current_time
```

## Database Issues

### Connection Pooling Problems

**Symptoms:**
- Connection exhausted errors
- Performance degradation
- Memory leaks

**Solutions:**
```python
# Implement connection pooling
import pyodbc

class DatabasePool:
    def __init__(self, connection_string, pool_size=5):
        self.connection_string = connection_string
        self.pool = []
        self.pool_size = pool_size

    def get_connection(self):
        if self.pool:
            return self.pool.pop()

        return pyodbc.connect(self.connection_string)

    def return_connection(self, conn):
        if len(self.pool) < self.pool_size:
            self.pool.append(conn)
        else:
            conn.close()

# Usage
db_pool = DatabasePool(connection_string)

def query_data(query):
    conn = db_pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    finally:
        db_pool.return_connection(conn)
```

### Index Fragmentation

**Symptoms:**
- Degrading query performance
- High CPU usage
- Slow backups

**Solutions:**
```sql
-- Check fragmentation
SELECT
    OBJECT_NAME(ips.object_id) as table_name,
    i.name as index_name,
    ips.avg_fragmentation_in_percent,
    ips.page_count
FROM sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL, NULL) ips
JOIN sys.indexes i ON ips.object_id = i.object_id AND ips.index_id = i.index_id
WHERE ips.avg_fragmentation_in_percent > 10;

-- Rebuild fragmented indexes
ALTER INDEX IX_trades_ts_exchange ON dbo.trades REBUILD;
ALTER INDEX IX_klines_open_time ON binance_btcusdt_klines REBUILD;
```

## HotSpine Issues

### Shared Memory Problems

**Symptoms:**
- Reader can't connect
- Buffer overflow
- Permission errors

**Diagnostics:**
```bash
# Check shared memory segments
ls -la /dev/shm/ | grep hotspine

# Check permissions
ls -la /dev/shm/btquant_hotspine

# Test reader connection
python3 -c "
from backtrader.hotspine.reader import HotSpineReader
try:
    reader = HotSpineReader('/btquant_hotspine')
    print('Reader connected successfully')
    print('Lost trades:', reader.get_lost_count())
    print('Healthy:', reader.is_healthy())
except Exception as e:
    print(f'Reader connection failed: {e}')
"
```

### Performance Issues

**Symptoms:**
- High latency
- Lost trades
- Buffer overflow

**Solutions:**
```python
# Monitor HotSpine performance
def monitor_hotspine():
    reader = HotSpineReader('/btquant_hotspine')

    while True:
        lost_count = reader.get_lost_count()
        if lost_count > 0:
            print(f"WARNING: {lost_count} trades lost")

        # Check buffer utilization
        # (Implementation depends on reader API)

        time.sleep(60)  # Check every minute

# Increase buffer size if needed
# Modify HotSpine writer configuration
# buffer_size = 2000000  # 2M trades instead of 1M
```

## System Diagnostics

### Comprehensive Health Check

```python
def system_health_check():
    """Comprehensive system health check"""
    health = {}

    # Python environment
    try:
        import backtrader as bt
        health['backtrader'] = f'Version {bt.__version__}'
    except Exception as e:
        health['backtrader'] = f'Error: {e}'

    # Database connectivity
    try:
        import pyodbc
        conn = pyodbc.connect(connection_string)
        conn.close()
        health['database'] = 'Connected'
    except Exception as e:
        health['database'] = f'Error: {e}'

    # Exchange connectivity
    try:
        import ccxt
        exchange = ccxt.binance()
        exchange.load_markets()
        health['exchange'] = 'Connected'
    except Exception as e:
        health['exchange'] = f'Error: {e}'

    # HotSpine status
    try:
        from backtrader.hotspine.reader import HotSpineReader
        reader = HotSpineReader('/btquant_hotspine')
        health['hotspine'] = f'Connected, lost: {reader.get_lost_count()}'
    except Exception as e:
        health['hotspine'] = f'Error: {e}'

    # System resources
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        health['cpu'] = f'{cpu_percent:.1f}%'
        health['memory'] = f'{memory.percent:.1f}% used'
        health['disk'] = f'{disk.percent:.1f}% used'
    except Exception as e:
        health['system'] = f'Error: {e}'

    return health

# Run health check
health = system_health_check()
print("System Health Check:")
for component, status in health.items():
    print(f"  {component}: {status}")
```

### Log Analysis

```python
# Analyze BTQuant logs
def analyze_logs(log_file='btquant.log'):
    """Analyze log files for issues"""
    import re
    from collections import Counter

    error_pattern = re.compile(r'ERROR|CRITICAL')
    warning_pattern = re.compile(r'WARNING')

    errors = []
    warnings = []

    with open(log_file, 'r') as f:
        for line in f:
            if error_pattern.search(line):
                errors.append(line.strip())
            elif warning_pattern.search(line):
                warnings.append(line.strip())

    print(f"Found {len(errors)} errors and {len(warnings)} warnings")

    if errors:
        print("\nRecent Errors:")
        for error in errors[-5:]:  # Last 5 errors
            print(f"  {error}")

    if warnings:
        print("\nRecent Warnings:")
        for warning in warnings[-5:]:  # Last 5 warnings
            print(f"  {warning}")

analyze_logs()
```

### Performance Profiling

```python
# Profile backtest performance
import cProfile
import pstats
import io

def profile_backtest(strategy_class, data):
    """Profile backtest performance"""
    pr = cProfile.Profile()
    pr.enable()

    # Run backtest
    result = backtest(strategy=strategy_class, data=data)

    pr.disable()

    # Analyze results
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)  # Top 20 functions

    print("Performance Profile:")
    print(s.getvalue())

    return result

# Usage
result = profile_backtest(MyStrategy, data)
```

This troubleshooting guide provides systematic approaches to diagnose and resolve BTQuant issues. Start with the relevant section based on your symptoms and follow the diagnostic steps in order.