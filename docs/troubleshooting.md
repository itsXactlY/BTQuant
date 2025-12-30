# BTQuant Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues with BTQuant. It covers installation problems, configuration errors, data issues, and runtime errors.

## Installation Issues

### Permission Denied Errors

**Problem:** Installation fails with permission errors.

**Solutions:**
```bash
# Make scripts executable
chmod +x Installers/install.sh
chmod +x dependencies/setup.py

# Run installer with sudo if needed
sudo bash Installers/install.sh

# Or install in user directory
pip install --user dependencies/
```

### Missing Dependencies

**Problem:** Installation fails due to missing system packages.

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential python3-dev unixodbc-dev git

# CentOS/RHEL
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y python3-devel unixODBC-devel git

# Arch Linux
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm base-devel pybind11 unixodbc git tk
```

### Virtual Environment Issues

**Problem:** Virtual environment not working properly.

**Solutions:**
```bash
# Recreate virtual environment
rm -rf .btq
python3 -m venv .btq
source .btq/bin/activate
pip install --upgrade pip setuptools wheel

# Verify virtual environment
which python
which pip
```

### Python Version Issues

**Problem:** Python version compatibility issues.

**Solutions:**
```bash
# Check Python version
python3 --version

# Ensure Python 3.12 or 3.13
# Install specific version if needed
sudo apt-get install python3.13 python3.13-dev python3.13-venv

# Create venv with specific Python version
python3.13 -m venv .btq
```

### Package Installation Failures

**Problem:** pip install fails with compilation errors.

**Solutions:**
```bash
# Upgrade pip and build tools
pip install --upgrade pip setuptools wheel

# Install with verbose output
pip install -v dependencies/

# Install dependencies individually
cd dependencies
pip install . --no-deps
pip install pybind11
pip install pyodbc
pip install ccxt
```

## Configuration Issues

### Database Connection Problems

**Problem:** Cannot connect to SQL Server.

**Solutions:**
```bash
# Test ODBC connection
isql -v your_dsn your_username your_password

# Check SQL Server status
sudo systemctl status mssql-server

# Test connection string
python3 -c "
import pyodbc
conn_str = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=BinanceData;UID=SA;PWD=YourStrong!Passw0rd;TrustServerCertificate=yes;'
try:
    conn = pyodbc.connect(conn_str)
    print('Connection successful')
    conn.close()
except Exception as e:
    print(f'Connection failed: {e}')
"
```

**Common SQL Server Issues:**
```bash
# Enable SQL Server if disabled
sudo systemctl enable mssql-server
sudo systemctl start mssql-server

# Check SQL Server configuration
sudo /opt/mssql/bin/mssql-conf setup

# Allow firewall through SQL Server port (1433)
sudo ufw allow 1433/tcp
```

### Exchange Configuration Issues

**Problem:** Exchange API connection fails.

**Solutions:**
```python
# Test CCXT connection
import ccxt
exchange = ccxt.binance({
    'apiKey': 'your_api_key',
    'secret': 'your_secret',
    'enableRateLimit': True,
})
try:
    markets = exchange.load_markets()
    print("Exchange connection successful")
except Exception as e:
    print(f"Exchange connection failed: {e}")
```

**Common Exchange Issues:**
- **Invalid API keys**: Double-check API key and secret
- **Rate limiting**: Enable rate limiting in configuration
- **Network issues**: Check firewall and proxy settings
- **Testnet vs Live**: Ensure correct endpoint for testnet/live

### Hotspine Configuration Issues

**Problem:** Hotspine data collection fails.

**Solutions:**
```bash
# Check Hotspine dependencies
ldd dependencies/ccapi/example/src/market_data_collector/mssql_bulk_inserter

# Verify C++ compilation
cd dependencies/ccapi
make clean
make

# Check Hotspine configuration
python3 -c "
from dependencies.ccapi.example.src.market_data_collector.market_data_collector import MarketDataCollector
config = {'exchanges': ['binance'], 'symbols': ['BTC/USDT']}
collector = MarketDataCollector(config)
print('Hotspine configuration valid')
"
```

## Data Issues

### No Data Available

**Problem:** Backtest fails with "No data available" error.

**Solutions:**
```python
# Test data fetching
from backtrader.utils.ccxt_data import get_crypto_data

try:
    data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02', '1h', 'binance')
    if data is None or data.is_empty():
        print("No data returned")
    else:
        print(f"Data shape: {data.shape}")
except Exception as e:
    print(f"Data fetch failed: {e}")
```

**Common Data Issues:**
- **Date range too large**: Try smaller date ranges
- **Invalid symbol**: Check symbol format (e.g., 'BTC/USDT')
- **Exchange unavailable**: Try different exchange
- **Rate limits**: Add delays between requests

### Data Format Issues

**Problem:** Data format errors or missing columns.

**Solutions:**
```python
# Check data structure
if data is not None:
    print("Columns:", data.columns)
    print("Data types:", data.dtypes)
    print("Sample data:")
    print(data.head())
    
    # Ensure required columns exist
    required_columns = ['TimestampStart', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
```

### Database Data Issues

**Problem:** SQL Server data queries fail.

**Solutions:**
```python
# Test database query
from backtrader.feeds.mssql_crypto import get_database_data

try:
    df = get_database_data(
        ticker='BTC',
        start_date='2024-01-01',
        end_date='2024-01-02',
        time_resolution='1h',
        pair='USDT'
    )
    print(f"Database query successful. Rows: {len(df)}")
except Exception as e:
    print(f"Database query failed: {e}")
```

## Runtime Errors

### Strategy Execution Errors

**Problem:** Strategy fails during execution.

**Solutions:**
```python
# Enable debug mode
class MyStrategy(BaseStrategy):
    params = (('debug', True),)
    
    def next(self):
        try:
            # Your strategy logic
            super().next()
        except Exception as e:
            print(f"Strategy error: {e}")
            import traceback
            traceback.print_exc()

# Add error handling to backtest
try:
    result = backtest(MyStrategy, data=data)
except Exception as e:
    print(f"Backtest failed: {e}")
    import traceback
    traceback.print_exc()
```

### Memory Issues

**Problem:** Out of memory errors during backtesting.

**Solutions:**
```python
# Enable garbage collection
import gc
import sys

# Force garbage collection
gc.collect()

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Reduce data size
# - Use smaller date ranges
# - Use larger timeframes
# - Limit number of coins in bulk backtest
```

### Performance Issues

**Problem:** Backtests run too slowly.

**Solutions:**
```python
# Optimize data loading
from backtrader.utils.backtest import PolarsDataLoader

loader = PolarsDataLoader()
data = loader.load_data(spec, use_cache=True)  # Enable caching

# Reduce backtest complexity
# - Use fewer indicators
# - Reduce parameter ranges in optimization
# - Use smaller datasets for testing

# Enable parallel processing
results = bulk_backtest(
    strategy=MyStrategy,
    coins=['BTC', 'ETH'],  # Limit coins
    max_workers=4,         # Adjust based on CPU cores
)
```

### Live Trading Issues

**Problem:** Live trading fails or behaves unexpectedly.

**Solutions:**
```python
# Test in sandbox mode first
ccxt_config = {
    'apiKey': 'your_api_key',
    'secret': 'your_secret',
    'sandbox': True,  # Enable sandbox
}

# Monitor order execution
class MyLiveStrategy(BaseStrategy):
    def notify_order(self, order):
        if order.status in [order.Completed]:
            print(f"Order completed: {order.executed.price}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order failed: {order.status}")

# Check exchange connectivity
import ccxt.async_support as ccxt
import asyncio

async def test_exchange():
    exchange = ccxt.binance({
        'apiKey': 'your_api_key',
        'secret': 'your_secret',
    })
    try:
        await exchange.load_markets()
        print("Exchange connection successful")
    except Exception as e:
        print(f"Exchange connection failed: {e}")
    finally:
        await exchange.close()

asyncio.run(test_exchange())
```

## Common Error Messages

### "ModuleNotFoundError: No module named 'backtrader'"

**Solution:**
```bash
# Ensure virtual environment is activated
source .btq/bin/activate

# Reinstall dependencies
cd dependencies
pip install .

# Verify installation
python3 -c "import backtrader; print(backtrader.__version__)"
```

### "pyodbc.Error: ('08001', '[08001] [Microsoft][ODBC Driver 18 for SQL Server]SSL Provider"

**Solution:**
```python
# Add TrustServerCertificate to connection string
connection_string = (
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER=localhost;'
    f'DATABASE=BinanceData;'
    f'UID=SA;'
    f'PWD=YourStrong!Passw0rd;'
    f'TrustServerCertificate=yes;'
)
```

### "ccxt.base.errors.AuthenticationError"

**Solution:**
```python
# Check API credentials
ccxt_config = {
    'apiKey': 'your_correct_api_key',
    'secret': 'your_correct_secret',
    'enableRateLimit': True,
}

# Test with minimal code
import ccxt
exchange = ccxt.binance(ccxt_config)
try:
    balance = exchange.fetch_balance()
    print("Authentication successful")
except ccxt.AuthenticationError as e:
    print(f"Authentication failed: {e}")
```

### "TypeError: 'NoneType' object is not subscriptable"

**Solution:**
```python
# Check for None values
if data is not None:
    # Safe access
    if hasattr(data, 'close') and len(data) > 0:
        price = data.close[0]
    else:
        print("Data is empty or invalid")
else:
    print("Data is None")
```

### "IndexError: list index out of range"

**Solution:**
```python
# Check data length before accessing
if len(self.data) > 0:
    price = self.data.close[0]
else:
    return  # Skip if no data

# Use safe indexing
try:
    price = self.data.close[0]
except IndexError:
    return
```

## Debugging Strategies

### Enable Debug Mode

```python
# Enable debug in strategy
class MyStrategy(BaseStrategy):
    params = (('debug', True),)
    
    def next(self):
        if self.p.debug:
            print(f"Price: {self.data.close[0]}")
            print(f"Position: {self.position.size}")
        super().next()
```

### Add Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use in strategy
class MyStrategy(BaseStrategy):
    def next(self):
        logger.debug(f"Processing bar: {self.data.datetime.datetime(0)}")
        super().next()
```

### Use Interactive Debugging

```python
# Add breakpoint
import pdb

class MyStrategy(BaseStrategy):
    def next(self):
        if some_condition:
            pdb.set_trace()  # Execution will pause here
        super().next()
```

### Monitor System Resources

```python
import psutil
import time

def monitor_resources():
    """Monitor CPU and memory usage"""
    while True:
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        print(f"CPU: {cpu}%, Memory: {memory}%")
        time.sleep(5)

# Run in separate thread
import threading
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()
```

## Getting Help

### Check Logs

BTQuant generates detailed logs. Check:
- Console output for immediate errors
- Log files in working directory
- Exchange API logs for connectivity issues

### Community Support

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Review relevant sections
- **Examples**: Check `Examples/` directory for working code

### Professional Support

For enterprise users:
- Contact BTQuant support team
- Provide detailed error logs
- Share minimal reproduction code

## Prevention Best Practices

### Regular Maintenance

```bash
# Update dependencies regularly
pip list --outdated
pip install --upgrade package_name

# Clean cache periodically
rm -rf ~/.cache/pip
rm -rf .btq_cache

# Backup configuration
cp dependencies/backtrader/dontcommit.py backup_dontcommit.py
```

### Testing Strategy

```python
# Always test with small datasets first
def test_strategy():
    # Use minimal data
    data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02', '1h', 'binance')
    
    # Run quick test
    result = backtest(MyStrategy, data=data, init_cash=1000, backtest=True)
    
    # Verify results
    assert result > 0, "Backtest should return positive value"
    print("Strategy test passed")

# Run before major changes
test_strategy()
```

### Configuration Validation

```python
def validate_setup():
    """Validate BTQuant setup"""
    errors = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 12):
        errors.append("Python 3.12+ required")
    
    # Check key imports
    try:
        import backtrader
    except ImportError:
        errors.append("Backtrader not installed")
    
    try:
        import pyodbc
    except ImportError:
        errors.append("pyodbc not installed")
    
    # Check database connection
    try:
        from backtrader.dontcommit import connection_string
        import pyodbc
        conn = pyodbc.connect(connection_string)
        conn.close()
    except Exception as e:
        errors.append(f"Database connection failed: {e}")
    
    if errors:
        print("Setup validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Setup validation passed")
    return True

validate_setup()
```

This troubleshooting guide covers the most common issues you might encounter with BTQuant. If you can't find a solution here, check the FAQ section or seek community support.