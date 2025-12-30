# Configuration Guide

This guide covers all configuration options in BTQuant, from basic setup to advanced tuning for production deployments.

## Table of Contents

- [Basic Configuration](#basic-configuration)
- [Secrets and Credentials](#secrets-and-credentials)
- [Database Configuration](#database-configuration)
- [Exchange Configuration](#exchange-configuration)
- [Strategy Configuration](#strategy-configuration)
- [Performance Tuning](#performance-tuning)
- [Monitoring Configuration](#monitoring-configuration)
- [Environment Variables](#environment-variables)

## Basic Configuration

### Installation Verification

After installation, verify your setup:

```bash
# Check Python version
python3 --version

# Verify BTQuant installation
python3 -c "import backtrader as bt; print(f'Backtrader: {bt.__version__}')"

# Check virtual environment
which python3
# Should point to .btq/bin/python3
```

### Directory Structure

BTQuant expects the following structure:

```
BTQuant/
├── dependencies/backtrader/dontcommit.py    # Secrets and configuration
├── .btq/                                    # Virtual environment
├── docs/                                    # Documentation
├── Examples/                                # Example scripts
├── Installers/                              # Installation scripts
└── hotspine/                                # HotSpine components
```

## Secrets and Credentials

### dontcommit.py Configuration

The `dependencies/backtrader/dontcommit.py` file contains all sensitive configuration:

```python
# JackRabbit Relay Configuration
identify = "your_unique_identifier"                    # Your JRR account ID
jrr_webhook_url = "http://127.0.0.1:80"               # JRR webhook endpoint
jrr_order_history = "/path/to/jrr/order_history/"     # Order history path

# Web3/DeFi Configuration
bsc_privaccount1 = "0x..."                            # BSC private key
bsc_privaccountaddress = "0x..."                      # BSC wallet address
solana_privkey_base58 = "..."                         # Solana private key
solana_wallet_address = "..."                         # Solana wallet address

# Discord Integration
discord_webhook_url = "https://discord.com/api/webhooks/..."  # Discord webhook

# Telegram Integration
telegram_api_id = 12345678                            # Telegram API ID
telegram_api_hash = "your_api_hash"                   # Telegram API hash
telegram_session_file = ".base.session"               # Session file
telegram_channel = -1001234567890                     # Channel ID

# SQL Server Configuration
server = 'localhost'                                  # SQL Server host
candle_database = 'BigBrainCentral'                   # Candle database
optuna_database = 'OptunaBT'                          # Optimization database
username = 'SA'                                       # SQL Server username
password = 'YourStrong!Passw0rd'                      # SQL Server password
driver = '{ODBC Driver 18 for SQL Server}'            # ODBC driver

# Connection strings (auto-generated)
connection_string = (
    f'DRIVER={driver};'
    f'SERVER={server};'
    f'DATABASE={candle_database};'
    f'UID={username};'
    f'PWD={password};'
    f'TrustServerCertificate=yes;'
)

optuna_connection_string = (
    f'DRIVER={driver};'
    f'SERVER={server};'
    f'DATABASE={optuna_database};'
    f'UID={username};'
    f'PWD={password};'
    f'TrustServerCertificate=yes;'
)
```

### Security Best Practices

```python
# File permissions
import os
import stat

# Secure dontcommit.py
os.chmod('dependencies/backtrader/dontcommit.py', stat.S_IRUSR | stat.S_IWUSR)

# Environment variable fallback
import os

def get_secret(key, default=None):
    """Get secret from environment or dontcommit.py"""
    return os.getenv(key) or globals().get(key, default)

# Usage
api_key = get_secret('BINANCE_API_KEY')
api_secret = get_secret('BINANCE_API_SECRET')
```

## Database Configuration

### SQL Server Setup

#### Basic Installation
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y curl apt-transport-https
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/mssql-server-2019.list | tee /etc/apt/sources.list.d/mssql-server-2019.list
sudo apt-get update
sudo apt-get install -y mssql-server

# Setup SQL Server
sudo /opt/mssql/bin/mssql-conf setup
```

#### Database Creation
```sql
-- Create databases
CREATE DATABASE BigBrainCentral;
CREATE DATABASE OptunaBT;

-- Create login
CREATE LOGIN btquant WITH PASSWORD = 'YourStrong!Passw0rd';
CREATE USER btquant FOR LOGIN btquant;
ALTER ROLE db_owner ADD MEMBER btquant;
```

#### Performance Optimization
```sql
-- Memory configuration
EXEC sp_configure 'show advanced options', 1;
RECONFIGURE;
EXEC sp_configure 'max server memory (MB)', 8192;  -- 8GB for 16GB system
RECONFIGURE;

-- Query optimization
EXEC sp_configure 'cost threshold for parallelism', 50;
EXEC sp_configure 'max degree of parallelism', 4;
RECONFIGURE;

-- TempDB optimization
ALTER DATABASE tempdb MODIFY FILE (NAME = tempdev, SIZE = 4GB, FILEGROWTH = 1GB);
ALTER DATABASE tempdb MODIFY FILE (NAME = templog, SIZE = 2GB, FILEGROWTH = 512MB);
```

### BigBrainCentral Schema

#### Tables Creation
```sql
-- Trades table
CREATE TABLE dbo.trades (
    id BIGINT IDENTITY PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    market_type VARCHAR(20) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    side TINYINT NOT NULL,
    aggressor_flag BIT,
    ts_exchange DATETIME2(6) NOT NULL,
    ts_local DATETIME2(6) NOT NULL,
    created_at DATETIME2(6) DEFAULT GETUTCDATE()
);

-- Orderbook snapshots table
CREATE TABLE dbo.orderbook_snapshots (
    id BIGINT IDENTITY PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    market_type VARCHAR(20) NOT NULL,
    bids NVARCHAR(MAX),
    asks NVARCHAR(MAX),
    ts_exchange DATETIME2(6) NOT NULL,
    ts_local DATETIME2(6) NOT NULL,
    checksum BINARY(32),
    created_at DATETIME2(6) DEFAULT GETUTCDATE()
);

-- OHLCV tables (per symbol)
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

#### Indexing Strategy
```sql
-- Trades indexes
CREATE CLUSTERED INDEX IX_trades_ts_exchange
ON dbo.trades (exchange, symbol, ts_exchange);

CREATE NONCLUSTERED INDEX IX_trades_symbol_ts
ON dbo.trades (symbol, ts_exchange)
INCLUDE (price, size, side);

-- OHLCV indexes
CREATE CLUSTERED INDEX IX_klines_open_time
ON binance_btcusdt_klines (open_time);

CREATE NONCLUSTERED INDEX IX_klines_symbol_time
ON binance_btcusdt_klines (symbol, open_time)
INCLUDE (open, high, low, close, volume);
```

## Exchange Configuration

### CCXT Configuration

#### API Credentials
```python
# In dontcommit.py
ccxt_config = {
    'binance': {
        'apiKey': 'your_binance_api_key',
        'secret': 'your_binance_secret',
        'enableRateLimit': True,
        'rateLimit': 20,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    },
    'coinbase': {
        'apiKey': 'your_coinbase_api_key',
        'secret': 'your_coinbase_secret',
        'password': 'your_coinbase_passphrase'
    },
    'kraken': {
        'apiKey': 'your_kraken_api_key',
        'secret': 'your_kraken_secret'
    }
}
```

#### Rate Limiting
```python
# Custom rate limiting
ccxt_config['binance']['rateLimit'] = 10  # More conservative
ccxt_config['binance']['options']['recvWindow'] = 10000  # 10 second window

# Testnet configuration
ccxt_config['binance_testnet'] = {
    'apiKey': 'test_api_key',
    'secret': 'test_secret',
    'urls': {
        'api': 'https://testnet.binance.vision'
    },
    'test': True
}
```

### Native Exchange Integration

#### Binance Native
```python
# In dontcommit.py
binance_config = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'testnet': False,
    'recv_window': 10000,
    'requests_params': {
        'timeout': 5000
    }
}
```

#### Bitget Native
```python
bitget_config = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'passphrase': 'your_passphrase',
    'testnet': False
}
```

#### MEXC Native
```python
mexc_config = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'testnet': False
}
```

### Web3/DeFi Configuration

#### BSC Configuration
```python
bsc_config = {
    'rpc_url': 'https://bsc-dataseed.binance.org/',
    'chain_id': 56,
    'gas_price': 5,  # gwei
    'gas_limit': 200000,
    'pancakeswap_router': '0x10ED43C718714eb63d5aA57B78B54704E256024E'
}
```

#### Solana Configuration
```python
solana_config = {
    'rpc_url': 'https://api.mainnet-beta.solana.com',
    'commitment': 'confirmed',
    'preflight_commitment': 'processed'
}
```

## Strategy Configuration

### BaseStrategy Parameters

```python
class MyStrategy(BaseStrategy):
    params = (
        # Risk Management
        ('take_profit', 2.0),          # Take profit percentage
        ('stop_loss', 5.0),            # Stop loss percentage
        ('trailing_stop', 1.0),        # Trailing stop percentage

        # DCA Configuration
        ('dca_levels', 3),             # Number of DCA levels
        ('dca_percentage', 0.5),       # DCA percentage per level
        ('dca_spacing', 2.0),          # Price spacing between levels

        # Position Sizing
        ('risk_per_trade', 0.02),      # Risk per trade (2%)
        ('max_risk_per_trade', 0.05),  # Maximum risk per trade
        ('max_open_trades', 5),        # Maximum concurrent trades

        # Indicator Parameters
        ('fast_period', 10),           # Fast MA period
        ('slow_period', 30),           # Slow MA period
        ('rsi_period', 14),            # RSI period
        ('rsi_overbought', 70),        # RSI overbought level
        ('rsi_oversold', 30),          # RSI oversold level

        # Transparency
        ('capture_data', False),       # Enable transparency logging
        ('debug', False),              # Enable debug logging
    )
```

### Indicator Configuration

```python
def __init__(self):
    super().__init__()

    # Configure indicators with transparency
    from backtrader import transparencypatch
    if self.p.capture_data:
        patch = transparencypatch.TransparencyPatch()
        patch.debug = self.p.debug
        patch.apply_indicator_patch()

    # Initialize indicators
    self.rsi = bt.indicators.RSI(
        self.data.close,
        period=self.p.rsi_period
    )

    self.macd = bt.indicators.MACD(
        self.data.close,
        period_me1=12,
        period_me2=26,
        period_signal=9
    )

    self.atr = bt.indicators.ATR(
        self.data,
        period=14
    )
```

### Backtest Configuration

```python
# Backtest parameters
backtest_config = {
    'init_cash': 100000,              # Starting capital
    'commission': 0.001,              # Commission per trade (0.1%)
    'margin': 1.0,                    # Leverage (1.0 = no leverage)
    'stake': 100,                     # Fixed stake per trade

    # Analysis
    'quantstats': True,               # Generate QuantStats report
    'plot': True,                     # Generate plots
    'benchmark': 'BTC',               # Benchmark asset

    # Performance
    'max_workers': 4,                 # Parallel processing workers
    'opt_criteria': 'sharpe',         # Optimization criteria
    'opt_direction': 'max',           # Maximize criteria
}
```

## Performance Tuning

### Memory Optimization

```python
# Python memory settings
import os
os.environ['PYTHONMALLOC'] = 'malloc'
os.environ['MALLOC_ARENA_MAX'] = '1'

# DataFrame optimization
data = data.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'float32'
})

# Indicator caching
self.indicators_cache = {}
```

### CPU Optimization

```python
# Parallel processing
import multiprocessing
cpu_count = multiprocessing.cpu_count()

# Optimize backtest
results = optimize_backtest(
    strategy=MyStrategy,
    data=data,
    max_workers=min(cpu_count, 8),  # Limit workers
    fast_period=[10, 15, 20],
    slow_period=[30, 40, 50]
)
```

### Database Optimization

```sql
-- Query optimization
SET STATISTICS IO ON;
SET STATISTICS TIME ON;

-- Index usage analysis
SELECT
    OBJECT_NAME(s.object_id) as table_name,
    i.name as index_name,
    s.user_seeks,
    s.user_scans,
    s.user_lookups,
    s.user_updates
FROM sys.dm_db_index_usage_stats s
JOIN sys.indexes i ON s.object_id = i.object_id AND s.index_id = i.index_id
WHERE s.database_id = DB_ID();
```

### Network Optimization

```python
# Connection pooling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.3)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Timeout configuration
ccxt_config['binance']['timeout'] = 10000  # 10 seconds
```

## Monitoring Configuration

### Logging Configuration

```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btquant.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Strategy logging
logger = logging.getLogger('MyStrategy')

class MyStrategy(BaseStrategy):
    def log(self, txt, dt=None, doprint=False):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f'{dt.isoformat()} {txt}')
        if doprint:
            print(f'{dt.isoformat()} {txt}')
```

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram

# Performance metrics
trades_total = Counter('btquant_trades_total', 'Total trades executed')
portfolio_value = Gauge('btquant_portfolio_value', 'Current portfolio value')
sharpe_ratio = Gauge('btquant_sharpe_ratio', 'Current Sharpe ratio')

# Latency metrics
trade_latency = Histogram('btquant_trade_latency', 'Trade execution latency')

class MonitoredStrategy(BaseStrategy):
    def next(self):
        # Update metrics
        portfolio_value.set(self.broker.getvalue())

        # Trade execution
        if self.buy_signal():
            start_time = time.time()
            self.buy()
            trade_latency.observe(time.time() - start_time)
            trades_total.inc()
```

### Health Checks

```python
def health_check():
    """System health check"""
    checks = {}

    # Database connectivity
    try:
        conn = pyodbc.connect(connection_string)
        conn.close()
        checks['database'] = 'healthy'
    except Exception as e:
        checks['database'] = f'unhealthy: {e}'

    # Exchange connectivity
    try:
        exchange = ccxt.binance(ccxt_config['binance'])
        exchange.load_markets()
        checks['exchange'] = 'healthy'
    except Exception as e:
        checks['exchange'] = f'unhealthy: {e}'

    # HotSpine status
    try:
        reader = HotSpineReader('/btquant_hotspine')
        if reader.is_healthy():
            checks['hotspine'] = 'healthy'
        else:
            checks['hotspine'] = 'unhealthy'
    except Exception as e:
        checks['hotspine'] = f'unhealthy: {e}'

    return checks

# Usage
health = health_check()
for component, status in health.items():
    print(f"{component}: {status}")
```

## Environment Variables

### Development vs Production

```bash
# Development
export BTQUANT_ENV=development
export BTQUANT_DEBUG=true
export BTQUANT_LOG_LEVEL=DEBUG

# Production
export BTQUANT_ENV=production
export BTQUANT_DEBUG=false
export BTQUANT_LOG_LEVEL=INFO
```

### Configuration Override

```python
import os

# Environment-based configuration
config = {
    'env': os.getenv('BTQUANT_ENV', 'development'),
    'debug': os.getenv('BTQUANT_DEBUG', 'false').lower() == 'true',
    'log_level': os.getenv('BTQUANT_LOG_LEVEL', 'INFO'),
    'database_url': os.getenv('DATABASE_URL'),
    'redis_url': os.getenv('REDIS_URL'),
}

# Conditional configuration
if config['env'] == 'production':
    config.update({
        'commission': 0.0005,  # Lower commission in production
        'max_workers': 16,     # More workers for production
    })
else:
    config.update({
        'commission': 0.001,   # Higher commission for testing
        'max_workers': 4,      # Fewer workers for development
    })
```

### (UNSUPPORTED) Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Environment variables
ENV BTQUANT_ENV=production
ENV BTQUANT_DEBUG=false
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run application
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  btquant:
    build: .
    environment:
      - BTQUANT_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/btquant
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=btquant
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password

  redis:
    image: redis:6-alpine
```

This comprehensive configuration guide covers all aspects of BTQuant setup and tuning. Proper configuration is essential for optimal performance and reliability.
