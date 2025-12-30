# BTQuant Configuration Guide (WORK IN PROGRESS)

## Overview

This guide covers all configuration options available in BTQuant, including exchange settings, database connections, Hotspine integration, and advanced configuration parameters.

## Configuration Files

### Main Configuration File

The primary configuration file is located at:
```
dependencies/backtrader/dontcommit.py
```

**⚠️ Important**: This file is excluded from version control (`.gitignore`) as it contains sensitive information like API keys and database credentials.

### Configuration Structure

```python
# Exchange and Trading Configuration
exchange = "binance"           # Exchange name
account = "your_account_id"    # Account identifier for JackRabbitRelay
asset = "BTC/USDT"            # Trading pair
amount = 0.001                # Fixed amount to trade
coin = "BTC"                  # Base currency
collateral = "USDT"           # Quote currency

# Database Configuration
server = 'localhost'
candle_database = 'BinanceData'
optuna_database = 'OptunaBT'
username = 'SA'
password = 'YourStrong!Passw0rd'
driver = '{ODBC Driver 18 for SQL Server}'

# Connection Strings
connection_string = (f'DRIVER={driver};'
                     f'SERVER={server};'
                     f'DATABASE={candle_database};'
                     f'UID={username};'
                     f'PWD={password};'
                     f'TrustServerCertificate=yes;')

optuna_connection_string = (f'DRIVER={driver};'
                           f'SERVER={server};'
                           f'DATABASE={optuna_database};'
                           f'UID={username};'
                           f'PWD={password};'
                           f'TrustServerCertificate=yes;')

# Web3 Configuration
bsc_privaccount1 = "your_private_key"
bsc_privaccountaddress = "your_wallet_address"
solana_privkey_base58 = "your_solana_private_key"
solana_wallet_address = "your_solana_address"

# Communication Configuration
discord_webhook_url = 'https://discord.com/api/webhooks/...'
telegram_api_id = 1234567
telegram_api_hash = "your_telegram_api_hash"
telegram_session_file = ".base.session"
telegram_channel = -1001234567890

# Alert Configuration
enable_alerts = True
alert_channel = "your_alert_channel_id"
```

## Exchange Configuration

### CCXT-Compatible Exchanges

BTQuant supports all CCXT-compatible exchanges. Configuration examples:

#### Binance
```python
ccxt_config = {
    'apiKey': 'your_api_key',
    'secret': 'your_api_secret',
    'enableRateLimit': True,
    'rateLimit': 20,
    'options': {
        'defaultType': 'spot',  # 'spot', 'future', 'margin'
        'adjustForTimeDifference': True,
    },
    'sandbox': False,  # Set to True for testnet
}
```

#### Bybit
```python
ccxt_config = {
    'apiKey': 'your_api_key',
    'secret': 'your_api_secret',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',  # 'spot', 'future', 'option'
        'adjustForTimeDifference': True,
    },
    'sandbox': True,  # Testnet
}
```

#### Kraken
```python
ccxt_config = {
    'apiKey': 'your_api_key',
    'secret': 'your_api_secret',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    },
}
```

### Native Exchange Integration

For exchanges with native WebSocket support:

#### Binance Native
```python
binance_config = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'testnet': False,
    'sandbox': False,
    'options': {
        'adjustForTimeDifference': True,
        'recvWindow': 5000,
    }
}
```

#### Bitget Native
```python
bitget_config = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'passphrase': 'your_passphrase',
    'testnet': False,
    'options': {
        'defaultType': 'spot',
    }
}
```

#### MEXC Native
```python
mexc_config = {
    'api_key': 'your_api_key',
    'secret_key': 'your_secret_key',
    'testnet': False,
    'options': {
        'defaultType': 'spot',
    }
}
```

### PancakeSwap (Web3)

```python
pancakeswap_config = {
    'coin': 'BNB',           # Base currency
    'collateral': 'USDT',    # Quote currency
    'private_key': 'your_private_key',
    'rpc_url': 'https://bsc-dataseed.binance.org/',
    'contract_address': '0x10ED43C718714eb63d5aA57B78B54704E256024E',  # PancakeSwap Router
}
```

## Database Configuration

### SQL Server Setup

#### Installation Requirements
```bash
# Ubuntu/Debian
sudo apt-get install -y unixodbc-dev
sudo apt-get install -y msodbcsql17 mssql-tools

# CentOS/RHEL
sudo yum install -y unixODBC-devel
sudo yum install -y msodbcsql17 mssql-tools

# Arch Linux
sudo pacman -S --noconfirm unixodbc
```

#### Connection String Parameters

```python
# Basic connection
connection_string = (
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER=localhost;'
    f'DATABASE=BinanceData;'
    f'UID=SA;'
    f'PWD=YourStrong!Passw0rd;'
    f'TrustServerCertificate=yes;'
)

# With connection pooling
connection_string = (
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER=localhost;'
    f'DATABASE=BinanceData;'
    f'UID=SA;'
    f'PWD=YourStrong!Passw0rd;'
    f'TrustServerCertificate=yes;'
    f'Pooling=yes;'
    f'Min Pool Size=5;'
    f'Max Pool Size=100;'
)

# With encryption
connection_string = (
    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
    f'SERVER=localhost;'
    f'DATABASE=BinanceData;'
    f'UID=SA;'
    f'PWD=YourStrong!Passw0rd;'
    f'Encrypt=yes;'
    f'TrustServerCertificate=no;'
    f'TrustServerCAListFile=/path/to/ca.pem;'
)
```

#### Database Schema

BTQuant automatically creates the following tables:

```sql
-- OHLCV data tables (one per symbol)
CREATE TABLE [dbo].[BTCUSDT_klines] (
    [TimestampStart] DATETIME2(6) NOT NULL,
    [Open] FLOAT NOT NULL,
    [High] FLOAT NOT NULL,
    [Low] FLOAT NOT NULL,
    [Close] FLOAT NOT NULL,
    [Volume] FLOAT NOT NULL,
    PRIMARY KEY ([TimestampStart])
);

-- Trades table
CREATE TABLE [dbo].[trades] (
    [id] BIGINT IDENTITY(1,1) PRIMARY KEY,
    [exchange] NVARCHAR(50) NOT NULL,
    [symbol] NVARCHAR(50) NOT NULL,
    [price] FLOAT NOT NULL,
    [size] FLOAT NOT NULL,
    [side] NVARCHAR(10) NOT NULL,
    [is_buyer_maker] BIT NOT NULL,
    [timestamp] DATETIME2(6) NOT NULL,
    [created_at] DATETIME2(6) DEFAULT GETUTCDATE()
);

-- Orderbook snapshots
CREATE TABLE [dbo].[orderbook_snapshots] (
    [id] BIGINT IDENTITY(1,1) PRIMARY KEY,
    [exchange] NVARCHAR(50) NOT NULL,
    [symbol] NVARCHAR(50) NOT NULL,
    [timestamp] DATETIME2(6) NOT NULL,
    [bids] NVARCHAR(MAX) NOT NULL,  -- JSON array of [price, size]
    [asks] NVARCHAR(MAX) NOT NULL,  -- JSON array of [price, size]
    [checksum] NVARCHAR(64),        -- Optional checksum
    [created_at] DATETIME2(6) DEFAULT GETUTCDATE()
);

-- Optuna optimization results
CREATE TABLE [dbo].[optuna_trials] (
    [trial_id] INT PRIMARY KEY,
    [study_name] NVARCHAR(255) NOT NULL,
    [params] NVARCHAR(MAX),         -- JSON object
    [value] FLOAT,
    [state] NVARCHAR(50),
    [datetime_start] DATETIME2(6),
    [datetime_complete] DATETIME2(6)
);
```

### Alternative Databases

#### PostgreSQL
```python
import psycopg2

connection_string = (
    f"postgresql://username:password@localhost:5432/database_name"
)

# In dontcommit.py
def get_postgres_connection():
    return psycopg2.connect(
        host="localhost",
        database="btquant_data",
        user="username",
        password="password"
    )
```

#### MySQL
```python
import mysql.connector

connection_string = (
    f"mysql://username:password@localhost:3306/database_name"
)

# In dontcommit.py
def get_mysql_connection():
    return mysql.connector.connect(
        host='localhost',
        database='btquant_data',
        user='username',
        password='password'
    )
```

## Hotspine Configuration

### Overview

Hotspine is BTQuant's high-performance market data ingestion system built with C++ and ccapi. It provides:

- Real-time market data collection
- Microsecond-precision timestamping
- High-throughput data storage
- Multi-exchange support

### Hotspine Setup

#### Prerequisites
```bash
# Install required dependencies
sudo apt-get install -y build-essential cmake libboost-all-dev libssl-dev
sudo apt-get install -y libcurl4-openssl-dev libjsoncpp-dev
```

#### Configuration

```python
# Hotspine configuration in dontcommit.py
hotspine_config = {
    'enabled': True,
    'exchanges': ['binance', 'okx', 'bitget'],
    'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    'timeframes': ['1s', '1m', '5m', '1h'],
    'market_types': ['spot', 'future'],
    
    # Database settings
    'database': {
        'server': 'localhost',
        'database': 'HotspineData',
        'username': 'SA',
        'password': 'YourStrong!Passw0rd',
        'driver': '{ODBC Driver 18 for SQL Server}',
    },
    
    # Performance settings
    'performance': {
        'batch_size': 1000,
        'flush_interval': 5,  # seconds
        'max_connections': 10,
        'buffer_size': 10000,
    },
    
    # Exchange-specific settings
    'exchange_settings': {
        'binance': {
            'api_key': 'your_api_key',
            'secret_key': 'your_secret_key',
            'testnet': False,
        },
        'okx': {
            'api_key': 'your_api_key',
            'secret_key': 'your_secret_key',
            'passphrase': 'your_passphrase',
            'testnet': False,
        },
    }
}
```

#### Hotspine Data Collection

```python
# Start Hotspine data collection
from dependencies.ccapi.example.src.market_data_collector.market_data_collector import MarketDataCollector

def start_hotspine():
    collector = MarketDataCollector(hotspine_config)
    collector.start()
    
    # Monitor collection
    try:
        while True:
            stats = collector.get_stats()
            print(f"Collected: {stats['trades']} trades, {stats['orderbooks']} orderbooks")
            time.sleep(60)
    except KeyboardInterrupt:
        collector.stop()
```

### Hotspine Data Access

```python
# Access Hotspine data in strategies
from backtrader.feeds.hotspine_feed import HotspineData

def get_hotspine_data(symbol, timeframe, start_date, end_date):
    return HotspineData(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        config=hotspine_config
    )

# Usage in strategy
class HotspineStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        
        # Get high-frequency data
        self.data_1s = get_hotspine_data('BTC/USDT', '1s', '2024-01-01', '2024-01-02')
        self.data_1m = get_hotspine_data('BTC/USDT', '1m', '2024-01-01', '2024-01-02')
        
        # Add to cerebro
        self.cerebro.adddata(self.data_1s)
        self.cerebro.adddata(self.data_1m)
```

## Strategy Configuration

### BaseStrategy Parameters

```python
class MyStrategy(BaseStrategy):
    params = (
        # Basic parameters
        ('init_cash', 100000.0),
        ('exchange', 'binance'),
        ('asset', 'BTC/USDT'),
        ('amount', None),
        ('coin', 'BTC'),
        ('collateral', 'USDT'),
        
        # Trading parameters
        ('percent_sizer', 0.1),      # Use 10% of capital per trade
        ('take_profit', 2.0),        # 2% take profit
        ('stop_loss', 1.0),          # 1% stop loss
        ('order_cooldown', 60),      # 60 seconds between orders
        
        # DCA parameters
        ('DCA', False),              # Enable DCA
        ('dca_levels', [2.0, 5.0]),  # DCA at 2%, 5% drops
        ('dca_amounts', [1.0, 2.0]), # DCA multipliers
        
        # Risk management
        ('max_position_size', 0.5),  # Max 50% of portfolio
        ('risk_per_trade', 0.01),    # 1% risk per trade
        
        # Advanced features
        ('debug', False),
        ('capture_data', False),     # Enable indicator transparency
        ('backtest', True),
        ('bulk', False),
        ('optuna', False),
        
        # Alert configuration
        ('enable_alerts', True),
        ('alert_channel', None),
    )
```

### Custom Strategy Configuration

```python
class CustomStrategy(BaseStrategy):
    params = (
        # Technical indicator parameters
        ('sma_fast_period', 10),
        ('sma_slow_period', 30),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        
        # Machine learning parameters
        ('ml_model_path', '/path/to/model.pkl'),
        ('ml_features', ['rsi', 'macd', 'volume']),
        ('ml_threshold', 0.6),
        
        # Multi-timeframe parameters
        ('slow_timeframe', '1h'),
        ('fast_timeframe', '15m'),
        ('trend_threshold', 0.01),
        
        # Event-driven parameters
        ('event_threshold', 0.02),   # 2% price move
        ('event_timeout', 10),       # 10 bars to act
    )
```

## Environment Configuration

### Development Environment

```python
# Development configuration
development_config = {
    'mode': 'development',
    'debug': True,
    'log_level': 'DEBUG',
    'testnet': True,
    'sandbox': True,
    
    # Development database
    'database': {
        'server': 'localhost',
        'database': 'BTQuantDev',
        'username': 'dev_user',
        'password': 'dev_password',
    },
    
    # Development exchanges
    'exchanges': {
        'binance': {
            'api_key': 'dev_api_key',
            'secret': 'dev_secret',
            'testnet': True,
        }
    }
}
```

### Production Environment

```python
# Production configuration
production_config = {
    'mode': 'production',
    'debug': False,
    'log_level': 'INFO',
    'testnet': False,
    'sandbox': False,
    
    # Production database
    'database': {
        'server': 'prod-server.company.com',
        'database': 'BTQuantProd',
        'username': 'prod_user',
        'password': 'prod_password',
        'encrypt': True,
        'connection_timeout': 30,
        'command_timeout': 60,
    },
    
    # Production exchanges
    'exchanges': {
        'binance': {
            'api_key': 'prod_api_key',
            'secret': 'prod_secret',
            'testnet': False,
        }
    },
    
    # Production monitoring
    'monitoring': {
        'enabled': True,
        'metrics_interval': 60,
        'alert_thresholds': {
            'max_drawdown': 10.0,
            'max_trades_per_day': 100,
            'min_daily_return': -5.0,
        }
    }
}
```

### Environment Variables

```bash
# Environment variables for configuration
export BTQ_ENV=production
export BTQ_DEBUG=false
export BTQ_LOG_LEVEL=INFO
export BTQ_DATABASE_URL="postgresql://user:pass@host:port/db"
export BTQ_REDIS_URL="redis://localhost:6379/0"
export BTQ_EXCHANGE_API_KEY="your_api_key"
export BTQ_EXCHANGE_SECRET="your_secret"
```

## Advanced Configuration

### Custom Data Feeds

```python
# Custom data feed configuration
class CustomDataFeed(bt.feeds.DataBase):
    params = (
        ('data_path', '/path/to/data'),
        ('data_format', 'csv'),
        ('timezone', 'UTC'),
        ('compression', 1),
        ('timeframe', bt.TimeFrame.Minutes),
    )
    
    def __init__(self):
        super().__init__()
        self.data_path = self.p.data_path
        self.data_format = self.p.data_format
        
    def start(self):
        # Custom data loading logic
        pass
    
    def _load(self):
        # Custom data parsing logic
        pass

# Usage
data = CustomDataFeed(
    dataname='/path/to/your/data.csv',
    data_format='csv',
    compression=1,
    timeframe=bt.TimeFrame.Minutes
)
```

### Custom Brokers

```python
# Custom broker configuration
class CustomBroker(bt.brokers.BrokerBase):
    params = (
        ('commission', 0.001),
        ('slippage', 0.0005),
        ('margin', False),
        ('leverage', 1.0),
    )
    
    def __init__(self):
        super().__init__()
        self.commission = self.p.commission
        self.slippage = self.p.slippage
        
    def getcash(self):
        # Custom cash retrieval logic
        pass
    
    def getvalue(self, datas=None):
        # Custom portfolio value calculation
        pass
    
    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None, parent=None,
            transmit=True, **kwargs):
        # Custom buy order logic
        pass

# Usage
cerebro = bt.Cerebro()
cerebro.broker = CustomBroker()
```

### Performance Optimization

```python
# Performance optimization configuration
performance_config = {
    'optimization': {
        'enabled': True,
        'max_workers': 8,
        'chunk_size': 1000,
        'memory_limit': '8GB',
        'cache_size': '2GB',
    },
    
    'data_processing': {
        'parallel_loading': True,
        'data_compression': True,
        'index_optimization': True,
    },
    
    'strategy_execution': {
        'jit_compilation': True,
        'vectorization': True,
        'memory_pooling': True,
    }
}
```

## Configuration Validation

### Configuration Testing

```python
def validate_configuration(config):
    """Validate configuration settings"""
    errors = []
    
    # Check required fields
    required_fields = ['exchange', 'asset', 'database']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate database connection
    try:
        test_connection(config['database'])
    except Exception as e:
        errors.append(f"Database connection failed: {e}")
    
    # Validate exchange credentials
    for exchange, creds in config.get('exchanges', {}).items():
        if not validate_exchange_credentials(exchange, creds):
            errors.append(f"Invalid credentials for {exchange}")
    
    return errors

def test_configuration():
    """Test configuration with sample data"""
    from backtrader.utils.ccxt_data import get_crypto_data
    
    try:
        # Test data fetching
        data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02', '1h', 'binance')
        if data is None or data.is_empty():
            return False, "Failed to fetch test data"
        
        # Test strategy execution
        from backtrader.strategies.base import BaseStrategy
        result = backtest(BaseStrategy, data=data, init_cash=1000, backtest=True)
        
        return True, f"Configuration test successful. Result: ${result:.2f}"
    
    except Exception as e:
        return False, f"Configuration test failed: {e}"
```

### Configuration Examples

#### Complete Configuration Example

```python
# Complete configuration example
COMPLETE_CONFIG = {
    # Exchange configuration
    'exchange': 'binance',
    'ccxt_config': {
        'apiKey': 'your_api_key',
        'secret': 'your_secret',
        'enableRateLimit': True,
        'sandbox': False,
    },
    
    # Database configuration
    'database': {
        'server': 'localhost',
        'candle_database': 'BinanceData',
        'optuna_database': 'OptunaBT',
        'username': 'SA',
        'password': 'YourStrong!Passw0rd',
        'driver': '{ODBC Driver 18 for SQL Server}',
    },
    
    # Strategy configuration
    'strategy': {
        'class': 'MyStrategy',
        'parameters': {
            'sma_fast_period': 10,
            'sma_slow_period': 30,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
        },
        'risk_management': {
            'max_position_size': 0.5,
            'risk_per_trade': 0.01,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 5.0,
        },
    },
    
    # Hotspine configuration
    'hotspine': {
        'enabled': True,
        'exchanges': ['binance', 'okx'],
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'timeframes': ['1s', '1m', '5m'],
    },
    
    # Alert configuration
    'alerts': {
        'enabled': True,
        'discord_webhook': 'https://discord.com/api/webhooks/...',
        'telegram': {
            'api_id': 1234567,
            'api_hash': 'your_api_hash',
            'channel': -1001234567890,
        },
    },
    
    # Performance configuration
    'performance': {
        'max_workers': 8,
        'memory_limit': '8GB',
        'cache_size': '2GB',
        'parallel_loading': True,
    },
}
```

This comprehensive configuration guide covers all aspects of BTQuant configuration, from basic setup to advanced optimization. Always ensure your configuration is properly tested before deploying to production environments.