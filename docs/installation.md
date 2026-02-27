# BTQuant Installation Guide

## Overview

BTQuant is a high-frequency algorithmic trading framework combining Python's flexibility with C++'s performance. It features real-time market manipulation detection, ultra-low latency shared memory data pipelines, and institutional-grade backtesting capabilities. This guide will help you set up the complete BTQuant system including Python components, C++ detectors, and market data collectors.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, Arch Linux, or equivalent)
- **Python**: 3.12 or 3.13
- **C++ Compiler**: GCC 7+ or Clang 5+ (C++17 support required)
- **Build System**: CMake 3.15+
- **Memory**: 8GB RAM (16GB recommended for optimal performance)
- **Storage**: 20GB free disk space (more if using SQL Server)
- **Network**: Stable internet connection for data feeds

### Recommended Requirements
- **Operating System**: Linux with kernel 5.14+
- **Python**: 3.13
- **C++ Compiler**: GCC 9+ or Clang 10+ with full C++17 support
- **Build System**: CMake 3.20+
- **Memory**: 16GB RAM or more
- **Storage**: 50GB+ SSD storage
- **Network**: High-speed, low-latency connection (for live trading)

## Prerequisites

### System Dependencies

Install the required system packages based on your Linux distribution:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev unixodbc-dev git cmake ninja-build
```

#### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y python3-devel unixODBC-devel git cmake ninja-build

# Fedora
sudo dnf groupinstall -y 'Development Tools'
sudo dnf install -y python3-devel unixODBC-devel git cmake ninja-build
```

#### Arch Linux
```bash
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm base-devel pybind11 unixodbc git tk cmake ninja
```

### Optional: SQL Server Setup

For full functionality with SQL Server data storage:

#### Microsoft SQL Server (Recommended)
1. Install SQL Server 2019 or later
2. Install SQL Server Management Studio (SSMS)
3. Create database with appropriate permissions

#### Alternative: SQL Server Express
- Free version suitable for development and small-scale deployments

## Installation Methods

### Method 1: Automated Installation (Recommended)

The easiest way to install BTQuant is using the provided installer script:

```bash
# Clone the repository
git clone --recurse-submodules https://github.com/ItsXactlY/BTQuant BTQuant
cd BTQuant

# Run the automated installer
bash Installers/install.sh
```

The installer will:
- Detect your Linux distribution
- Install system dependencies (Python and C++)
- Create a Python virtual environment (`.btq`)
- Install all required Python packages
- Build C++ components (market data collectors and detectors)
- Set up the Backtrader fork with extensions
- Configure shared memory segments for HotSpine

### Method 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone --recurse-submodules https://github.com/ItsXactlY/BTQuant BTQuant
cd BTQuant
```

#### Step 2: Create Virtual Environment
```bash
python3 -m venv .btq
source .btq/bin/activate
```

#### Step 3: Install Python Dependencies
```bash
# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install main dependencies
cd dependencies
pip install .

# Install QuantStats fork
cd quantstats_lumi_btquant
pip install .

# Install MSSQL extension
cd ../MsSQL
pip install .
```

#### Step 4: Build C++ Components
```bash
# Build market data collector
cd ../../dependencies/ccapi/example
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Build manipulation detectors
cd ../../../tests/new
chmod +x BUILD_AND_RUN.sh
./BUILD_AND_RUN.sh release
```

#### Step 5: Verify Installation
```bash
# Test Python components
python3 -c "import backtrader as bt; print(f'Backtrader version: {bt.__version__}')"

# Test C++ components
cd tests/new/build
./manipulation_monitor --help
```

## Configuration

### Basic Configuration

After installation, you need to configure BTQuant for your environment:

#### 1. Set Up Secrets and Credentials

Edit the `dependencies/backtrader/dontcommit.py` file:

```python
# JackRabbit Relay
identify = "your_jrr_identify_string"
jrr_webhook_url = "http://127.0.0.1:80"
jrr_order_history = "/path/to/jrr/history/"

# Web3 Configuration
bsc_privaccount1 = "your_private_key"
bsc_privaccountaddress = "your_wallet_address"

# Solana Configuration
solana_privkey_base58 = "your_solana_private_key"
solana_wallet_address = "your_solana_address"

# Discord Webhook
discord_webhook_url = "https://discord.com/api/webhooks/..."

# Telegram Configuration
telegram_api_id = 1234567
telegram_api_hash = "your_telegram_api_hash"
telegram_session_file = ".base.session"
telegram_channel = -1001234567890

# SQL Server Configuration
server = 'localhost'
candle_database = 'BinanceData'
optuna_database = 'OptunaBT'
username = 'SA'
password = 'YourStrong!Passw0rd'
driver = '{ODBC Driver 18 for SQL Server}'
```

#### 2. Database Configuration

For SQL Server setup:

```python
# Connection strings
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
```

### Exchange Configuration

#### CCXT Exchanges
For CCXT-compatible exchanges (Binance, Bybit, etc.):

```python
ccxt_config = {
    'apiKey': 'your_api_key',
    'secret': 'your_api_secret',
    'enableRateLimit': True,
    'rateLimit': 20,
    'options': {
        'defaultType': 'spot'  # or 'future', 'margin'
    }
}
```

#### Native Exchange Integration
For native WebSocket feeds (Binance, Bitget, MEXC):

```python
# Exchange-specific configuration
exchange_config = {
    'binance': {
        'api_key': 'your_api_key',
        'secret_key': 'your_secret_key',
        'testnet': False
    },
    'bitget': {
        'api_key': 'your_api_key',
        'secret_key': 'your_secret_key',
        'passphrase': 'your_passphrase'
    }
}
```

## Testing Your Installation

### Python Components Test

```python
from backtrader import backtest
from backtrader.strategies.Vumanchu_A import VuManchCipher_A
from backtrader.utils.ccxt_data import get_crypto_data

# Test with sample data
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-08', '15m', 'binance')

if __name__ == '__main__':
    result = backtest(
        strategy=VuManchCipher_A,
        data=data,
        init_cash=1000,
        backtest=True,
        plot=False,
        quantstats=False,
        asset_name='BTC/USDT'
    )
    print(f"Backtest completed. Final value: ${result:.2f}")
```

### C++ Components Test

```bash
# Test market data collector
cd dependencies/ccapi/example/build/src/market_data_collector
./market_data_collector --test

# Test manipulation detectors
cd ../../../../tests/new/build
./simple_monitor

# Should see output like:
# 🚨 StopHunt(symbol=BTC-USDT, exchange=binance, deviation=-1.2%, signal=LONG)
# 💰 Arbitrage(buy=kraken@42150, sell=binance@42250, profit=65bps)
```

### Database Connection Test

```python
from backtrader.feeds.mssql_crypto import get_database_data

# Test database connection
try:
    df = get_database_data(
        ticker='BTC',
        start_date='2024-01-01',
        end_date='2024-01-02',
        time_resolution='1h',
        pair='USDT'
    )
    print(f"Database test successful. Retrieved {len(df)} rows.")
except Exception as e:
    print(f"Database test failed: {e}")
```

## Troubleshooting

### Common Installation Issues

#### 1. Permission Denied Errors
```bash
# Fix file permissions
chmod +x Installers/install.sh
chmod +x dependencies/setup.py
chmod +x tests/new/BUILD_AND_RUN.sh
```

#### 2. Missing Dependencies
```bash
# Reinstall system dependencies
sudo apt-get install -y build-essential python3-dev unixodbc-dev cmake ninja-build

# Reinstall Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### 3. C++ Compilation Errors
```bash
# Check compiler version
g++ --version

# Update CMake
sudo apt-get install cmake

# Clean and rebuild
cd tests/new
rm -rf build
./BUILD_AND_RUN.sh release
```

#### 4. Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf .btq
python3 -m venv .btq
source .btq/bin/activate
pip install --upgrade pip setuptools wheel
```

#### 4. SQL Server Connection Issues
```bash
# Test ODBC connection
isql -v your_dsn your_username your_password

# Check SQL Server status
sudo systemctl status mssql-server
```

### Performance Optimization

#### 1. Memory Management
```python
# Increase Python memory allocation
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=1
```

#### 2. Database Optimization
- Enable SQL Server query optimization
- Create appropriate indexes on OHLCV tables
- Configure connection pooling

#### 3. Network Optimization
- Use low-latency network connections
- Configure firewall for WebSocket traffic
- Use VPN if required for exchange access

### Getting Help

If you encounter issues:

1. **Check the logs**: BTQuant provides detailed logging for both Python and C++ components
2. **Review system requirements**: Ensure your system meets minimum requirements for C++ compilation
3. **Consult documentation**: Check the troubleshooting and FAQ sections
4. **Community support**: Join the BTQuant community for assistance

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: Learn how to create your first strategy and run backtests
2. **Set up Market Data**: Configure and start the C++ market data collectors
3. **Run Detection Monitors**: Start real-time manipulation detection
4. **Explore Examples**: Check the `Examples/` directory for working code
5. **Configure Data Sources**: Set up your preferred data feeds and exchanges
6. **Start Strategy Development**: Begin building your trading strategies
7. **Launch Dashboard**: Use the QuantStats dashboard for performance analysis

## Uninstallation

To completely remove BTQuant:

```bash
# Remove virtual environment
rm -rf .btq

# Remove repository
cd ..
rm -rf BTQuant

# Optional: Remove system packages (if no longer needed)
sudo apt-get remove -y build-essential python3-dev unixodbc-dev
```

## Support

For additional support and questions:
- Check the [FAQ section](faq.md)
- Review [troubleshooting guide](troubleshooting.md)
- Explore [strategy development](user-guide/strategies.md)
- Visit the BTQuant community forums