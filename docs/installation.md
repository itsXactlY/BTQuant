# BTQuant Installation Guide

## Overview

BTQuant is an institutional-grade algorithmic trading framework built on Backtrader with support for multiple exchanges, SQL Server data storage, and advanced market data ingestion. This guide will help you set up BTQuant on your system.

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, Arch Linux, or equivalent)
- **Python**: 3.12 or 3.13
- **Memory**: 8GB RAM (16GB recommended for optimal performance)
- **Storage**: 20GB free disk space (more if using SQL Server)
- **Network**: Stable internet connection for data feeds

### Recommended Requirements
- **Operating System**: Linux with kernel 5.14+
- **Python**: 3.13
- **Memory**: 16GB RAM or more
- **Storage**: 50GB+ SSD storage
- **Network**: High-speed, low-latency connection

## Prerequisites

### System Dependencies

Install the required system packages based on your Linux distribution:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev unixodbc-dev git
```

#### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y python3-devel unixODBC-devel git

# Fedora
sudo dnf groupinstall -y 'Development Tools'
sudo dnf install -y python3-devel unixODBC-devel git
```

#### Arch Linux
```bash
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm base-devel pybind11 unixodbc git tk
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
- Install system dependencies
- Create a Python virtual environment (`.btq`)
- Install all required Python packages
- Set up the Backtrader fork with extensions

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

#### Step 3: Install Dependencies
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

#### Step 4: Verify Installation
```python
python3 -c "import backtrader as bt; print(f'Backtrader version: {bt.__version__}')"
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

### Basic Functionality Test

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
```

#### 2. Missing Dependencies
```bash
# Reinstall system dependencies
sudo apt-get install -y build-essential python3-dev unixodbc-dev

# Reinstall Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### 3. Virtual Environment Issues
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

1. **Check the logs**: BTQuant provides detailed logging for debugging
2. **Review system requirements**: Ensure your system meets minimum requirements
3. **Consult documentation**: Check the troubleshooting and FAQ sections
4. **Community support**: Join the BTQuant community for assistance

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: Learn how to create your first strategy
2. **Explore Examples**: Check the `Examples/` directory for working code
3. **Configure Data Sources**: Set up your preferred data feeds
4. **Start Strategy Development**: Begin building your trading strategies

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
- Explore [strategy development](strategy-development.md)
- Visit the BTQuant community forums