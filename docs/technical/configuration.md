# BTQuant Configuration Reference

## Overview

BTQuant uses multiple configuration systems depending on the component. This document covers all configuration options and their formats.

## 1. CCXT Exchange Configuration (`ccxt_config.py`)

### Function Signature

```python
def load_ccxt_config(
    exchange: str,
    account: str = "main",
    *,
    default_type: str = "spot",
    require_keys: bool = True,
) -> Dict[str, Any]:
```

### Configuration Search Order

1. `<venv>/ccxt/{exchange}_{account}.json`
2. `<venv>/ccxt/{exchange}.json`
3. Environment variables (override/fill)

### File-Based Configuration

Create JSON files in your virtual environment's `ccxt/` directory:

```
.btq/
└── ccxt/
    ├── binance_main.json
    ├── binance_trading.json
    ├── bybit_main.json
    └── okx_main.json
```

**Example `binance_main.json`:**
```json
{
    "apiKey": "your-api-key-here",
    "secret": "your-secret-here",
    "enableRateLimit": true,
    "options": {
        "adjustForTimeDifference": true,
        "defaultType": "spot"
    }
}
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `BTQ_{EXCHANGE}_{ACCOUNT}_API_KEY` | API key (account-specific) |
| `BTQ_{EXCHANGE}_{ACCOUNT}_API_SECRET` | API secret (account-specific) |
| `BTQ_{EXCHANGE}_{ACCOUNT}_PASSWORD` | Password (for some exchanges) |
| `BTQ_{EXCHANGE}_{ACCOUNT}_UID` | User ID (for some exchanges) |
| `BTQ_{EXCHANGE}_{ACCOUNT}_DEFAULT_TYPE` | Default market type |
| `BTQ_{EXCHANGE}_API_KEY` | API key (exchange-level fallback) |
| `BTQ_{EXCHANGE}_API_SECRET` | API secret (exchange-level fallback) |
| `BTQ_{EXCHANGE}_PASSWORD` | Password (exchange-level fallback) |
| `BTQ_{EXCHANGE}_UID` | UID (exchange-level fallback) |

**Example:**
```bash
export BTQ_BINANCE_MAIN_API_KEY="your-api-key"
export BTQ_BINANCE_MAIN_API_SECRET="your-secret"
```

### Output Format

The function returns a dict suitable for `ccxt.exchange(config)`:

```python
{
    "apiKey": "...",
    "secret": "...",
    "enableRateLimit": True,
    "options": {
        "adjustForTimeDifference": True,
        "defaultType": "spot"  # or "future", "margin"
    }
}
```

## 2. Secrets Configuration (`dontcommit.py`)

This file contains sensitive credentials. **Never commit to version control.**

```python
# JackRabbitRelay
identify = ""                          # JRR identity string
jrr_webhook_url = "http://127.0.0.1:80"  # JRR webhook endpoint
jrr_order_history = "/home/JackrabbitRelay2/Data/Mimic/"

# Web3 (BSC)
bsc_privaccount1 = ""                  # Private key
bsc_privaccountaddress = ""            # Wallet address

# Solana
solana_privkey_base58 = ""             # Base58 private key
solana_wallet_address = ""             # Wallet address

# Discord
discord_webhook_url = ''               # Discord webhook URL

# Telegram
telegram_api_id = 1111111              # Telegram API ID (integer)
telegram_api_hash = ""                 # Telegram API hash
telegram_session_file = ".base.session"  # Session file path
telegram_channel = -100                # Channel ID (negative for groups)

# SQL Server
server = 'localhost'                   # SQL Server host
candle_database = 'BinanceData'        # Historical data database
optuna_database = 'OptunaBT'           # Optimization database
username = 'SA'                        # SQL username
password = 'YourStrong!Passw0rd'       # SQL password
driver = '{ODBC Driver 18 for SQL Server}'

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

## 3. HotSpine Configuration (`hotspine/config.py`)

### HotSpineConfig Dataclass

```python
@dataclass
class HotSpineConfig:
    # Shared memory configuration
    shm_name: str = "/btquant_hotspine"
    shm_capacity: int = 1000000
    
    # Reader configuration
    poll_interval: float = 0.0001
    batch_mode: bool = False
    max_batch_size: int = 1000
    
    # SQL integration
    enable_sql_storage: bool = True
    sql_batch_size: int = 100
    sql_queue_size: int = 10000
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_interval: float = 1.0
    
    # Error handling
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    
    # Symbol management
    symbol_mapping: Optional[Dict[int, Dict[str, str]]] = None
    market_type_filter: str = "all"
    symbol_whitelist: Optional[List[str]] = None
```

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HOTSPINE_SHM_NAME` | str | `/btquant_hotspine` | Shared memory segment name |
| `HOTSPINE_SHM_CAPACITY` | int | `1000000` | Buffer capacity |
| `HOTSPINE_POLL_INTERVAL` | float | `0.0001` | Poll interval (seconds) |
| `HOTSPINE_BATCH_MODE` | bool | `false` | Enable batch mode |
| `HOTSPINE_MAX_BATCH_SIZE` | int | `1000` | Max batch size |
| `HOTSPINE_ENABLE_SQL_STORAGE` | bool | `true` | Enable SQL storage |
| `HOTSPINE_SQL_BATCH_SIZE` | int | `100` | SQL batch size |
| `HOTSPINE_SQL_QUEUE_SIZE` | int | `10000` | SQL queue size |
| `HOTSPINE_ENABLE_MONITORING` | bool | `true` | Enable monitoring |
| `HOTSPINE_METRICS_INTERVAL` | float | `1.0` | Metrics interval (seconds) |
| `HOTSPINE_MAX_RECONNECT_ATTEMPTS` | int | `5` | Max reconnect attempts |
| `HOTSPINE_RECONNECT_DELAY` | float | `1.0` | Reconnect delay (seconds) |
| `HOTSPINE_SYMBOL_MAPPING` | JSON | `null` | Symbol ID mapping |
| `HOTSPINE_MARKET_TYPE_FILTER` | str | `all` | Market type filter |
| `HOTSPINE_SYMBOL_WHITELIST` | JSON | `null` | Symbol whitelist |

### Symbol Mapping Format

```json
{
    "1": {"symbol": "BTC-USDT", "market_type": "spot"},
    "2": {"symbol": "ETH-USDT", "market_type": "spot"},
    "101": {"symbol": "BTC-USDT", "market_type": "futures"}
}
```

### Configuration File Operations

```python
from backtrader.hotspine.config import (
    HotSpineConfig,
    load_config_from_file,
    save_config_to_file,
    get_default_config,
    configure_logging
)

# Get default config
config = get_default_config()

# Load from JSON file
config = load_config_from_file("/path/to/hotspine_config.json")

# Save to JSON file
save_config_to_file(config, "/path/to/hotspine_config.json")

# Configure logging
configure_logging(level=logging.DEBUG, log_file="/var/log/hotspine.log")
```

## 4. BigBrainCentral MSSQL Configuration

### MSSQLConfig (`bigbraincentral/storage_mssql.py`)

```python
@dataclass
class MSSQLConfig:
    server: str = "localhost"
    database: str = "BTQ_MarketData"
    username: str = "SA"
    password: str = ""
    driver: str = "{ODBC Driver 18 for SQL Server}"
    trust_server_certificate: bool = True
```

### MSSQLFeedConfig (`feeds/db_ohlcv_mssql.py`)

```python
@dataclass
class MSSQLFeedConfig:
    server: str = "localhost"
    database: str = "BTQ_MarketData"
    username: str = "SA"
    password: str = ""
    driver: str = "{ODBC Driver 18 for SQL Server}"
    trust_server_certificate: bool = True
```

## 5. CLI Configuration (btq)

The `btq` command accepts configuration via command-line arguments:

```bash
btq backtest \
    --coin BTC \
    --strategy SuperTrend_Scalp \
    --interval 15m \
    --start 2024-01-01 \
    --end 2025-01-01 \
    --cash 10000 \
    --commission 0.001 \
    --leverage 5 \
    --take-profit 2.0 \
    --stop-loss 1.0 \
    --plot \
    --quantstats
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--collateral` | `USDT` | Collateral currency |
| `--interval` | `15m` | Timeframe |
| `--end` | `2025-01-01` | End date |
| `--cash` | `1000` | Initial capital |
| `--commission` | `0.00075` | Commission rate (0.075%) |
| `--leverage` | `1` | Leverage multiplier |
| `--slippage` | `5.0` | Slippage (basis points) |
| `--workers` | `8` | Parallel workers (bulk) |
| `--trials` | `200` | Optimization trials |
| `--min-trades` | `30` | Minimum trades |
| `--pruner` | `hyperband` | Optuna pruner |
| `--seed` | `42` | Random seed |

## 6. Trading Configuration (`config/trading_config.py`)

Trading-specific settings can be found in `dependencies/backtrader/config/trading_config.py`.

## Configuration Files Summary

| File | Location | Purpose | Git Safe? |
|------|----------|---------|-----------|
| `ccxt/*.json` | `<venv>/ccxt/` | Exchange API keys | No |
| `dontcommit.py` | `dependencies/backtrader/` | All secrets | No |
| `hotspine/config.py` | `dependencies/backtrader/` | HotSpine defaults | Yes |
| `ccxt_config.py` | `dependencies/backtrader/` | Config loader | Yes |
| `storage_mssql.py` | `dependencies/backtrader/` | SQL defaults | Yes |

## Best Practices

1. **Never commit secrets**: Use `.gitignore` for `dontcommit.py` and `ccxt/*.json`
2. **Use environment variables** for sensitive data in production
3. **Use JSON config files** for development (easier to manage)
4. **Set `require_keys=False`** when testing without real API keys
5. **Use separate accounts** for paper trading vs live trading
