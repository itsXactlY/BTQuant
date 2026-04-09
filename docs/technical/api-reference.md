# BTQuant API Reference

## Live Trading Functions (`livetrading.py`)

### livetrade_ccxt

```python
def livetrade_ccxt(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy_class: str,
    config: Optional[Dict[str, Any]] = None,
) -> None
```

Live trade using CCXT exchange directly.

**Parameters:**
- `coin`: Coin identifier (e.g., "BTC")
- `collateral`: Collateral currency (e.g., "USDT")
- `exchange`: Exchange name (e.g., "binance")
- `account`: Account name for config lookup
- `asset`: Full asset symbol (e.g., "BTC/USDT")
- `strategy_class`: Strategy class or name
- `config`: Optional CCXT config dict (overrides file-based config)

### livetrade_web3

```python
def livetrade_web3(
    coin: str,
    collateral: str,
    web3ws: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    timezone: str = 'Europe/Berlin',
    start_hours_ago: int = 2,
    enable_alerts: bool = False,
) -> None
```

Live trade on PancakeSwap via Web3.

**Parameters:**
- `coin`: Token contract address
- `collateral`: Collateral contract address
- `web3ws`: WebSocket URL for Web3
- `exchange`: Exchange name (e.g., "pancakeswap")
- `account`: Account type (e.g., "web3")
- `asset`: Asset pair (e.g., "$CAT/wBNB")
- `strategy`: Strategy class name
- `timezone`: Timezone for data (default: "Europe/Berlin")
- `start_hours_ago`: Hours of historical data for warmup
- `enable_alerts`: Enable Telegram/Discord alerts

### livetrade_hotspine

```python
def livetrade_hotspine(
    symbol_id: int,
    strategy_class,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001,
    **strategy_params
) -> None
```

Live trade using HotSpine shared memory.

**Parameters:**
- `symbol_id`: Symbol ID to filter trades
- `strategy_class`: Strategy class or instance
- `shm_name`: Shared memory segment name
- `batch_mode`: Enable batch reading
- `poll_interval`: Polling interval in seconds
- `**strategy_params`: Additional strategy parameters

### livetrade_hotspine_multi_symbol

```python
def livetrade_hotspine_multi_symbol(
    symbol_ids: list,
    strategy,
    shm_name: str = "/btquant_hotspine",
    batch_mode: bool = False,
    poll_interval: float = 0.0001,
    **strategy_params
) -> None
```

Live trade multiple symbols using HotSpine.

**Parameters:**
- `symbol_ids`: List of symbol IDs to trade
- `strategy`: Strategy class
- Other parameters same as `livetrade_hotspine`

### livetrade_binance

```python
def livetrade_binance(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = "",
) -> None
```

Live trade on Binance using BinanceStore.

**Parameters:**
- `coin`: Coin symbol (e.g., "BTC")
- `collateral`: Collateral currency (e.g., "USDT")
- `exchange`: Exchange name ("binance")
- `account`: Account identifier
- `asset`: Asset pair (e.g., "BTC/USDT")
- `strategy`: Strategy class name
- `start_hours_ago`: Historical data hours for warmup
- `enable_alerts`: Enable alert system
- `alert_channel`: Alert channel ID

### livetrade_mexc

```python
def livetrade_mexc(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = ""
) -> None
```

Live trade on MEXC exchange.

### livetrade_bitget

```python
def livetrade_bitget(
    coin: str,
    collateral: str,
    exchange: str,
    account: str,
    asset: str,
    strategy: str = "",
    start_hours_ago: int = 1,
    enable_alerts: bool = False,
    alert_channel: str = ""
) -> None
```

Live trade on Bitget exchange.

## Broker APIs

### JrrBroker (`brokers/jrrbroker.py`)

JackRabbitRelay webhook-based broker.

```python
class JrrBroker(bt.BrokerBase):
    params = (
        ('cash', 10000.0),
        ('exchange', 'mimic'),
        ('account', 'default'),
        ('debug', True),
    )
    
    def __init__(self, alert_manager: Optional[AlertManager] = None):
    
    def getcash(self) -> float:
    def get_cash(self) -> float:
    def getvalue(self, datas=None) -> float:
    def get_value(self, datas=None) -> float:
    
    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None, **kwargs) -> Order:
    
    def sell(self, owner, data, size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None, **kwargs) -> Order:
    
    def getposition(self, data) -> Position:
```

#### JrrOrderBase

```python
class JrrOrderBase:
    def __init__(self, alert_manager: Optional[AlertManager] = None):
    
    def send_jrr_buy_request(self, 
                            exchange: str, 
                            account: str, 
                            asset: str, 
                            amount: float) -> str:
    
    def send_jrr_close_request(self, 
                              exchange: str, 
                              account: str, 
                              asset: str) -> str:
```

**JRR Payload Format:**
```python
# Buy request
{
    "Exchange": "mimic",
    "Market": "spot",
    "Account": "default",
    "Action": "Buy",
    "Asset": "BTC/USDT",
    "USD": "100.0",
    "Identity": "your-identity"
}

# Close request
{
    "Exchange": "mimic",
    "Market": "spot",
    "Account": "default",
    "Action": "Close",
    "Asset": "BTC/USDT",
    "Identity": "your-identity"
}
```

### CCXTBroker (`brokers/ccxtbroker.py`)

Direct CCXT exchange broker.

```python
class CCXTBroker(BrokerBase):
    order_types = {
        Order.Market: 'market',
        Order.Limit: 'limit',
        Order.Stop: 'stop',
        Order.StopLimit: 'stop limit'
    }
    
    def __init__(self, exchange, currency, config, retries=5):
    
    def getcash(self) -> float:
    def getvalue(self, datas=None) -> float:
    def get_cash(self) -> float:
    def get_value(self, datas=None, mkt=False, lever=False) -> float:
    
    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None, **kwargs) -> Order:
    
    def sell(self, owner, data, size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None, **kwargs) -> Order:
    
    def cancel(self, order) -> bool:
    def get_orders_open(self, safe=False) -> List[Order]:
    def get_order_status(self, order) -> str:
    def check_orders(self) -> None:
    
    def set_initial_position(self, data, size) -> Order:
    def load_initial_positions(self, data) -> List[Order]:
    def get_position_info(self, data) -> Dict:
    def fetch_trades_history(self, symbol, since=None, limit=100) -> List:
    def find_first_buy_after_last_sell(self, symbol, limit=100) -> Optional:
```

#### SimpleOrder

```python
class SimpleOrder:
    def __init__(self, symbol, side, size, price, order_id=None):
    
    # Fields
    symbol: str
    side: str              # "buy" or "sell"
    size: float
    price: float
    id: str
    status: str            # "closed"
    executed: float
    executed_price: float
    executed_value: float
    created: datetime
    ccxt_order: dict
```

#### SimpleTrade

```python
class SimpleTrade(SimpleOrder):
    def __init__(self, symbol, side, size, price, order_id=None, timestamp=None):
    
    @classmethod
    def from_exchange_trade(cls, trade) -> SimpleTrade:
```

## Alert System APIs

### TelegramService (`brokers/jrrbroker.py`)

```python
class TelegramService(MessagingService):
    def __init__(self, api_id: int, api_hash: str, 
                 session_file: str, channel_id: int):
    
    async def initialize(self, loop=None):
    async def send_message(self, message: str) -> None:
    async def disconnect(self):
```

### DiscordService

```python
class DiscordService(MessagingService):
    def __init__(self, webhook_url: str):
    
    async def send_message(self, message: str) -> None:
```

### AlertManager

```python
class AlertManager:
    def __init__(self, messaging_services: list[MessagingService], loop=None):
    
    def send_alert(self, message: str) -> None:
```

## Configuration APIs

### load_ccxt_config

```python
def load_ccxt_config(
    exchange: str,
    account: str = "main",
    *,
    default_type: str = "spot",
    require_keys: bool = True,
) -> Dict[str, Any]:
```

Load CCXT configuration for exchange.

### HotSpineConfig

```python
@dataclass
class HotSpineConfig:
    shm_name: str = "/btquant_hotspine"
    shm_capacity: int = 1000000
    poll_interval: float = 0.0001
    batch_mode: bool = False
    max_batch_size: int = 1000
    enable_sql_storage: bool = True
    sql_batch_size: int = 100
    sql_queue_size: int = 10000
    enable_monitoring: bool = True
    metrics_interval: float = 1.0
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    symbol_mapping: Optional[Dict[int, Dict[str, str]]] = None
    market_type_filter: str = "all"
    symbol_whitelist: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HotSpineConfig:
```

### HotSpine Config Functions

```python
def get_default_config() -> HotSpineConfig:
def load_config_from_file(file_path: str) -> HotSpineConfig:
def save_config_to_file(config: HotSpineConfig, file_path: str) -> bool:
def configure_logging(level: int = logging.INFO,
                     format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     log_file: Optional[str] = None):
```

### MSSQLConfig

```python
@dataclass
class MSSQLConfig:
    server: str = "localhost"
    database: str = "BTQ_MarketData"
    username: str = "SA"
    password: str = ""
    driver: str = "{ODBC Driver 18 for SQL Server}"
    trust_server_certificate: bool = True
    
    def get_connection_string(self) -> str:
```

## Utility Functions

### backtest

```python
def backtest(
    strategy_class,
    coin: str,
    collateral: str = "USDT",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "15m",
    init_cash: float = 1000,
    commission: float = 0.00075,
    plot: bool = False,
    quantstats: bool = False,
    debug: bool = False,
    exchange: Optional[str] = None,
    slippage_bps: float = 5.0,
    params: Optional[Dict] = None,
) -> Dict:
```

### bulk_backtest

```python
def bulk_backtest(
    strategy_class,
    coins: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1h",
    collateral: str = "USDT",
    init_cash: float = 1000,
    max_workers: int = 8,
    save_results: bool = False,
    output_file: Optional[str] = None,
    params: Optional[Dict] = None,
    commission: float = 0.00075,
) -> Dict:
```

### optimize

```python
@dataclass
class OptimizationConfig:
    strategy_class: type
    coin: str
    interval: str = "1h"
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    collateral: str = "USDT"
    init_cash: float = 1000
    commission: float = 0.00075
    exchange: Optional[str] = None
    n_trials: int = 200
    n_jobs: int = 1
    study_name: Optional[str] = None
    pruner: Optional[str] = None
    seed: int = 42
    min_trades: int = 30
    plot_best: bool = False
    quantstats_best: bool = False

def optimize(config: OptimizationConfig, param_space_fn=None) -> Any:
```

## Symbol Mapper API

```python
class SymbolMapper:
    DEFAULT_MAPPINGS: Dict[int, Tuple[str, str]]
    
    def __init__(self, mapping_file: Optional[str] = None):
    def load_from_file(self, filepath: str):
    def save_to_file(self, filepath: str):
    def get_symbol_info(self, symbol_id: int) -> Tuple[str, str]:
    def get_symbol_id(self, exchange: str, symbol: str) -> int:
    def register_symbol(self, exchange: str, symbol: str, 
                        symbol_id: Optional[int] = None) -> int:
```
