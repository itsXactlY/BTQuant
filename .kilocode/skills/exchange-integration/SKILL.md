---
name: exchange-integration
description: Exchange API integration and live trading setup for BTQuant
---

# BTQuant Exchange Integration

When integrating with crypto exchanges for live trading:

## Supported Exchanges

BTQuant supports 400+ exchanges via CCXT including:
- **Binance**: Spot and futures
- **OKX**: Advanced features
- **Bybit**: Perpetual futures
- **Mexc**: Lower fees
- **Bitget**: Copy trading
- **Kraken**: Fiat pairs

## CCXT Configuration

```python
from btquant import ExchangeConnector

connector = ExchangeConnector(
    exchange='binance',
    api_key='your_api_key',
    api_secret='your_api_secret',
    testnet=True  # Use for development
)

balance = connector.fetch_balance()
print(f"USDT Balance: {balance['USDT']['free']}")
```

## Credential Management

### Secure Storage

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
```

### Testnet Trading

```python
connector = ExchangeConnector(
    exchange='binance',
    api_key=testnet_key,
    api_secret=testnet_secret,
    testnet=True
)

test_order = connector.create_limit_buy_order(
    symbol='BTC/USDT',
    amount=0.1,
    price=45000
)
```

## Order Management

### Market Orders

```python
order = connector.create_market_buy_order(
    symbol='BTC/USDT',
    amount=0.1
)

print(f"Filled: {order['filled']}")
print(f"Cost: {order['cost']}")
```

### Limit Orders

```python
order = connector.create_limit_buy_order(
    symbol='BTC/USDT',
    amount=0.1,
    price=45000
)

status = connector.fetch_order(order['id'], 'BTC/USDT')
print(f"Status: {status['status']}")
```

## Position Management

### Fetching Account State

```python
balance = connector.fetch_balance()
for asset in ['BTC', 'ETH', 'USDT']:
    free = balance[asset]['free']
    used = balance[asset]['used']
    total = balance[asset]['total']
```

### Websocket Streams

```python
from btquant import WebsocketStream

stream = WebsocketStream(
    exchange='binance',
    symbols=['BTC/USDT', 'ETH/USDT'],
    data_types=['klines', 'trade']
)

@stream.on_kline
def handle_kline(symbol, candle):
    print(f"{symbol}: {candle['close']}")

stream.start()
```

## Error Handling

```python
from ccxt import NetworkError, ExchangeNotAvailable, RateLimitExceeded

try:
    order = connector.create_market_buy_order('BTC/USDT', 0.1)
except NetworkError:
    print("Network error - will retry")
except RateLimitExceeded:
    print("Rate limited - waiting...")
```

## Rate Limiting

```python
from btquant import RateLimiter

limiter = RateLimiter(calls=1200, period=60)

for symbol in symbols:
    limiter.wait_if_needed()
    candles = connector.fetch_ohlcv(symbol)
```
