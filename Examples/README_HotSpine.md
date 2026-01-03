# HotSpine Live Trading Examples

This directory contains examples demonstrating how to use HotSpine for ultra-low latency live trading with BTQuant.

## Overview

HotSpine is BTQuant's shared memory infrastructure that provides sub-microsecond access to market data while maintaining complete separation between live trading and storage operations.

### Architecture
```
Exchange Data → HotSpine Writer → Shared Memory → HotSpine Reader → Trading Strategy
                      ↓
                 SQL Storage (Async)
```

## Examples

### Live_Trading_HotSpine_SMA.py

Complete example showing live trading with the SMA Cross strategy using HotSpine data feed.

**Features:**
- Ultra-low latency trade processing
- Real-time performance monitoring
- Live broker integration (optional)
- Comprehensive error handling
- Strategy status reporting

**Usage:**
```bash
python Live_Trading_HotSpine_SMA.py
```

**Configuration:**
- Symbol ID: 123 (BTC/USDT)
- Shared Memory: `/btquant_hotspine`
- Batch Mode: Disabled (single trade for lowest latency)
- Polling Interval: 100 microseconds

## Prerequisites

### 1. HotSpine Shared Memory
Ensure HotSpine writer is running and creating the shared memory segment:
```bash
# Check if shared memory exists
ls -la /dev/shm/ | grep hotspine

# Expected output:
/btquant_hotspine
```

### 2. HotSpine Library
The HotSpine reader library must be compiled and available:
```bash
# Check if library exists
ls -la hotspine/libhotspine_reader.so
```

### 3. Dependencies
```bash
pip install backtrader ccxt numpy pandas
```

## Configuration

### Exchange Setup (Optional)
For live trading, configure your exchange API keys in the example:

```python
exchange_config = {
    'exchange': 'mexc',
    'apiKey': 'your_api_key_here',
    'secret': 'your_secret_here',
    'enableRateLimit': True,
    'rateLimit': 20,
}
```

### Symbol Configuration
Configure the symbol ID for HotSpine filtering:
```python
symbol_id = 123  # BTC/USDT
```

## Running the Examples

### Basic Live Trading
```bash
cd Examples
python Live_Trading_HotSpine_SMA.py
```

### Simulation Mode
If no API keys are configured, the example runs in simulation mode with paper trading.

### Performance Monitoring
The example provides real-time statistics:
- Trades processed per second
- Current position
- Live price updates
- Performance metrics

## HotSpine Data Feed

The `HotSpineData` feed provides:

- **Single Trade Mode**: Lowest latency (0.16µs), best for HFT
- **Batch Mode**: Higher throughput (14M+ trades/sec), best for processing
- **Symbol Filtering**: Process only specific symbols
- **Live Data**: Real-time market data integration

### Feed Parameters
```python
data = HotSpineData(
    symbol_id=123,              # Symbol to filter
    shm_name="/btquant_hotspine", # Shared memory name
    batch_mode=False,           # Single trade mode
    poll_interval=0.0001        # 100µs polling
)
```

## Strategy Integration

### Extending Existing Strategies
```python
from backtrader.strategies.SMA_Cross_MESAdaptive_Prime import SMA_Cross_MESAdaptivePrime

class HotSpineSMALiveStrategy(SMA_Cross_MESAdaptive_Prime):
    def __init__(self):
        super().__init__()
        # Add HotSpine-specific monitoring
        self.hotspine_trade_count = 0

    def next(self):
        super().next()
        self.hotspine_trade_count += 1
        # Add real-time monitoring logic
```

### Custom HotSpine Strategies
```python
class MyHotSpineStrategy(bt.Strategy):
    def __init__(self):
        # Standard indicators
        self.sma = bt.ind.SMA(period=20)

    def next(self):
        # Access HotSpine trade data
        price = self.data.close[0]
        volume = self.data.volume[0]

        # Trading logic here
        if self.sma > price:
            self.buy()
        elif self.sma < price:
            self.sell()
```

## Performance Characteristics

### Single Trade Mode
- **Throughput**: 6,105,246 trades/second
- **Latency**: 0.16 microseconds
- **Use Case**: Ultra-low latency strategies

### Batch Mode
- **Throughput**: 14,493,103 trades/second
- **Latency**: 0.07 microseconds (amortized)
- **Use Case**: High-frequency data processing

## Troubleshooting

### HotSpine Connection Issues
```bash
# Check shared memory
ls -la /dev/shm/btquant_hotspine

# Check permissions
chmod 666 /dev/shm/btquant_hotspine

# Verify HotSpine writer is running
ps aux | grep hotspine
```

### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify Backtrader installation
python -c "import backtrader; print('Backtrader OK')"

# Check HotSpine feed import
python -c "from backtrader.feeds.hotspine_feed import HotSpineData; print('HotSpine OK')"
```

### Performance Issues
- Use single trade mode for lowest latency
- Pin processes to specific CPU cores
- Ensure NUMA-aware memory allocation
- Monitor buffer utilization

## Architecture Benefits

1. **Separation of Concerns**: Live trading independent of storage
2. **Ultra-Low Latency**: Sub-microsecond trade processing
3. **Scalability**: Handle millions of trades per second
4. **Reliability**: Lock-free concurrent access
5. **Memory Efficiency**: 32 bytes per trade

## Next Steps

- Explore batch mode for high-throughput strategies
- Implement custom indicators optimized for live data
- Add risk management and position sizing
- Integrate with additional data sources
- Implement strategy backtesting with historical HotSpine data

## Support

For issues with HotSpine integration:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Review the HotSpine technical documentation
4. Check the test suite for validation examples