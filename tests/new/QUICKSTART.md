# BTQuant Manipulation Detection - Quick Start Guide

## Prerequisites

1. **Running Market Data Collector**
    - Ensure the C++ market data collector is running and feeding data to HotSpine
    - Shared memory segment `/dev/shm/btquant_hotspine` should be active

2. **System Requirements**
    - Linux (Ubuntu 20.04+, Arch, etc.)
    - GCC 7+ or Clang 5+ (C++17 support)
    - CMake 3.15+
    - 4GB+ RAM
    - Running HotSpine market data feed

## Installation

### 1. Clone/Copy the Project
```bash
cd ~/projects/BTQuant/dependencies/ccapi/example/
mkdir manipulation_detector
cd manipulation_detector
# Copy all files here
```

### 2. Build
```bash
chmod +x BUILD_AND_RUN.sh
./BUILD_AND_RUN.sh release
```

Or manually:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 3. Configure Detection

Edit `config/btquant.yaml` to configure detector settings:

```yaml
# BTQuant Detection Configuration
shared_memory:
  path: "/btquant_hotspine"
  buffer_size: 65536

monitoring:
  poll_interval_ms: 1000

exchanges:
  binance:
    enabled: true
    base_id: 1
  okx:
    enabled: true
    base_id: 301

detectors:
  stop_hunt:
    enabled: true
    threshold_pct: 0.5
  liquidity_imbalance:
    enabled: true
    depth_ratio_threshold: 3.0
  whale_frontrun:
    enabled: true
    threshold_usd: 100000
```

### 4. Run
```bash
cd build

# Full monitor with all detectors
./manipulation_monitor

# Simple stop hunt monitor
./simple_monitor

# Multi-exchange price comparison
./multi_exchange_monitor
```

## Basic Usage

### Monitor All Detectors
```bash
./manipulation_monitor
```

Output:
```
🚨 StopHunt(symbol=BTC-USDT, exchange=binance, deviation=-1.2%, signal=LONG)
💰 Arbitrage(buy=kraken@42150, sell=binance@42250, profit=65bps)
🐋 WhaleDetected(symbol=ETH-USDT, size=$250000, lagging=3 exchanges)
```

### Monitor Specific Symbol

Edit `src/main_monitor.cpp` and modify:
```cpp
std::vector symbols = {"BTC-USDT"};  // Only BTC
```

### Adjust Detection Thresholds
```cpp
// In your code or main_monitor.cpp
stop_hunt.set_threshold(0.3);           // 0.3% instead of 0.5%
liquidity.set_depth_ratio_threshold(5.0); // 5x instead of 3x
whale.set_threshold_usd(50'000);        // $50k instead of $100k
arbitrage.set_min_profit_bps(30);       // 0.3% instead of 0.5%
```

## Troubleshooting

### "Failed to attach to shared memory"
```bash
# Check if HotSpine is running
ls -lh /dev/shm/btquant_hotspine

# If not found, start your market data collector first
cd ~/projects/BTQuant/dependencies/ccapi/example/build/src/market_data_collector
./market_data_collector
```

### "Symbol ID not found"

Your `symbol_mapping.json` doesn't match the symbols your collector is using.

**Fix:**

1. Check what symbols your collector is monitoring
2. Update `config/symbol_mapping.json` accordingly
3. Make sure symbol IDs match

### High CPU Usage

Reduce polling frequency:
```cpp
// In main_monitor.cpp, increase sleep time
std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Instead of 100ms
```

### No Detections

Possible reasons:

1. Not enough exchanges (need 3+ for stop hunt detection)
2. Thresholds too high
3. Market too stable (no manipulation happening)
4. Symbol mapping incorrect

**Debug:**
```cpp
// Add debug output
auto prices = reader.get_all_exchange_prices("BTC-USDT");
for (const auto& [exchange, price] : prices) {
    std::cout << exchange << ": $" << price << "\n";
}
```

## Performance Tips

### Optimize for Your Hardware
```bash
# Use native CPU optimizations
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
make -j$(nproc)
```

### Monitor Specific Exchanges Only

Edit `SymbolRegistry` to only load exchanges you care about:
```json
{
  "exchanges": [
    {"name": "binance", "symbols": [...]},
    {"name": "coinbase", "symbols": [...]}
  ]
}
```

### Reduce History Size

In `hotspine_extended_reader.hpp`:
```cpp
size_t max_trade_history_ = 500;  // Instead of 1000
```

## Integration with Trading Bot

### Signal Output to File
```cpp
// In main_monitor.cpp
std::ofstream signal_file("signals.json", std::ios::app);

if (auto signal = stop_hunt.detect(symbol)) {
    nlohmann::json j;
    j["type"] = "stop_hunt";
    j["symbol"] = signal->symbol;
    j["exchange"] = signal->hunt_exchange;
    j["signal"] = signal->is_long_signal ? "LONG" : "SHORT";
    j["timestamp"] = signal->timestamp_us;
    
    signal_file << j.dump() << "\n";
    signal_file.flush();
}
```

### Signal Output to WebSocket
```cpp
#include <websocketpp/...>

void send_signal_to_trading_bot(const StopHuntSignal& signal) {
    // Your WebSocket code here
    ws_client.send(signal.to_json());
}
```

### Signal Output to Redis/Message Queue
```cpp
#include <hiredis/hiredis.h>

redisContext* redis = redisConnect("127.0.0.1", 6379);
redisCommand(redis, "PUBLISH signals %s", signal.to_json().c_str());
```

## Advanced Configuration

### Custom Detector Configuration File

Create `config/detector_settings.json`:
```json
{
  "stop_hunt": {
    "threshold_pct": 0.5,
    "min_exchanges": 3
  },
  "liquidity": {
    "depth_ratio_threshold": 3.0,
    "imbalance_threshold": 0.3
  },
  "whale": {
    "threshold_usd": 100000,
    "min_lagging_exchanges": 2,
    "reaction_threshold_bps": 5.0
  },
  "arbitrage": {
    "min_profit_bps": 50,
    "maker_fee_bps": 10,
    "taker_fee_bps": 20
  },
  "spoofing": {
    "min_cancel_count": 3,
    "max_duration_ms": 1000,
    "min_size": 10
  }
}
```

## What's Next?

1. **Add More Exchanges**: Edit symbol_mapping.json
2. **Fine-tune Thresholds**: Adjust based on your backtesting
3. **Add Alerting**: Integrate with Telegram/Discord/Slack
4. **Connect to Trading Bot**: Use signals for automated trading
5. **Add ML Layer**: Train models on detected patterns

---

**Happy Hunting! 🎯**