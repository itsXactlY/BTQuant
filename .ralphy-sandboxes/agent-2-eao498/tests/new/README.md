# BTQuant Market Manipulation Detector

Real-time cross-exchange market manipulation detection system for cryptocurrency trading.

## Features

🎯 **5 Advanced Detectors:**
- **Stop Hunt Detector**: Identifies fake wicks and stop hunts across exchanges
- **Liquidity Imbalance Detector**: Detects thin orderbooks and manipulation targets
- **Whale Front-Run Detector**: Catches large trades before other exchanges react
- **Spread Arbitrage Detector**: Finds profitable cross-exchange opportunities
- **Spoofing Detector**: Identifies fake orders and market maker manipulation

⚡ **Ultra-Low Latency:**
- Direct shared memory access via HotSpine
- Microsecond-level detection
- Lock-free data structures
- Parallel processing with configurable threads

🔄 **Multi-Exchange Support:**
- Binance, Coinbase, Kraken, OKX, Bybit
- Easy to add more exchanges
- Real-time cross-exchange correlation

## Requirements

- C++17 compatible compiler (GCC 7+ or Clang 5+)
- CMake 3.15+
- Linux (shared memory via `/dev/shm`)
- Running HotSpine market data collector

## Quick Start

### 1. Build
```bash
cd btquant_manipulation_detector
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 2. Configure Symbols

Edit `config/symbol_mapping.json`:
```json
{
  "exchanges": [
    {
      "name": "binance",
      "symbols": ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    },
    {
      "name": "coinbase",
      "symbols": ["BTC-USDT", "ETH-USDT"]
    }
  ]
}
```

### 3. Run
```bash
# Main monitor (all detectors)
./manipulation_monitor

# Simple monitor (trades only)
./simple_monitor

# Multi-exchange monitor (cross-exchange correlation)
./multi_exchange_monitor
```

## Usage Examples

### Basic Monitoring
```cpp
#include "hotspine_extended_reader.hpp"
#include "detectors/stop_hunt_detector.hpp"

int main() {
    // Initialize reader
    HotSpineExtendedReader reader("/dev/shm/btquant_hotspine");
    
    // Initialize detector
    StopHuntDetector detector(reader);
    detector.set_threshold(0.5); // 0.5% deviation threshold
    
    // Monitor loop
    while (true) {
        if (auto signal = detector.detect("BTC-USDT")) {
            std::cout << "🚨 Stop Hunt Detected!\n"
                      << "   Exchange: " << signal->hunt_exchange << "\n"
                      << "   Deviation: " << signal->hunt_deviation_pct << "%\n"
                      << "   Signal: " << (signal->is_long_signal ? "LONG" : "SHORT") << "\n";
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```

### Cross-Exchange Arbitrage
```cpp
SpreadArbitrageDetector arb_detector(reader);
arb_detector.set_min_profit(50); // 0.5% minimum profit

auto opportunities = arb_detector.detect_all("BTC-USDT");
for (const auto& opp : opportunities) {
    std::cout << "💰 Arbitrage: Buy " << opp.buy_exchange 
              << " @ " << opp.buy_price
              << ", Sell " << opp.sell_exchange
              << " @ " << opp.sell_price
              << " (Profit: " << opp.potential_profit_bps << " bps)\n";
}
```

## Configuration

### Detector Thresholds
```cpp
// Stop Hunt Detector
stop_hunt.set_threshold(0.5);  // 0.5% price deviation

// Liquidity Detector
liquidity.set_depth_ratio_threshold(3.0);  // 3x orderbook depth difference

// Whale Detector
whale.set_threshold_usd(100000);  // $100k minimum trade size

// Arbitrage Detector
arbitrage.set_min_profit_bps(50);  // 0.5% minimum profit after fees
arbitrage.set_fees(10, 20);  // Maker 0.1%, Taker 0.2%
```

## Architecture
```
┌─────────────────────────────────────────────────────┐
│          Market Data Collector (C++)                │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Binance  │  │ Coinbase │  │  Kraken  │         │
│  │ WebSocket│  │ WebSocket│  │ WebSocket│         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │             │             │                │
│       └─────────────┴─────────────┘                │
│                     │                              │
│              ┌──────▼───────┐                      │
│              │  HotSpine    │                      │
│              │  Writer      │                      │
│              └──────┬───────┘                      │
└─────────────────────┼──────────────────────────────┘
                      │
              ┌───────▼────────┐
              │  Shared Memory │
              │  /dev/shm      │
              └───────┬────────┘
                      │
┌─────────────────────┼──────────────────────────────┐
│                     │                              │
│              ┌──────▼───────┐                      │
│              │  HotSpine    │                      │
│              │  Extended    │                      │
│              │  Reader      │                      │
│              └──────┬───────┘                      │
│                     │                              │
│       ┌─────────────┼─────────────┐               │
│       │             │             │               │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐           │
│  │ Stop    │  │Liquidity│  │  Whale  │           │
│  │ Hunt    │  │Imbalance│  │Front-Run│           │
│  │Detector │  │Detector │  │Detector │           │
│  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │                 │
│       └────────────┴────────────┘                 │
│                    │                              │
│             ┌──────▼───────┐                      │
│             │   Trading    │                      │
│             │   Signals    │                      │
│             └──────────────┘                      │
│                                                   │
│      Manipulation Detection System                │
└───────────────────────────────────────────────────┘
```

## Performance

Typical latency measurements on Ryzen 9 5950X:

- Trade processing: ~5μs per trade
- Orderbook processing: ~15μs per snapshot
- Stop hunt detection: ~50μs per check
- Cross-exchange correlation: ~100μs for 5 exchanges
- Memory footprint: ~100MB (shared with collector)

## Troubleshooting

### "Shared memory not found"
```bash
# Check if collector is running
ls -lh /dev/shm/btquant_hotspine

# If not, start the market data collector first
```

### "Symbol ID not found"
```bash
# Update symbol_mapping.json with your symbols
# Symbol IDs are generated automatically from exchange+symbol hash
```

### High CPU usage
```bash
# Reduce polling frequency in detector loop
std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Instead of tight loop
```

## Contributing

This is a high-performance trading system. Pull requests welcome for:
- New detector algorithms
- Performance optimizations
- Additional exchange support
- Bug fixes

## License

MIT License - See LICENSE file

## Author

Built by aLca (@itsXactlY) for BTQuant

---

**⚠️ Trading Risk Warning**: This software is for informational purposes only. Cryptocurrency trading involves substantial risk. Always do your own research and never trade with money you can't afford to lose.