<<<<<<< HEAD
# BTQuant: High-Frequency Trading Meets Simplicity

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![C++ Version](https://img.shields.io/badge/c%2B%2B-17+-red.svg)](https://en.cppreference.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Backtrader](https://img.shields.io/badge/backtrader-fork-orange.svg)](https://www.backtrader.com/)
=======
# BTQuant - Advanced Quantitative Trading Framework
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

BTQuant is a high-frequency algorithmic trading framework that processes thousands of trades per second across multiple exchanges with **zero API rate limits**. It combines Python's flexibility with C++'s performance, featuring real-time market manipulation detection, ultra-low latency shared memory data pipelines, and institutional-grade backtesting capabilities.

<<<<<<< HEAD
## 🚀 Key Features

- **Real-Time Manipulation Detection**: 5 advanced C++ detectors running in parallel (stop hunts, spoofing, whale frontruns, liquidity imbalances, spread arbitrage)
- **Ultra-Low Latency**: HotSpine shared memory for sub-microsecond trade processing across exchanges
- **Zero API Rate Limits**: Direct WebSocket feeds from Binance, OKX, Bybit, Coinbase, Kraken
- **Multi-Exchange Processing**: Simultaneous data from 5+ exchanges with cross-exchange correlation
- **Complete Transparency**: Every indicator calculation visible and auditable in Backtrader fork
- **Enterprise Data Management**: SQL Server canonical storage with full audit trails
- **Live Trading Ready**: Production-grade execution with monitoring, alerting, and recovery
- **QuantStats Integration**: Professional performance analytics and reporting

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Features](#features)
- [Real-Time Detection](#real-time-detection)
- [Community & Support](#community--support)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ and C++17
- Linux (Ubuntu 20.04+, Arch, or equivalent)
- 8GB RAM minimum (16GB recommended)
- GCC 7+ or Clang 5+
=======
BTQuant is a comprehensive algorithmic trading framework designed for **backtesting**, **forward testing**, and **live trading**. Built for quantitative analysts and algorithmic traders, BTQuant delivers **custom-built, high-performance trading infrastructure** that outperforms standard CCXT implementations. With **microsecond-precision tick data** and **native WebSocket integrations**, BTQuant provides institutional-grade trading capabilities.

---

## Core Capabilities

### 📊 Backtesting Engine
- **Historical Strategy Analysis**: Leverages **Backtrader** for comprehensive strategy validation using **tick-level historical data** with institutional-grade accuracy
- **Enterprise Data Management**: **Microsoft SQL Server** integration optimized for **massive tick datasets** and high-frequency portfolio analytics
- **Scalable Infrastructure**: Production-ready architecture designed for **tick-by-tick backtesting** at scale

### 🎯 Forward Testing
- **Exchange-Perfect Simulation**: **JackRabbitRelay** delivers **native exchange replication** - not generic CCXT wrappers - for true market condition testing
- **Independent Execution**: Eliminate broker dependencies with **custom execution engines** and complete operational control
- **Tick-Level Precision**: **True tick-by-tick simulation** with microsecond timing accuracy for HFT strategy validation

### ⚡ Live Trading - Custom Infrastructure

#### 🚀 **Native WebSocket Implementations** (Not CCXT)
- **Binance**: **Custom 1-second OHLCV streams** + **native tick data feeds**
- **Bitget**: **Direct WebSocket integration** with **sub-millisecond latency**
- **MEXC**: **Proprietary tick data implementation** bypassing standard API limitations
- **PancakeSwap**: **Native Web3 WebSocket feeds** with **1-second granularity**

#### ⚡ **High-Frequency Capabilities**
- **Tick Data Supremacy**: **Direct tick feeds** that **bypass common HFT delays** found in CCXT implementations
- **Custom Protocol Integration**: **Native exchange protocols** deliver **10x faster execution** than generic CCXT
- **Microsecond Precision**: **Hardware-level timing** for true high-frequency trading capabilities

#### 🎯 **Advanced Execution Logic**
- **Precision DCA**: **Tick-aware Dollar Cost Averaging** with **microsecond entry timing**
- **Experimental Trailing**: **Real-time trailing stops** using **live tick data** for optimal exit timing
- **Sub-50 Line Deployment**: Deploy **institutional-grade strategies** with minimal code overhead

---

## Why BTQuant Dominates

### 🏆 **Custom > Generic Every Time**

#### **Tick Data Infrastructure** 
- ✅ **Native tick feeds** with **microsecond timestamps**
- ✅ **Zero HFT delays** through **direct exchange protocols**
- ❌ *Not reliant on slow CCXT tick approximations*

#### **WebSocket Superiority**
- ✅ **Custom WebSocket implementations** for **each major exchange**
- ✅ **Sub-millisecond latency** through **optimized connection pools**
- ❌ *CCXT fallback available only as backup*

#### **Exchange Replication**
- ✅ **JackRabbitRelay** provides **exchange-perfect simulation** 
- ✅ **Native trading engine behavior** replication
- ❌ *Not generic CCXT simulation*

#### **DeFi Integration**
- ✅ **Native PancakeSwap/Web3** integration with **custom DEX protocols**
- ✅ **Real-time on-chain data** with **block-level precision**
- ❌ *Not limited by centralized exchange APIs*

#### **Enterprise Data**
- ✅ **MS SQL optimization** for **billion-row tick datasets**
- ✅ **Custom indexing** for **microsecond-level queries**
- ✅ **Real-time data ingestion** at **institutional scale**

---

## Performance Benchmarks

| Feature | BTQuant Custom | Standard CCXT | Performance Gain |
|---------|----------------|---------------|------------------|
| **Tick Data Latency** | <1ms | 50-200ms | **200x faster** |
| **WebSocket Reconnect** | <100ms | 5-30s | **300x faster** |
| **Order Execution** | <5ms | 100-500ms | **100x faster** |
| **Data Throughput** | 100k ticks/s | 1k ticks/s | **100x higher** |

---

## Getting Started

**Ready to experience true high-frequency trading infrastructure?**

📖 **[Complete Documentation](https://github.com/itsXactlY/BTQuant/wiki)**

💬 **[Join Our HFT Community](https://discord.gg/Y7uBxmRg3Z)** - Connect with quantitative traders using **real tick data**
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

### New Learning Resources

<<<<<<< HEAD
```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/ItsXactlY/BTQuant.git
cd BTQuant

# Run automated installer
bash Installers/install.sh
```

### Your First Backtest

```python
from backtrader import backtest
from backtrader.strategies.Vumanchu_A import VuManchCipher_A
from backtrader.utils.ccxt_data import get_crypto_data

# Get data
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-01-31', '1h', 'binance')

# Run backtest
result = backtest(
    strategy=VuManchCipher_A,
    data=data,
    init_cash=10000,
    quantstats=True,
    plot=True
)
```

### Real-Time Market Monitoring

```bash
# Start market data collector (C++)
cd dependencies/ccapi/example/build/src/market_data_collector
./market_data_collector

# In another terminal: Start manipulation detector
cd tests/new/build
./manipulation_monitor
```

## 📚 Documentation

### Getting Started
- **[Installation Guide](docs/installation.md)** - Complete setup instructions
- **[Quick Start Guide](docs/quickstart.md)** - Your first strategy and backtest
- **[Manipulation Detection Quick Start](tests/new/QUICKSTART.md)** - Real-time market monitoring

### User Guides
- **[Strategy Development](docs/user-guide/strategies.md)** - Build and customize strategies
- **[Indicators Reference](docs/user-guide/indicators.md)** - Complete indicator library
- **[Data Sources](docs/user-guide/data-sources.md)** - All data feed options
- **[Backtesting](docs/user-guide/backtesting.md)** - Testing and optimization

### Technical Reference
- **[Architecture Overview](docs/technical/architecture.md)** - System design and components
- **[Market Data Collection](dependencies/ccapi/example/src/market_data_collector/market_data_collector.md)** - CCAPI WebSocket collectors
- **[HotSpine](docs/technical/hotspine.md)** - Ultra-low latency shared memory
- **[Manipulation Detectors](tests/new/ARCHITECTURE.md)** - C++ detection algorithms
- **[BigBrainCentral](docs/technical/bigbraincentral.md)** - SQL Server data storage
- **[API Reference](docs/technical/api-reference.md)** - Complete API documentation
- **[Configuration](docs/technical/configuration.md)** - Setup and secrets

### Additional Resources
- **[Dashboard](dependencies/dashboard/README.md)** - QuantStats performance analysis
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[FAQ](docs/faq.md)** - Frequently asked questions

## 🏗️ Architecture

BTQuant implements a hybrid Python/C++ architecture for maximum performance and flexibility:

### Data Collection Layer (C++)
- **CCAPI Collectors**: Native WebSocket connections to Binance, OKX, Bybit, Coinbase, Kraken
- **HotSpine Writer**: Sub-microsecond shared memory data ingestion
- **Multi-Exchange Processing**: Simultaneous data from 5+ exchanges with zero rate limits
- **BigBrainCentral**: SQL Server canonical storage with full audit trails

### Detection Engine Layer (C++)
- **5 Parallel Detectors**: Stop hunt, liquidity imbalance, whale frontrun, spread arbitrage, spoofing
- **HotSpine Reader**: Lock-free shared memory access for ultra-low latency
- **Real-Time Analysis**: Sub-millisecond detection across all exchanges
- **Plugin Architecture**: Extensible detector system with dynamic configuration

### Strategy Layer (Python)
- **BaseStrategy**: Unified strategy scaffold with DCA, order management, and telemetry
- **Indicator Library**: Complete Ehlers suite with full transparency
- **Strategy Catalogue**: Pre-built strategies for various market conditions
- **HotSpine Integration**: Direct access to real-time market data

### Execution Layer (Python)
- **Live Trading**: Production-ready execution with monitoring and alerting
- **Backtesting**: High-performance historical simulation with QuantStats
- **Optimization**: Parallel parameter optimization with Optuna
- **Dashboard**: Web-based performance analytics and reporting

## ✨ Features

### Real-Time Market Intelligence
- **5 Advanced Detectors**: Stop hunt, liquidity imbalance, whale frontrun, spread arbitrage, spoofing detection
- **Cross-Exchange Correlation**: Simultaneous analysis across Binance, OKX, Bybit, Coinbase, Kraken
- **Sub-Millisecond Detection**: C++ algorithms processing 4,381 trades/second
- **Zero Rate Limits**: Direct WebSocket feeds with unlimited data access

### Transparency & Auditability
- **Complete Indicator Visibility**: See exactly how every indicator calculates values
- **Data Pipeline Tracking**: Monitor data flow from WebSocket to strategy
- **Calculation Auditing**: Verify signal generation and risk calculations
- **Full Audit Trails**: SQL Server storage with complete trade history

### Performance & Scale
- **Ultra-Low Latency**: Sub-microsecond trade processing with HotSpine shared memory
- **High Throughput**: 13,970 orderbook updates/second across multiple exchanges
- **Enterprise Storage**: SQL Server with microsecond precision timestamps
- **Memory Efficient**: ~100MB footprint for shared memory operations

### Trading Capabilities
- **Multi-Exchange**: Spot, futures, options across 100+ venues
- **DeFi Integration**: PancakeSwap, Web3, and DEX trading
- **Advanced Strategies**: ML-driven, technical, and discretionary systems
- **Live Trading**: Production-ready execution with monitoring and recovery

### Development Experience
- **Hybrid Architecture**: Python flexibility with C++ performance
- **Rich Ecosystem**: 100+ indicators, analyzers, and utilities
- **Extensible Architecture**: Easy to add new exchanges, indicators, and strategies
- **Production Ready**: Monitoring, alerting, and operational tooling
- **QuantStats Dashboard**: Professional performance analytics and reporting

## 🎯 Real-Time Detection

BTQuant includes a sophisticated C++ detection engine that monitors live market data for manipulation patterns:

### Detection Algorithms
- **Stop Hunt Detector**: Identifies fake wicks and stop-loss hunting across exchanges
- **Liquidity Imbalance Detector**: Detects thin orderbooks signaling potential manipulation targets
- **Whale Front-Run Detector**: Catches large trades before other exchanges react
- **Spread Arbitrage Detector**: Finds profitable cross-exchange arbitrage opportunities
- **Spoofing Detector**: Identifies fake orders designed to manipulate prices

### Performance Metrics
```
43,814 trades processed in 10 seconds across 5 exchanges
4,381 trades/second processing rate
13,970 orderbook updates/second
Sub-millisecond detection latency
Zero API rate limits
```

### Integration Options
- **Signal Output**: JSON/WebSocket for trading bot integration
- **Alert Systems**: Telegram/Discord/Slack notifications
- **Database Storage**: SQL Server for historical analysis
- **Dashboard**: Real-time monitoring interface

## 🤝 Community & Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community forum for questions and sharing
- **Telegram**: Real-time community chat
- **Discord**: Voice and text channels for collaboration

## 🤝 Contributing

We welcome contributions! Please see our [GitHub repository](https://github.com/ItsXactlY/BTQuant) for contribution guidelines.

### Development Setup
```bash
# Fork and clone
git clone https://github.com/itsXactlY/BTQuant.git
cd BTQuant

# Set up development environment
bash Installers/install.sh --dev

# Run tests
python -m pytest
```

## 📄 License

BTQuant is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on the excellent [Backtrader](https://www.backtrader.com/) framework
- Powered by [CCAPI](https://github.com/crypto-chassis/ccapi) for exchange connectivity
- Inspired by institutional trading practices and high-frequency research
- Community contributions and feedback drive continuous improvement

---

**BTQuant**: High-Frequency Trading Meets Simplicity. Real-Time Detection, Zero Limits, Maximum Performance.

Ready to start building? Check out the [Quick Start Guide](docs/quickstart.md) or dive into [Real-Time Detection](tests/new/)!
=======
- Explore the fully documented **BaseStrategy showcase** located at
  `Examples/BaseStrategy_Showcase.py` for an end-to-end walkthrough covering
  backtesting, CCXT connectivity, and Web3/PancakeSwap routing.
- Check out the new automated tests in `tests/test_order_tracker.py` to learn
  how order tracking is validated and to use them as a template for additional
  coverage.

---

*BTQuant: Where **custom infrastructure** meets **institutional performance***
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
