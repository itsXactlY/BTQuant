# BTQuant: Institutional-Grade Algorithmic Trading Framework

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Backtrader](https://img.shields.io/badge/backtrader-fork-orange.svg)](https://www.backtrader.com/)

BTQuant is an institutional-grade algorithmic trading framework that unifies historical research, forward simulation, and live execution around an extended Backtrader fork. It provides complete transparency in indicator calculations, ultra-low latency data pipelines, and enterprise-grade data management.

## 🚀 Key Features

- **Complete Transparency**: Every indicator calculation is visible and auditable
- **Ultra-Low Latency**: HotSpine shared memory for sub-microsecond trade processing
- **Enterprise Data Spine**: BigBrainCentral with SQL Server canonical storage
- **Multi-Exchange Support**: CCXT, native WebSockets, and DEX integrations
- **Institutional Indicators**: Full Ehlers indicator suite with advanced signal processing
- **Advanced Strategies**: Pre-built strategies with DCA, risk management, and optimization
- **Live Trading Ready**: Production-grade execution with monitoring and alerting

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Documentation](#documentation)
- [Architecture](#architecture)
- [Features](#features)
- [Community & Support](#community--support)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Linux (Ubuntu 20.04+, CentOS 8+, or equivalent)
- 8GB RAM minimum (16GB recommended)

### Installation

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

## 📚 Documentation

### Getting Started
- **[Installation Guide](docs/installation.md)** - Complete setup instructions
- **[Quick Start Guide](docs/quickstart.md)** - Your first strategy and backtest

### User Guides
- **[Strategy Development](docs/user-guide/strategies.md)** - Build and customize strategies
- **[Indicators Reference](docs/user-guide/indicators.md)** - Complete indicator library
- **[Data Sources](docs/user-guide/data-sources.md)** - All data feed options
- **[Backtesting](docs/user-guide/backtesting.md)** - Testing and optimization

### Technical Reference
- **[Architecture](docs/technical/architecture.md)** - System design and components
- **[BigBrainCentral](docs/technical/bigbraincentral.md)** - Data spine documentation
- **[HotSpine](docs/technical/hotspine.md)** - Low-latency integration
- **[API Reference](docs/technical/api-reference.md)** - Complete API documentation
- **[Configuration](docs/technical/configuration.md)** - Setup and secrets

### Additional Resources
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[FAQ](docs/faq.md)** - Frequently asked questions

## 🏗️ Architecture

BTQuant implements a layered architecture designed for institutional-grade trading:

### Data Layer
- **BigBrainCentral**: Always-on market data spine with SQL Server canonical storage
- **HotSpine**: Shared memory L1 cache for ultra-low latency live trading
- **Multi-Source Feeds**: CCXT, native WebSockets, SQL Server, CSV/Pandas

### Strategy Layer
- **BaseStrategy**: Unified strategy scaffold with DCA, order management, and telemetry
- **Indicator Library**: Complete Ehlers suite with full transparency
- **Strategy Catalogue**: Pre-built strategies for various market conditions

### Execution Layer
- **Live Trading**: Production-ready execution with monitoring
- **Backtesting**: High-performance historical simulation
- **Optimization**: Parallel parameter optimization with Optuna

### Transparency Layer
- **Indicator Transparency**: Every calculation visible and auditable
- **Data Pipeline Visibility**: Complete data flow tracking
- **Performance Monitoring**: Real-time metrics and debugging

## ✨ Features

### Transparency & Auditability
- **Complete Indicator Visibility**: See exactly how every indicator calculates values
- **Data Pipeline Tracking**: Monitor data flow from source to strategy
- **Calculation Auditing**: Verify signal generation and risk calculations

### Performance & Scale
- **Ultra-Low Latency**: Sub-microsecond trade processing with HotSpine
- **High Throughput**: Millions of trades/second processing capability
- **Enterprise Storage**: SQL Server with microsecond precision timestamps

### Trading Capabilities
- **Multi-Exchange**: Spot, futures, options across 100+ venues
- **DeFi Integration**: PancakeSwap, Web3, and DEX trading
- **Advanced Strategies**: ML-driven, technical, and discretionary systems

### Development Experience
- **Rich Ecosystem**: 100+ indicators, analyzers, and utilities
- **Extensible Architecture**: Easy to add new exchanges, indicators, and strategies
- **Production Ready**: Monitoring, alerting, and operational tooling

## 🤝 Community & Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community forum for questions and sharing
- **Telegram**: Real-time community chat
- **Discord**: Voice and text channels for collaboration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

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
- Inspired by institutional trading practices and academic research
- Community contributions and feedback drive continuous improvement

---

**BTQuant**: Where Every Calculation is Visible, Every Signal is Understandable, and Every Result is Trustworthy.

Ready to start building? Check out the [Quick Start Guide](docs/quickstart.md)!
