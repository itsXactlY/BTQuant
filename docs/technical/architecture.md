# System Architecture

This document provides a comprehensive overview of BTQuant's system architecture, designed for institutional-grade algorithmic trading with complete transparency and ultra-low latency capabilities.

## Table of Contents

- [Overview](#overview)
- [Data Layer](#data-layer)
- [Strategy Layer](#strategy-layer)
- [Execution Layer](#execution-layer)
- [Transparency Layer](#transparency-layer)
- [Infrastructure Components](#infrastructure-components)
- [Performance Characteristics](#performance-characteristics)
- [Scalability Considerations](#scalability-considerability)

## Overview

BTQuant implements a layered architecture that separates concerns while maintaining tight integration between components. The architecture is designed for:

- **Institutional-grade performance** with microsecond-level latencies
- **Complete transparency** in all calculations and data flows
- **Enterprise scalability** with SQL Server as the canonical data store
- **Multi-exchange support** across 100+ venues
- **Live trading capabilities** with production-grade reliability

## Data Layer

### BigBrainCentral Data Spine

BigBrainCentral serves as the institutional data backbone, providing:

#### Architecture
```
Exchange APIs → C++ Collectors → SQL Server → Python Adapters → Strategies
```

#### Components

**C++ Collectors (ccapi-based)**
- Ultra-low latency WebSocket connections
- Multi-exchange support (Binance, Bitget, MEXC, OKX, etc.)
- Real-time trade and orderbook aggregation
- Microsecond timestamp precision
- Direct ODBC bulk inserts to SQL Server

**SQL Server Canonical Store**
- Microsecond-precision datetime storage (`DATETIME2(6)`)
- Per-symbol OHLCV tables with indexes
- Trade and orderbook snapshot archives
- ACID-compliant transactions
- Optimized for time-series queries

**Python Adapters**
- `DatabaseOHLCVData`: Backtrader feed for historical data
- `ReadOnlyOHLCV`: SELECT-only access for analytics
- `MarketDataStorage`: JackRabbitRelay integration
- Automatic data validation and gap detection

#### Data Flow
1. **Ingestion**: C++ collectors stream live data to SQL Server
2. **Storage**: ACID transactions ensure data integrity
3. **Access**: Python adapters provide Backtrader-compatible feeds
4. **Analytics**: Direct SQL access for research and reporting

### HotSpine Live Cache

HotSpine provides L1 cache functionality for live trading:

#### Architecture
```
Exchange APIs → C++ Writers → Shared Memory → C++ Readers → Strategies
```

#### Components

**Shared Memory Layout**
- Fixed-size ring buffer (configurable, default 1M trades)
- Atomic operations for thread safety
- Memory-mapped I/O for zero-copy access
- Cross-process synchronization

**C++ Hot Path**
- Direct memory access without Python GIL
- Sub-microsecond trade processing
- Batch and single-trade modes
- Health monitoring and overflow detection

**Python Bindings**
- `HotSpineReader`: Low-level shared memory access
- `HotSpineRuntime`: Strategy execution environment
- `HotSpineData`: Backtrader feed integration

#### Performance Characteristics
- **Single Trade Mode**: 6M+ trades/second, <1µs latency
- **Batch Mode**: 14M+ trades/second, ~0.07µs per trade
- **Memory Footprint**: 32 bytes per trade
- **Thread Safety**: Concurrent read/write operations

### CCXT Historical Data

For backtesting and research:

#### Features
- 100+ exchange support
- Automatic retry logic with backoff
- Rate limit handling
- Data deduplication and validation
- Polars DataFrame output

#### Integration
```python
from backtrader.utils.ccxt_data import get_crypto_data

data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-12-31', '1h', 'binance')
```

## Strategy Layer

### BaseStrategy Framework

The `BaseStrategy` provides comprehensive trading infrastructure:

#### Core Features
- **Order Management**: Unified order creation and tracking
- **DCA Engine**: Configurable dollar-cost averaging
- **Position Tracking**: Real-time P&L and exposure monitoring
- **Risk Controls**: Stop-loss, take-profit, and position limits
- **Telemetry**: Built-in logging and performance metrics

#### Architecture
```
Strategy Logic → BaseStrategy → Order Management → Broker → Exchange
```

#### Key Methods
- `buy_or_short_condition()`: Entry signal logic
- `sell_or_cover_condition()`: Exit signal logic
- `create_order()`: DCA-aware order creation
- `check_stop_loss()`: Risk management
- `calculate_position_size()`: Risk-based sizing

### Indicator Library

Complete transparency-enabled indicator suite:

#### Transparency System
```python
from backtrader import transparencypatch

# Enable full transparency
patch = transparencypatch.TransparencyPatch()
patch.debug = True
patch.apply_indicator_patch()
```

#### Indicator Categories
- **Trend**: SMA, EMA, SuperTrend, Alligator, MAMA
- **Momentum**: RSI, MACD, Stochastic, CCI, RMI
- **Volatility**: ATR, Bollinger Bands, Chaikin Volatility
- **Volume**: Volume Oscillator, Chaikin Money Flow, Klinger
- **Oscillators**: QQE, Ultimate, Williams %R, Awesome
- **Advanced**: Ehlers suite, Vumanchu Cipher, Order Chain

#### Calculation Transparency
Every indicator exposes:
- Input data validation
- Step-by-step calculations
- Intermediate values
- Parameter impact analysis
- Performance metrics

## Execution Layer

### Live Trading Stack

Production-grade execution infrastructure:

#### Exchange Integration
- **Native APIs**: Direct WebSocket connections for ultra-low latency
- **CCXT**: Standardized API access for 100+ exchanges
- **Custom Brokers**: Venue-specific optimizations

#### Order Management
- **Order Routing**: Intelligent venue selection
- **Fill Tracking**: Real-time execution monitoring
- **Rejection Handling**: Automatic retry and fallback logic

#### Risk Management
- **Position Limits**: Configurable exposure controls
- **Circuit Breakers**: Automatic shutdown on extreme conditions
- **Compliance**: Audit trails and reporting

### Backtesting Engine

High-performance historical simulation:

#### Cerebro Integration
- Extended Backtrader Cerebro with custom analyzers
- Parallel optimization with Optuna
- QuantStats integration for professional reporting

#### Optimization Framework
```python
from backtrader.utils.backtest import optimize_backtest

results = optimize_backtest(
    strategy=MyStrategy,
    data=data,
    fast_period=[10, 15, 20],
    slow_period=[30, 40, 50],
    max_workers=8
)
```

#### Analysis Tools
- **QuantStats**: Professional performance reports
- **PyFolio**: Risk and return analysis
- **Custom Analyzers**: Domain-specific metrics

## Transparency Layer

### Indicator Transparency

Complete visibility into indicator calculations:

#### Real-time Monitoring
```python
def next(self):
    if self.p.capture_data:
        self.log(f"RSI[0]: {self.rsi[0]:.4f}")
        self.log(f"MACD Signal: {self.macd.signal[0]:.6f}")
        self.log(f"ATR[0]: {self.atr[0]:.6f}")
```

#### Calculation Chain Visibility
Every indicator shows:
- Raw input processing
- Mathematical transformations
- Parameter effects
- Performance impact

### Data Pipeline Transparency

End-to-end data flow visibility:

#### Data Validation
- Gap detection algorithms
- Outlier identification
- Quality assurance checks
- Audit trail maintenance

#### Performance Monitoring
- Latency tracking
- Throughput measurement
- Memory usage analysis
- Error rate monitoring

## Infrastructure Components

### Database Layer

SQL Server as the canonical data store:

#### Schema Design
- **Trades Table**: Raw trade data with microsecond precision
- **Orderbooks Table**: Bid/ask snapshots for microstructure analysis
- **OHLCV Tables**: Per-symbol aggregated candles
- **Metadata Tables**: Exchange and symbol information

#### Performance Optimizations
- Clustered indexes on timestamp columns
- Partitioning by date for large datasets
- Query optimization with execution plans
- Connection pooling for high throughput

### Shared Memory Layer

HotSpine shared memory infrastructure:

#### Memory Layout
```
struct HotSpineHeader {
    uint64_t version;
    uint64_t write_pos;
    uint64_t read_pos;
    uint64_t buffer_size;
    uint32_t lost_count;
};

struct HotTrade {
    uint64_t ts_exchange;    // Microsecond precision
    uint64_t ts_local;       // Local timestamp
    double price;            // Trade price
    double size;             // Trade size
    uint32_t symbol_id;      // Symbol identifier
    uint8_t side;            // 0=BUY, 1=SELL
};
```

#### Synchronization
- Atomic operations for thread safety
- Memory barriers for consistency
- Lock-free ring buffer implementation
- Cross-process coordination

### Network Layer

Multi-exchange connectivity:

#### Connection Management
- **WebSocket Pools**: Persistent connections with reconnection logic
- **Rate Limiting**: Venue-specific throttling
- **Failover**: Automatic backup connection routing
- **Health Monitoring**: Connection quality assessment

#### Data Normalization
- **Timestamp Alignment**: Exchange timestamp to UTC conversion
- **Symbol Mapping**: Unified symbol representation
- **Data Validation**: Real-time quality checks
- **Error Handling**: Graceful degradation on failures

## Performance Characteristics

### Latency Breakdown

| Component | Latency | Throughput |
|-----------|----------|------------|
| HotSpine (Single) | <1µs | 6M trades/sec |
| HotSpine (Batch) | ~0.07µs | 14M trades/sec |
| SQL Server Read | ~100µs | 10K queries/sec |
| SQL Server Write | ~50µs | 20K inserts/sec |
| CCXT API | 100-500ms | 10 requests/sec |
| WebSocket | <1ms | 1000+ messages/sec |

### Memory Usage

| Component | Memory Footprint | Scaling |
|-----------|------------------|---------|
| HotSpine Buffer | 32MB (1M trades) | Linear |
| SQL Server | 1-10GB | Linear |
| Python Process | 100-500MB | Sub-linear |
| Indicators | 10-50MB | Constant |

### CPU Utilization

| Operation | CPU Usage | Notes |
|-----------|-----------|-------|
| HotSpine Processing | <5% | C++ optimized |
| SQL Server | 10-30% | Depends on load |
| Python Strategy | 5-20% | Indicator calculations |
| Data Ingestion | 5-15% | C++ collectors |

## Scalability Considerations

### Horizontal Scaling

#### Data Ingestion
- Multiple C++ collectors per exchange
- Partitioned SQL Server tables
- Load-balanced ingestion pipeline

#### Strategy Execution
- Multi-process strategy instances
- Sharded data access
- Distributed optimization

### Vertical Scaling

#### Memory Optimization
- Efficient data structures
- Streaming processing
- Memory-mapped files

#### CPU Optimization
- C++ hot paths
- Parallel processing
- GPU acceleration for ML

### Cloud Deployment

#### Infrastructure as Code
- Docker containerization
- Kubernetes orchestration
- Infrastructure automation

#### Monitoring and Observability
- Prometheus metrics
- Grafana dashboards
- ELK stack logging
- Alert management

This architecture provides the foundation for institutional-grade algorithmic trading while maintaining the transparency and flexibility that makes BTQuant unique in the quantitative trading landscape.