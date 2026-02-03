# Memory Pool System

## Overview
This module implements a high-performance memory pool system for frequently allocated objects in the BTQuant trading platform. The system pre-allocates large blocks of memory for commonly used objects to reduce allocation overhead and improve performance.

## Components

### 1. Standard Memory Pools
- **TradeData Pool**: Manages pre-allocation of `TradeData` objects
  - Used for storing trade information (timestamp, price, volume, etc.)
  - Thread-safe with mutex protection
  - Supports rapid allocation/deallocation of trade objects

- **ClusterCell Pool**: Manages pre-allocation of `ClusterCell` objects
  - Used in analytics for clustering market data
  - Optimized for high-frequency allocation in cluster analysis

- **Indicator Pools**: Separate pools for different indicator types:
  - `EMAIndicatorPool` - Exponential Moving Average indicators
  - `SMAIndicatorPool` - Simple Moving Average indicators
  - `RSIIndicatorPool` - Relative Strength Index indicators
  - `MACDIndicatorPool` - MACD indicators
  - `BollingerBandIndicatorPool` - Bollinger Band indicators
  - `StochasticIndicatorPool` - Stochastic indicators
  - `ATRIndicatorPool` - Average True Range indicators
- Each pool supports parameterized construction (e.g., period length)

### 2. Enhanced Fast Memory Pools
- **Performance-optimized pools** with higher pre-allocation counts
- **Thread-local storage** for critical performance paths
- **Built-in performance counters** for allocation/deallocation tracking
- **Fast variants** for frequently accessed objects:
  - `FastTradeDataPool` - Higher capacity for trade data
  - `FastClusterCellPool` - Optimized for clustering operations
  - `FastEMAIndicatorPool` - Enhanced EMA indicator handling
  - `FastFootprintCellPool` - Optimized for footprint charts
  - `FastHotspineTradeTickPool` - High-frequency trade tick processing

### 3. Trading-Specific Pools
- **Order and Trade Management**:
  - `OrderPool` - Order management objects
  - `ProcessedTradePool` - Processed trade records
  - `TradeRecordPool` - Position tracking records
- **Rendering Objects**:
  - `OHLCVCandlePool` - OHLCV candlestick data
  - `VolumeProfileLevelPool` - Volume profile levels
  - `CandleClusterPool` - Clustered candle data
  - `FootprintCellPool` - Footprint chart cells
  - `HotspineTradeTickPool` - High-frequency trade ticks

## Usage

### Basic Usage
```cpp
// Allocate a TradeData object
auto& trade_pool = BTQuant::TradeDataPool::getInstance();
auto* trade = trade_pool.allocate();

// Use the object
trade->timestamp = 1234567890;
trade->price = 100.50;
trade->volume = 10.0f;

// Return object to pool when done
trade_pool.deallocate(trade);
```

### Fast Pool Usage (Performance-Critical Paths)
```cpp
// Use fast pools for high-frequency allocations
auto& fast_trade_pool = BTQuant::FastTradeDataPool::getInstance();
auto* trade = fast_trade_pool.allocate();

// Use the object
trade->timestamp = 1234567890;
trade->price = 100.50;
trade->volume = 10.0f;

// Return object to pool when done
fast_trade_pool.deallocate(trade);
```

### Parameterized Allocation (Indicators)
```cpp
// Allocate an EMA indicator with period 14
auto& ema_pool = BTQuant::EMAIndicatorPool::getInstance();
auto* ema = ema_pool.allocate(14);  // Pass constructor parameters

// Use the indicator
ema->update(100.0f);
ema->update(101.0f);

// Return to pool when done
ema_pool.deallocate(ema);
```

### Pool Statistics and Monitoring
```cpp
// Basic statistics
auto total = trade_pool.getTotalObjects();
auto free = trade_pool.getFreeObjects();
auto used = trade_pool.getUsedObjects();

// Performance counters (for fast pools)
auto alloc_count = fast_trade_pool.getAllocationCount();
auto dealloc_count = fast_trade_pool.getDeallocationCount();

// Performance monitoring
RECORD_POOL_STATS("FastTradeDataPool", fast_trade_pool);
BTQuant::MemoryPoolMonitor::getInstance().printStatistics();
```

## Performance Benefits

- **Reduced Allocation Overhead**: Pre-allocated blocks eliminate malloc/free calls
- **Better Memory Locality**: Objects are allocated in contiguous memory regions
- **Reduced Fragmentation**: Large blocks minimize heap fragmentation
- **Thread Safety**: Mutex-protected access for concurrent usage
- **High Throughput**: Achieves millions of allocations/deallocations per second
- **Enhanced Performance**: Fast pools with thread-local optimizations
- **Detailed Monitoring**: Built-in performance counters and utilization tracking
- **Memory Savings**: Significant reduction in allocation overhead

## Architecture

The memory pool uses a generic `ObjectPool<T>` template that:
1. Pre-allocates large memory blocks
2. Maintains a free list of available object slots
3. Uses placement new for object construction
4. Supports parameterized construction via variadic templates
5. Provides thread-safe access through mutex protection

The enhanced `ThreadLocalObjectPool<T>` adds:
1. Performance counters for allocation/deallocation tracking
2. Higher pre-allocation counts for frequently used objects
3. Better performance monitoring capabilities

## Files

- `include/memory/memory_pool.hpp`: Main header with template definitions
- `src/memory/memory_pool.cpp`: Implementation of singleton pools
- `include/memory/pool_monitor.hpp`: Performance monitoring utilities
- `src/memory/test_memory_pool.cpp`: Basic functionality test
- `src/memory/performance_test.cpp`: Performance benchmark test