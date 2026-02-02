# Memory Pool System

## Overview
This module implements a high-performance memory pool system for frequently allocated objects in the BTQuant trading platform. The system pre-allocates large blocks of memory for commonly used objects to reduce allocation overhead and improve performance.

## Components

### 1. TradeData Pool
- Manages pre-allocation of `TradeData` objects
- Used for storing trade information (timestamp, price, volume, etc.)
- Thread-safe with mutex protection
- Supports rapid allocation/deallocation of trade objects

### 2. ClusterCell Pool  
- Manages pre-allocation of `ClusterCell` objects
- Used in analytics for clustering market data
- Optimized for high-frequency allocation in cluster analysis

### 3. Indicator Pools
- Separate pools for different indicator types:
  - `EMAIndicatorPool` - Exponential Moving Average indicators
  - `SMAIndicatorPool` - Simple Moving Average indicators  
  - `RSIIndicatorPool` - Relative Strength Index indicators
- Each pool supports parameterized construction (e.g., period length)

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

### Pool Statistics
```cpp
auto total = trade_pool.getTotalObjects();
auto free = trade_pool.getFreeObjects();
auto used = trade_pool.getUsedObjects();
```

## Performance Benefits

- **Reduced Allocation Overhead**: Pre-allocated blocks eliminate malloc/free calls
- **Better Memory Locality**: Objects are allocated in contiguous memory regions
- **Reduced Fragmentation**: Large blocks minimize heap fragmentation
- **Thread Safety**: Mutex-protected access for concurrent usage
- **High Throughput**: Achieves millions of allocations/deallocations per second

## Architecture

The memory pool uses a generic `ObjectPool<T>` template that:
1. Pre-allocates large memory blocks
2. Maintains a free list of available object slots
3. Uses placement new for object construction
4. Supports parameterized construction via variadic templates
5. Provides thread-safe access through mutex protection

## Files

- `include/memory/memory_pool.hpp`: Main header with template definitions
- `src/memory/memory_pool.cpp`: Implementation of singleton pools
- `src/memory/test_memory_pool.cpp`: Basic functionality test
- `src/memory/performance_test.cpp`: Performance benchmark test