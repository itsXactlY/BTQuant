#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stack>
#include <vector>
#include <unordered_map>
#include <functional>

#include "../include/data/TradeData.h"
#include "../include/analytics/cluster_engine.hpp"
#include "../include/indicator.hpp"
#include "../include/trading/order_manager.hpp"
#include "../include/analytics/technical_analysis.hpp"
#include "../include/data/data_types.hpp"
#include "../include/trading/position_manager.hpp"
#include "../include/trading/HotspineData.h"
#include "../include/data/compression.hpp"
#include "../include/components/chart_panel.hpp"
#include "../include/vulkan_dashboard_advanced.hpp"
#include "../include/components/volume_profile_panel.hpp"
#include "../include/components/orderbook_panel.hpp"

// Forward declaration for TradePaceData instead of including tape_panel.hpp
namespace BTQuant {
    namespace TapePanel {
        struct TradePaceData;
    }
}

namespace BTQuant {

// Generic memory pool template for fixed-size objects
template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t initial_capacity = 1024);
    ~ObjectPool();

    // Allocate an object from the pool
    template<typename... Args>
    T* allocate(Args&&... args);

    // Deallocate an object back to the pool
    void deallocate(T* obj);

    // Pre-allocate more objects to the pool
    void preallocate(size_t count);

    // Get pool statistics
    size_t get_total_objects() const { return total_objects_; }
    size_t get_free_objects() const { return free_list_.size(); }
    size_t get_used_objects() const { return total_objects_ - free_list_.size(); }

private:
    struct PoolBlock {
        alignas(T) char data[sizeof(T)];
    };

    std::mutex mutex_;
    std::stack<T*> free_list_;
    std::vector<std::unique_ptr<char[]>> blocks_;
    size_t total_objects_;
    size_t objects_per_block_;
};

// Specialized memory pools for specific object types
class TradeDataPool {
public:
    static TradeDataPool& getInstance();
    
    Data::TradeData* allocate();
    void deallocate(Data::TradeData* trade);
    void preallocate(size_t count = 1024);
    
    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    TradeDataPool() = default;
    ObjectPool<Data::TradeData> pool_;
};

class ClusterCellPool {
public:
    static ClusterCellPool& getInstance();
    
    Analytics::ClusterCell* allocate();
    void deallocate(Analytics::ClusterCell* cell);
    void preallocate(size_t count = 512);
    
    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    ClusterCellPool() = default;
    ObjectPool<Analytics::ClusterCell> pool_;
};

// Specific pools for different indicator types
class EMAIndicatorPool {
public:
    static EMAIndicatorPool& getInstance();

    EMAIndicator* allocate(int period);
    void deallocate(EMAIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    EMAIndicatorPool() = default;
    ObjectPool<EMAIndicator> pool_;
};

class SMAIndicatorPool {
public:
    static SMAIndicatorPool& getInstance();

    SMAIndicator* allocate(int period);
    void deallocate(SMAIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    SMAIndicatorPool() = default;
    ObjectPool<SMAIndicator> pool_;
};

class RSIIndicatorPool {
public:
    static RSIIndicatorPool& getInstance();

    RSIIndicator* allocate(int period);
    void deallocate(RSIIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    RSIIndicatorPool() = default;
    ObjectPool<RSIIndicator> pool_;
};

// MACDIndicatorPool for MACD indicators
class MACDIndicatorPool {
public:
    static MACDIndicatorPool& getInstance();

    MACDIndicator* allocate(int fast_period = 12, int slow_period = 26, int signal_period = 9);
    void deallocate(MACDIndicator* indicator);
    void preallocate(size_t count = 128);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    MACDIndicatorPool() = default;
    ObjectPool<MACDIndicator> pool_;
};

// BollingerBandIndicatorPool for Bollinger Band indicators
class BollingerBandIndicatorPool {
public:
    static BollingerBandIndicatorPool& getInstance();

    BollingerBandIndicator* allocate(int period, double std_dev = 2.0);
    void deallocate(BollingerBandIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    BollingerBandIndicatorPool() = default;
    ObjectPool<BollingerBandIndicator> pool_;
};

// StochasticIndicatorPool for Stochastic indicators
class StochasticIndicatorPool {
public:
    static StochasticIndicatorPool& getInstance();

    StochasticIndicator* allocate(int k_period = 14, int d_period = 3, int slowing_period = 3);
    void deallocate(StochasticIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    StochasticIndicatorPool() = default;
    ObjectPool<StochasticIndicator> pool_;
};

// ATRIndicatorPool for Average True Range indicators
class ATRIndicatorPool {
public:
    static ATRIndicatorPool& getInstance();

    ATRIndicator* allocate(int period = 14);
    void deallocate(ATRIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    ATRIndicatorPool() = default;
    ObjectPool<ATRIndicator> pool_;
};

// Additional specialized memory pools for other frequently allocated objects
class OrderPool {
public:
    static OrderPool& getInstance();

    OrderManager::Order* allocate();
    void deallocate(OrderManager::Order* order);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    OrderPool() = default;
    ObjectPool<OrderManager::Order> pool_;
};

class ProcessedTradePool {
public:
    static ProcessedTradePool& getInstance();

    ProcessedTrade* allocate();
    void deallocate(ProcessedTrade* trade);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    ProcessedTradePool() = default;
    ObjectPool<ProcessedTrade> pool_;
};

class OHLCVCandlePool {
public:
    static OHLCVCandlePool& getInstance();

    RenderEngine::OHLCVCandle* allocate();
    void deallocate(RenderEngine::OHLCVCandle* candle);
    void preallocate(size_t count = 512);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    OHLCVCandlePool() = default;
    ObjectPool<RenderEngine::OHLCVCandle> pool_;
};

class VolumeProfileLevelPool {
public:
    static VolumeProfileLevelPool& getInstance();

    VolumeProfileLevel* allocate();
    void deallocate(VolumeProfileLevel* level);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    VolumeProfileLevelPool() = default;
    ObjectPool<VolumeProfileLevel> pool_;
};

class TradeRecordPool {
public:
    static TradeRecordPool& getInstance();

    PositionManager::TradeRecord* allocate();
    void deallocate(PositionManager::TradeRecord* record);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    TradeRecordPool() = default;
    ObjectPool<PositionManager::TradeRecord> pool_;
};

// HotspineTradeTickPool for frequently allocated trade ticks
class HotspineTradeTickPool {
public:
    static HotspineTradeTickPool& getInstance();

    RenderEngine::HotspineTradeTick* allocate();
    void deallocate(RenderEngine::HotspineTradeTick* tick);
    void preallocate(size_t count = 2048); // Higher count since these are frequently allocated

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    HotspineTradeTickPool() = default;
    ObjectPool<RenderEngine::HotspineTradeTick> pool_;
};

// OrderBookLevelPool for order book levels
class OrderBookLevelPool {
public:
    static OrderBookLevelPool& getInstance();

    RenderEngine::OrderBookLevel* allocate();
    void deallocate(RenderEngine::OrderBookLevel* level);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    OrderBookLevelPool() = default;
    ObjectPool<RenderEngine::OrderBookLevel> pool_;
};

// CompressedCandlePool for compressed candle data
class CompressedCandlePool {
public:
    static CompressedCandlePool& getInstance();

    Data::CompressedCandle* allocate();
    void deallocate(Data::CompressedCandle* candle);
    void preallocate(size_t count = 512);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    CompressedCandlePool() = default;
    ObjectPool<Data::CompressedCandle> pool_;
};

// CompressedTradePool for compressed trade data
class CompressedTradePool {
public:
    static CompressedTradePool& getInstance();

    Data::CompressedTrade* allocate();
    void deallocate(Data::CompressedTrade* trade);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    CompressedTradePool() = default;
    ObjectPool<Data::CompressedTrade> pool_;
};

// FibonacciLevelPool for fibonacci retracement levels
class FibonacciLevelPool {
public:
    static FibonacciLevelPool& getInstance();

    FibonacciLevel* allocate();
    void deallocate(FibonacciLevel* level);
    void preallocate(size_t count = 64);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    FibonacciLevelPool() = default;
    ObjectPool<FibonacciLevel> pool_;
};

// GpuOrderBookLevelPool for GPU order book levels
class GpuOrderBookLevelPool {
public:
    static GpuOrderBookLevelPool& getInstance();

    RenderEngine::GpuOrderBookLevel* allocate();
    void deallocate(RenderEngine::GpuOrderBookLevel* level);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    GpuOrderBookLevelPool() = default;
    ObjectPool<RenderEngine::GpuOrderBookLevel> pool_;
};

// DepthLevelPool for depth levels in analytics
class DepthLevelPool {
public:
    static DepthLevelPool& getInstance();

    BTQuant::MarketDepthAnalyzer::DepthLevel* allocate();
    void deallocate(BTQuant::MarketDepthAnalyzer::DepthLevel* level);
    void preallocate(size_t count = 512);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    DepthLevelPool() = default;
    ObjectPool<BTQuant::MarketDepthAnalyzer::DepthLevel> pool_;
};

// HotOrderbookLevelPool for hot order book levels
class HotOrderbookLevelPool {
public:
    static HotOrderbookLevelPool& getInstance();

    HotOrderbookLevel* allocate();
    void deallocate(HotOrderbookLevel* level);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    HotOrderbookLevelPool() = default;
    ObjectPool<HotOrderbookLevel> pool_;
};

// RAII wrapper for automatic deallocation
template<typename T>
class PooledObject {
public:
    explicit PooledObject(T* obj, std::function<void(T*)> deleter_func)
        : obj_(obj), deleter_func_(deleter_func) {}

    ~PooledObject() {
        if (obj_ && deleter_func_) {
            deleter_func_(obj_);
        }
    }

    T* get() const { return obj_; }
    T& operator*() const { return *obj_; }
    T* operator->() const { return obj_; }
    explicit operator bool() const { return obj_ != nullptr; }

private:
    T* obj_;
    std::function<void(T*)> deleter_func_;
};

// Template implementations (included in header for template instantiation)
template<typename T>
ObjectPool<T>::ObjectPool(size_t initial_capacity)
    : total_objects_(0), objects_per_block_(0) {
    preallocate(initial_capacity);
}

template<typename T>
ObjectPool<T>::~ObjectPool() {
    // Clean up all objects
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.clear();
    // Clear the stack
    std::stack<T*> empty_stack;
    free_list_.swap(empty_stack);
}

template<typename T>
template<typename... Args>
T* ObjectPool<T>::allocate(Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (free_list_.empty()) {
        // Double the capacity if we run out
        size_t new_count = total_objects_ > 0 ? total_objects_ : 128;
        preallocate(new_count);
    }

    if (free_list_.empty()) {
        return nullptr; // No memory available
    }

    T* obj = free_list_.top();
    free_list_.pop();

    // Construct the object in place with provided arguments and return it
    return new (obj) T(std::forward<Args>(args)...);
}

template<typename T>
void ObjectPool<T>::deallocate(T* obj) {
    if (!obj) return;

    // Destruct the object
    obj->~T();

    std::lock_guard<std::mutex> lock(mutex_);

    // Add back to free list
    free_list_.push(obj);
}

template<typename T>
void ObjectPool<T>::preallocate(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate how much memory we need
    size_t total_size = count * sizeof(PoolBlock);
    auto block_memory = std::make_unique<char[]>(total_size);

    // Initialize each PoolBlock and add to free list
    char* block_ptr = block_memory.get();

    for (size_t i = 0; i < count; ++i) {
        PoolBlock* pool_block = reinterpret_cast<PoolBlock*>(block_ptr);

        // Get the address where the T object will be constructed
        T* obj_addr = reinterpret_cast<T*>(pool_block->data);

        // Add to free list
        free_list_.push(obj_addr);
        total_objects_++;

        // Move to next PoolBlock
        block_ptr += sizeof(PoolBlock);
    }

    blocks_.push_back(std::move(block_memory));
}

} // namespace BTQuant