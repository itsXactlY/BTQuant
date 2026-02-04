#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stack>
#include <vector>
#include <unordered_map>
#include <functional>
#include <thread>

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
#include "../include/widgets/VolumeProfileNode.h"
#include "../include/widgets/FootprintCell.h"
#include "../include/analytics/trading_analytics.hpp"
#include "../include/hotspine_data_bridge.hpp"

// Forward declaration for TradePaceData instead of including tape_panel.hpp
namespace BTQuant {
    namespace TapePanel {
        struct TradePaceData {
            uint64_t timestamp{0};
            int trades_per_minute{0};
            double volume_per_minute{0.0};
            int peak_trades_per_minute{0};
            double peak_volume_per_minute{0.0};

            TradePaceData() = default;
            TradePaceData(uint64_t ts, int tpm, double vpm, int ptm, double pvpm)
                : timestamp(ts), trades_per_minute(tpm), volume_per_minute(vpm),
                  peak_trades_per_minute(ptm), peak_volume_per_minute(pvpm) {}
        };
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

// CandleClusterPool for frequently allocated candle clusters
class CandleClusterPool {
public:
    static CandleClusterPool& getInstance();

    RenderEngine::CandleCluster* allocate(float x = 0.0f, float y = 0.0f, float w = 0.0f, float h = 0.0f,
                                         uint32_t bidVol = 0, uint32_t askVol = 0, uint32_t tradeCnt = 0,
                                         float vw = 0.0f, bool hasTrades = false);
    void deallocate(RenderEngine::CandleCluster* cluster);
    void preallocate(size_t count = 512);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    CandleClusterPool() = default;
    ObjectPool<RenderEngine::CandleCluster> pool_;
};

// VolumeProfileNodePool for frequently allocated volume profile nodes
class VolumeProfileNodePool {
public:
    static VolumeProfileNodePool& getInstance();

    VolumeProfileNode* allocate();
    void deallocate(VolumeProfileNode* node);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    VolumeProfileNodePool() = default;
    ObjectPool<VolumeProfileNode> pool_;
};

// FootprintCellPool for frequently allocated footprint cells
class FootprintCellPool {
public:
    static FootprintCellPool& getInstance();

    FootprintCell* allocate();
    void deallocate(FootprintCell* cell);
    void preallocate(size_t count = 2048);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    FootprintCellPool() = default;
    ObjectPool<FootprintCell> pool_;
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

// Thread-local memory pool for even better performance in multi-threaded scenarios
template<typename T>
class ThreadLocalObjectPool {
public:
    explicit ThreadLocalObjectPool(size_t initial_capacity = 1024);

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

    // Get allocation/deallocation counters for performance monitoring
    size_t get_allocation_count() const { return allocation_count_.load(std::memory_order_relaxed); }
    size_t get_deallocation_count() const { return deallocation_count_.load(std::memory_order_relaxed); }

private:
    struct PoolBlock {
        alignas(T) char data[sizeof(T)];
    };

    std::mutex mutex_;
    std::stack<T*> free_list_;
    std::vector<std::unique_ptr<char[]>> blocks_;
    size_t total_objects_;
    size_t objects_per_block_;

    // Performance counters
    std::atomic<size_t> allocation_count_{0};
    std::atomic<size_t> deallocation_count_{0};
};

// Enhanced memory pools with thread-local storage for frequently allocated objects
class FastTradeDataPool {
public:
    static FastTradeDataPool& getInstance();

    Data::TradeData* allocate();
    void deallocate(Data::TradeData* trade);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastTradeDataPool() = default;
    ThreadLocalObjectPool<Data::TradeData> pool_;
};

class FastClusterCellPool {
public:
    static FastClusterCellPool& getInstance();

    Analytics::ClusterCell* allocate();
    void deallocate(Analytics::ClusterCell* cell);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastClusterCellPool() = default;
    ThreadLocalObjectPool<Analytics::ClusterCell> pool_;
};

class FastEMAIndicatorPool {
public:
    static FastEMAIndicatorPool& getInstance();

    EMAIndicator* allocate(int period);
    void deallocate(EMAIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastEMAIndicatorPool() = default;
    ThreadLocalObjectPool<EMAIndicator> pool_;
};

class FastSMAIndicatorPool {
public:
    static FastSMAIndicatorPool& getInstance();

    SMAIndicator* allocate(int period);
    void deallocate(SMAIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastSMAIndicatorPool() = default;
    ThreadLocalObjectPool<SMAIndicator> pool_;
};

class FastRSIIndicatorPool {
public:
    static FastRSIIndicatorPool& getInstance();

    RSIIndicator* allocate(int period);
    void deallocate(RSIIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastRSIIndicatorPool() = default;
    ThreadLocalObjectPool<RSIIndicator> pool_;
};

class FastMACDIndicatorPool {
public:
    static FastMACDIndicatorPool& getInstance();

    MACDIndicator* allocate(int fast_period, int slow_period, int signal_period);
    void deallocate(MACDIndicator* indicator);
    void preallocate(size_t count = 256); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastMACDIndicatorPool() = default;
    ThreadLocalObjectPool<MACDIndicator> pool_;
};

class FastBollingerBandIndicatorPool {
public:
    static FastBollingerBandIndicatorPool& getInstance();

    BollingerBandIndicator* allocate(int period, double std_dev);
    void deallocate(BollingerBandIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastBollingerBandIndicatorPool() = default;
    ThreadLocalObjectPool<BollingerBandIndicator> pool_;
};

class FastStochasticIndicatorPool {
public:
    static FastStochasticIndicatorPool& getInstance();

    StochasticIndicator* allocate(int k_period, int d_period, int slowing_period);
    void deallocate(StochasticIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastStochasticIndicatorPool() = default;
    ThreadLocalObjectPool<StochasticIndicator> pool_;
};

class FastATRIndicatorPool {
public:
    static FastATRIndicatorPool& getInstance();

    ATRIndicator* allocate(int period);
    void deallocate(ATRIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastATRIndicatorPool() = default;
    ThreadLocalObjectPool<ATRIndicator> pool_;
};

class FastOrderPool {
public:
    static FastOrderPool& getInstance();

    OrderManager::Order* allocate();
    void deallocate(OrderManager::Order* order);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastOrderPool() = default;
    ThreadLocalObjectPool<OrderManager::Order> pool_;
};

class FastProcessedTradePool {
public:
    static FastProcessedTradePool& getInstance();

    ProcessedTrade* allocate();
    void deallocate(ProcessedTrade* trade);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastProcessedTradePool() = default;
    ThreadLocalObjectPool<ProcessedTrade> pool_;
};

class FastOHLCVCandlePool {
public:
    static FastOHLCVCandlePool& getInstance();

    RenderEngine::OHLCVCandle* allocate();
    void deallocate(RenderEngine::OHLCVCandle* candle);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastOHLCVCandlePool() = default;
    ThreadLocalObjectPool<RenderEngine::OHLCVCandle> pool_;
};

class FastVolumeProfileLevelPool {
public:
    static FastVolumeProfileLevelPool& getInstance();

    VolumeProfileLevel* allocate();
    void deallocate(VolumeProfileLevel* level);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastVolumeProfileLevelPool() = default;
    ThreadLocalObjectPool<VolumeProfileLevel> pool_;
};

class FastTradeRecordPool {
public:
    static FastTradeRecordPool& getInstance();

    PositionManager::TradeRecord* allocate();
    void deallocate(PositionManager::TradeRecord* record);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastTradeRecordPool() = default;
    ThreadLocalObjectPool<PositionManager::TradeRecord> pool_;
};

class FastHotspineTradeTickPool {
public:
    static FastHotspineTradeTickPool& getInstance();

    RenderEngine::HotspineTradeTick* allocate();
    void deallocate(RenderEngine::HotspineTradeTick* tick);
    void preallocate(size_t count = 4096); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastHotspineTradeTickPool() = default;
    ThreadLocalObjectPool<RenderEngine::HotspineTradeTick> pool_;
};

class FastOrderBookLevelPool {
public:
    static FastOrderBookLevelPool& getInstance();

    RenderEngine::OrderBookLevel* allocate();
    void deallocate(RenderEngine::OrderBookLevel* level);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastOrderBookLevelPool() = default;
    ThreadLocalObjectPool<RenderEngine::OrderBookLevel> pool_;
};

class FastCompressedCandlePool {
public:
    static FastCompressedCandlePool& getInstance();

    Data::CompressedCandle* allocate();
    void deallocate(Data::CompressedCandle* candle);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastCompressedCandlePool() = default;
    ThreadLocalObjectPool<Data::CompressedCandle> pool_;
};

class FastCompressedTradePool {
public:
    static FastCompressedTradePool& getInstance();

    Data::CompressedTrade* allocate();
    void deallocate(Data::CompressedTrade* trade);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastCompressedTradePool() = default;
    ThreadLocalObjectPool<Data::CompressedTrade> pool_;
};

class FastFibonacciLevelPool {
public:
    static FastFibonacciLevelPool& getInstance();

    FibonacciLevel* allocate();
    void deallocate(FibonacciLevel* level);
    void preallocate(size_t count = 128); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastFibonacciLevelPool() = default;
    ThreadLocalObjectPool<FibonacciLevel> pool_;
};

class FastGpuOrderBookLevelPool {
public:
    static FastGpuOrderBookLevelPool& getInstance();

    RenderEngine::GpuOrderBookLevel* allocate();
    void deallocate(RenderEngine::GpuOrderBookLevel* level);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastGpuOrderBookLevelPool() = default;
    ThreadLocalObjectPool<RenderEngine::GpuOrderBookLevel> pool_;
};

class FastDepthLevelPool {
public:
    static FastDepthLevelPool& getInstance();

    BTQuant::MarketDepthAnalyzer::DepthLevel* allocate();
    void deallocate(BTQuant::MarketDepthAnalyzer::DepthLevel* level);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastDepthLevelPool() = default;
    ThreadLocalObjectPool<BTQuant::MarketDepthAnalyzer::DepthLevel> pool_;
};

class FastHotOrderbookLevelPool {
public:
    static FastHotOrderbookLevelPool& getInstance();

    HotOrderbookLevel* allocate();
    void deallocate(HotOrderbookLevel* level);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastHotOrderbookLevelPool() = default;
    ThreadLocalObjectPool<HotOrderbookLevel> pool_;
};

class FastCandleClusterPool {
public:
    static FastCandleClusterPool& getInstance();

    RenderEngine::CandleCluster* allocate(float x = 0.0f, float y = 0.0f, float w = 0.0f, float h = 0.0f,
                                         uint32_t bidVol = 0, uint32_t askVol = 0, uint32_t tradeCnt = 0,
                                         float vw = 0.0f, bool hasTrades = false);
    void deallocate(RenderEngine::CandleCluster* cluster);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastCandleClusterPool() = default;
    ThreadLocalObjectPool<RenderEngine::CandleCluster> pool_;
};

class FastVolumeProfileNodePool {
public:
    static FastVolumeProfileNodePool& getInstance();

    VolumeProfileNode* allocate();
    void deallocate(VolumeProfileNode* node);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastVolumeProfileNodePool() = default;
    ThreadLocalObjectPool<VolumeProfileNode> pool_;
};

class FastFootprintCellPool {
public:
    static FastFootprintCellPool& getInstance();

    FootprintCell* allocate();
    void deallocate(FootprintCell* cell);
    void preallocate(size_t count = 4096); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastFootprintCellPool() = default;
    ThreadLocalObjectPool<FootprintCell> pool_;
};

// Additional memory pools for other frequently allocated objects

// TradePool for generic trade objects
class TradePool {
public:
    static TradePool& getInstance();

    BTQuant::Trade* allocate();
    void deallocate(BTQuant::Trade* trade);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    TradePool() = default;
    ObjectPool<BTQuant::Trade> pool_;
};

// FastTradePool for high-performance trade allocation
class FastTradePool {
public:
    static FastTradePool& getInstance();

    BTQuant::Trade* allocate();
    void deallocate(BTQuant::Trade* trade);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastTradePool() = default;
    ThreadLocalObjectPool<BTQuant::Trade> pool_;
};

// HotOrderbookSnapshotPool for frequently allocated snapshots
class HotOrderbookSnapshotPool {
public:
    static HotOrderbookSnapshotPool& getInstance();

    BTQuant::HotOrderbookSnapshot* allocate();
    void deallocate(BTQuant::HotOrderbookSnapshot* snapshot);
    void preallocate(size_t count = 512);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    HotOrderbookSnapshotPool() = default;
    ObjectPool<BTQuant::HotOrderbookSnapshot> pool_;
};

// FastHotOrderbookSnapshotPool for high-performance snapshot allocation
class FastHotOrderbookSnapshotPool {
public:
    static FastHotOrderbookSnapshotPool& getInstance();

    BTQuant::HotOrderbookSnapshot* allocate();
    void deallocate(BTQuant::HotOrderbookSnapshot* snapshot);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastHotOrderbookSnapshotPool() = default;
    ThreadLocalObjectPool<BTQuant::HotOrderbookSnapshot> pool_;
};

// IndicatorResultPool for frequently allocated indicator results
class IndicatorResultPool {
public:
    static IndicatorResultPool& getInstance();

    BTQuant::TechnicalIndicators::IndicatorResult* allocate();
    void deallocate(BTQuant::TechnicalIndicators::IndicatorResult* result);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    IndicatorResultPool() = default;
    ObjectPool<BTQuant::TechnicalIndicators::IndicatorResult> pool_;
};

// FastIndicatorResultPool for high-performance indicator result allocation
class FastIndicatorResultPool {
public:
    static FastIndicatorResultPool& getInstance();

    BTQuant::TechnicalIndicators::IndicatorResult* allocate();
    void deallocate(BTQuant::TechnicalIndicators::IndicatorResult* result);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastIndicatorResultPool() = default;
    ThreadLocalObjectPool<BTQuant::TechnicalIndicators::IndicatorResult> pool_;
};

// OrderExecutionPool for frequently allocated order executions
class OrderExecutionPool {
public:
    static OrderExecutionPool& getInstance();

    OrderManager::OrderExecution* allocate();
    void deallocate(OrderManager::OrderExecution* execution);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    OrderExecutionPool() = default;
    ObjectPool<OrderManager::OrderExecution> pool_;
};

// FastOrderExecutionPool for high-performance order execution allocation
class FastOrderExecutionPool {
public:
    static FastOrderExecutionPool& getInstance();

    OrderManager::OrderExecution* allocate();
    void deallocate(OrderManager::OrderExecution* execution);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastOrderExecutionPool() = default;
    ThreadLocalObjectPool<OrderManager::OrderExecution> pool_;
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

// ThreadLocalObjectPool implementation
template<typename T>
ThreadLocalObjectPool<T>::ThreadLocalObjectPool(size_t initial_capacity)
    : total_objects_(0), objects_per_block_(0) {
    preallocate(initial_capacity);
}

template<typename T>
template<typename... Args>
T* ThreadLocalObjectPool<T>::allocate(Args&&... args) {
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

    // Increment allocation counter
    allocation_count_.fetch_add(1, std::memory_order_relaxed);

    // Construct the object in place with provided arguments and return it
    return new (obj) T(std::forward<Args>(args)...);
}

template<typename T>
void ThreadLocalObjectPool<T>::deallocate(T* obj) {
    if (!obj) return;

    // Destruct the object
    obj->~T();

    std::lock_guard<std::mutex> lock(mutex_);

    // Add back to free list
    free_list_.push(obj);

    // Increment deallocation counter
    deallocation_count_.fetch_add(1, std::memory_order_relaxed);
}

template<typename T>
void ThreadLocalObjectPool<T>::preallocate(size_t count) {
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

// Additional memory pools for other frequently allocated objects

// TradePaceDataPool for trade pace analysis data
class TradePaceDataPool {
public:
    static TradePaceDataPool& getInstance();

    BTQuant::TapePanel::TradePaceData* allocate();
    void deallocate(BTQuant::TapePanel::TradePaceData* data);
    void preallocate(size_t count = 512);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    TradePaceDataPool() = default;
    ObjectPool<BTQuant::TapePanel::TradePaceData> pool_;
};

// FastTradePaceDataPool for high-performance trade pace allocation
class FastTradePaceDataPool {
public:
    static FastTradePaceDataPool& getInstance();

    BTQuant::TapePanel::TradePaceData* allocate();
    void deallocate(BTQuant::TapePanel::TradePaceData* data);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastTradePaceDataPool() = default;
    ThreadLocalObjectPool<BTQuant::TapePanel::TradePaceData> pool_;
};

// TradePairPool for pairs of trades used in analysis
class TradePairPool {
public:
    static TradePairPool& getInstance();

    std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* allocate();
    void deallocate(std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* pair);
    void preallocate(size_t count = 1024);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    TradePairPool() = default;
    ObjectPool<std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>> pool_;
};

// FastTradePairPool for high-performance trade pair allocation
class FastTradePairPool {
public:
    static FastTradePairPool& getInstance();

    std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* allocate();
    void deallocate(std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>* pair);
    void preallocate(size_t count = 2048); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastTradePairPool() = default;
    ThreadLocalObjectPool<std::pair<BTQuant::Data::TradeData, BTQuant::Data::TradeData>> pool_;
};

// Additional memory pools for other frequently allocated objects

// FastIndicatorValuePairPool for high-performance indicator value pair allocation
class FastIndicatorValuePairPool {
public:
    static FastIndicatorValuePairPool& getInstance();

    std::pair<float, float>* allocate();
    void deallocate(std::pair<float, float>* pair);
    void preallocate(size_t count = 4096); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastIndicatorValuePairPool() = default;
    ThreadLocalObjectPool<std::pair<float, float>> pool_;
};

// Additional specialized memory pools for other indicator types that might be frequently allocated

// Parabolic SAR Indicator Pool
class PSARIndicatorPool {
public:
    static PSARIndicatorPool& getInstance();

    PSARIndicator* allocate(float acceleration_step = 0.02f, float acceleration_max = 0.2f);
    void deallocate(PSARIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    PSARIndicatorPool() = default;
    ObjectPool<PSARIndicator> pool_;
};

class FastPSARIndicatorPool {
public:
    static FastPSARIndicatorPool& getInstance();

    PSARIndicator* allocate(float acceleration_step = 0.02f, float acceleration_max = 0.2f);
    void deallocate(PSARIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastPSARIndicatorPool() = default;
    ThreadLocalObjectPool<PSARIndicator> pool_;
};

// CCI Indicator Pool (Commodity Channel Index)
class CCIIndicatorPool {
public:
    static CCIIndicatorPool& getInstance();

    CCIIndicator* allocate(int period = 20);
    void deallocate(CCIIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    CCIIndicatorPool() = default;
    ObjectPool<CCIIndicator> pool_;
};

class FastCCIIndicatorPool {
public:
    static FastCCIIndicatorPool& getInstance();

    CCIIndicator* allocate(int period = 20);
    void deallocate(CCIIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastCCIIndicatorPool() = default;
    ThreadLocalObjectPool<CCIIndicator> pool_;
};

// Williams %R Indicator Pool
class WilliamsRIndicatorPool {
public:
    static WilliamsRIndicatorPool& getInstance();

    WilliamsRIndicator* allocate(int period = 14);
    void deallocate(WilliamsRIndicator* indicator);
    void preallocate(size_t count = 256);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    WilliamsRIndicatorPool() = default;
    ObjectPool<WilliamsRIndicator> pool_;
};

class FastWilliamsRIndicatorPool {
public:
    static FastWilliamsRIndicatorPool& getInstance();

    WilliamsRIndicator* allocate(int period = 14);
    void deallocate(WilliamsRIndicator* indicator);
    void preallocate(size_t count = 512); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastWilliamsRIndicatorPool() = default;
    ThreadLocalObjectPool<WilliamsRIndicator> pool_;
};

} // namespace BTQuant