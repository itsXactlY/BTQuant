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
#include <cstdint>

// Lock-free free-list stack node for slab allocator
template<typename T>
struct alignas(64) FreeListNode {
    T* ptr;                              // Pointer to the actual object
    std::atomic<FreeListNode<T>*> next;  // Next node in the free-list (cache-line aligned)
    
    FreeListNode() : ptr(nullptr), next(nullptr) {}
    explicit FreeListNode(T* p) : ptr(p), next(nullptr) {}
};

// Lock-free free-list stack using std::atomic<Node*> for O(1) acquire/release
template<typename T>
class LockFreeFreeList {
public:
    LockFreeFreeList() : head_(nullptr), size_(0) {}
    
    ~LockFreeFreeList() {
        // Clean up remaining nodes
        FreeListNode<T>* current = head_.load(std::memory_order_relaxed);
        while (current) {
            FreeListNode<T>* next = current->next.load(std::memory_order_relaxed);
            delete current;
            current = next;
        }
    }
    
    // Acquire a node from the free-list (pop operation) - O(1)
    T* acquire() {
        FreeListNode<T>* old_head;
        FreeListNode<T>* new_head;
        
        do {
            old_head = head_.load(std::memory_order_acquire);
            if (!old_head) {
                return nullptr;  // Free-list is empty
            }
            new_head = old_head->next.load(std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(old_head, new_head,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire));
        
        T* result = old_head->ptr;
        size_.fetch_sub(1, std::memory_order_relaxed);
        
        // Delete the node structure (not the actual object)
        delete old_head;
        
        return result;
    }
    
    // Release a node back to the free-list (push operation) - O(1)
    void release(T* obj) {
        if (!obj) return;
        
        FreeListNode<T>* new_node = new FreeListNode<T>(obj);
        FreeListNode<T>* old_head;
        
        do {
            old_head = head_.load(std::memory_order_relaxed);
            new_node->next.store(old_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(old_head, new_node,
                                               std::memory_order_release,
                                               std::memory_order_relaxed));
        
        size_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Get current size (approximate, for statistics)
    size_t size() const {
        return size_.load(std::memory_order_relaxed);
    }
    
    // Check if empty
    bool empty() const {
        return head_.load(std::memory_order_acquire) == nullptr;
    }

private:
    std::atomic<FreeListNode<T>*> head_;  // Head of the free-list stack
    std::atomic<size_t> size_;            // Current size (approximate)
};

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

#include "../include/data/TradeData.h"
#include "../include/analytics/cluster_engine.hpp"
#include "../include/indicator.hpp"
#include "../include/analytics/technical_analysis.hpp"
#include "../include/data/data_types.hpp"
#include "../include/trading/HotspineData.h"
#include "../include/data/compression.hpp"
#include "../include/components/chart_panel.hpp"
#include "../include/vulkan_dashboard_advanced.hpp"
#include "../include/components/volume_profile_panel.hpp"
#include "../include/components/orderbook_panel.hpp"
#include "../include/widgets/VolumeProfileNode.h"
#include "../include/widgets/FootprintCell.h"
#include "../include/analytics/trading_analytics.hpp"


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

// Generic memory pool template for fixed-size objects with lock-free allocation
template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t arena_size = 1024 * 1024 * 1024);  // 1GB default arena
    ~ObjectPool();

    // Allocate an object from the pool - O(1) lock-free
    template<typename... Args>
    T* allocate(Args&&... args);

    // Deallocate an object back to the pool - O(1) lock-free
    void deallocate(T* obj);

    // Pre-allocate more objects to the pool
    void preallocate(size_t count);

    // Get pool statistics
    size_t get_total_objects() const { return total_objects_; }
    size_t get_free_objects() const { return free_list_.size(); }
    size_t get_used_objects() const { return total_objects_ - free_list_.size(); }

    // Get arena information
    size_t get_arena_size() const { return arena_size_; }
    size_t get_arena_used() const { return arena_offset_; }

private:
    struct PoolBlock {
        alignas(T) char data[sizeof(T)];
    };

    // Monolithic memory arena
    void* arena_;           // Pointer to the mmap/VirtualAlloc block
    size_t arena_size_;     // Total size of the arena
    size_t arena_offset_;   // Current offset within the arena for new allocations

    // Lock-free free-list for O(1) acquire/release without mutex
    LockFreeFreeList<T> free_list_;
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
// Uses lock-free free-list for O(1) acquire/release operations
template<typename T>
class ThreadLocalObjectPool {
public:
    explicit ThreadLocalObjectPool(size_t arena_size = 1024 * 1024 * 1024);  // 1GB default arena
    ~ThreadLocalObjectPool();

    // Allocate an object from the pool - O(1) lock-free
    template<typename... Args>
    T* allocate(Args&&... args);

    // Deallocate an object back to the pool - O(1) lock-free
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

    // Get arena information
    size_t get_arena_size() const { return arena_size_; }
    size_t get_arena_used() const { return arena_offset_; }

private:
    struct PoolBlock {
        alignas(T) char data[sizeof(T)];
    };

    // Monolithic memory arena
    void* arena_;           // Pointer to the mmap/VirtualAlloc block
    size_t arena_size_;     // Total size of the arena
    size_t arena_offset_;   // Current offset within the arena for new allocations

    // Lock-free free-list for O(1) acquire/release without mutex
    LockFreeFreeList<T> free_list_;
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

// OrderBookSnapshotPool for frequently allocated snapshots
class OrderBookSnapshotPool {
public:
    static OrderBookSnapshotPool& getInstance();

    BTQuant::OrderBookSnapshot* allocate();
    void deallocate(BTQuant::OrderBookSnapshot* snapshot);
    void preallocate(size_t count = 512);

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }

private:
    OrderBookSnapshotPool() = default;
    ObjectPool<BTQuant::OrderBookSnapshot> pool_;
};

// FastOrderBookSnapshotPool for high-performance snapshot allocation
class FastOrderBookSnapshotPool {
public:
    static FastOrderBookSnapshotPool& getInstance();

    BTQuant::OrderBookSnapshot* allocate();
    void deallocate(BTQuant::OrderBookSnapshot* snapshot);
    void preallocate(size_t count = 1024); // Higher count for frequent allocation

    size_t getTotalObjects() const { return pool_.get_total_objects(); }
    size_t getFreeObjects() const { return pool_.get_free_objects(); }
    size_t getUsedObjects() const { return pool_.get_used_objects(); }
    size_t getAllocationCount() const { return pool_.get_allocation_count(); }
    size_t getDeallocationCount() const { return pool_.get_deallocation_count(); }

private:
    FastOrderBookSnapshotPool() = default;
    ThreadLocalObjectPool<BTQuant::OrderBookSnapshot> pool_;
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
ObjectPool<T>::ObjectPool(size_t arena_size)
    : arena_(nullptr), arena_size_(arena_size), arena_offset_(0), total_objects_(0), objects_per_block_(0) {
    // Allocate monolithic memory arena using mmap (Linux) or VirtualAlloc (Windows)
#ifdef _WIN32
    arena_ = VirtualAlloc(nullptr, arena_size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    arena_ = mmap(nullptr, arena_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena_ == MAP_FAILED) {
        arena_ = nullptr;
    }
#endif
}

template<typename T>
ObjectPool<T>::~ObjectPool() {
    // Clean up all objects
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear the stack
    std::stack<T*> empty_stack;
    free_list_.swap(empty_stack);
    
    // Free the monolithic arena
    if (arena_) {
#ifdef _WIN32
        VirtualFree(arena_, 0, MEM_RELEASE);
#else
        munmap(arena_, arena_size_);
#endif
        arena_ = nullptr;
    }
}

template<typename T>
template<typename... Args>
T* ObjectPool<T>::allocate(Args&&... args) {
    // Try to acquire from lock-free free-list first - O(1)
    T* obj = free_list_.acquire();
    
    if (!obj) {
        // Free-list is empty, need to allocate more objects from arena
        // Use a simple spin-lock for arena expansion (rare operation)
        std::atomic<bool> lock(false);
        while (lock.exchange(true, std::memory_order_acquire)) {
            // Spin-wait with pause for better performance
            #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                #ifdef _MSC_VER
                    _mm_pause();
                #else
                    __builtin_ia32_pause();
                #endif
            #endif
        }
        
        // Double-check after acquiring lock
        if (free_list_.empty()) {
            // Allocate more objects from the arena if we run out
            size_t new_count = total_objects_ > 0 ? total_objects_ : 128;
            
            if (arena_ && arena_offset_ + new_count * sizeof(PoolBlock) <= arena_size_) {
                // Initialize each PoolBlock and add to free list from the monolithic arena
                char* block_ptr = static_cast<char*>(arena_) + arena_offset_;
                
                for (size_t i = 0; i < new_count; ++i) {
                    PoolBlock* pool_block = reinterpret_cast<PoolBlock*>(block_ptr);
                    T* obj_addr = reinterpret_cast<T*>(pool_block->data);
                    free_list_.release(obj_addr);
                    total_objects_++;
                    block_ptr += sizeof(PoolBlock);
                }
                
                arena_offset_ += new_count * sizeof(PoolBlock);
            }
        }
        
        lock.store(false, std::memory_order_release);
        
        // Try to acquire again
        obj = free_list_.acquire();
    }
    
    if (!obj) {
        return nullptr; // No memory available
    }

    // Construct the object in place with provided arguments and return it
    return new (obj) T(std::forward<Args>(args)...);
}

template<typename T>
void ObjectPool<T>::deallocate(T* obj) {
    if (!obj) return;

    // Destruct the object
    obj->~T();

    // Release back to lock-free free-list - O(1)
    free_list_.release(obj);
}

template<typename T>
void ObjectPool<T>::preallocate(size_t count) {
    if (!arena_) {
        return; // Arena not allocated
    }

    // Calculate how much memory we need
    size_t total_size = count * sizeof(PoolBlock);

    // Check if we have enough space in the arena
    if (arena_offset_ + total_size > arena_size_) {
        return; // Not enough space in arena
    }

    // Initialize each PoolBlock and add to free list from the monolithic arena
    char* block_ptr = static_cast<char*>(arena_) + arena_offset_;

    for (size_t i = 0; i < count; ++i) {
        PoolBlock* pool_block = reinterpret_cast<PoolBlock*>(block_ptr);

        // Get the address where the T object will be constructed
        T* obj_addr = reinterpret_cast<T*>(pool_block->data);

        // Add to lock-free free-list - O(1) release
        free_list_.release(obj_addr);
        total_objects_++;

        // Move to next PoolBlock
        block_ptr += sizeof(PoolBlock);
    }

    arena_offset_ += total_size;
}

// ThreadLocalObjectPool implementation
template<typename T>
ThreadLocalObjectPool<T>::ThreadLocalObjectPool(size_t arena_size)
    : arena_(nullptr), arena_size_(arena_size), arena_offset_(0), total_objects_(0), objects_per_block_(0) {
    // Allocate monolithic memory arena using mmap (Linux) or VirtualAlloc (Windows)
#ifdef _WIN32
    arena_ = VirtualAlloc(nullptr, arena_size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    arena_ = mmap(nullptr, arena_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena_ == MAP_FAILED) {
        arena_ = nullptr;
    }
#endif
}

template<typename T>
ThreadLocalObjectPool<T>::~ThreadLocalObjectPool() {
    // Free the monolithic arena
    if (arena_) {
#ifdef _WIN32
        VirtualFree(arena_, 0, MEM_RELEASE);
#else
        munmap(arena_, arena_size_);
#endif
        arena_ = nullptr;
    }
}

template<typename T>
template<typename... Args>
T* ThreadLocalObjectPool<T>::allocate(Args&&... args) {
    // Try to acquire from lock-free free-list first - O(1)
    T* obj = free_list_.acquire();
    
    if (!obj) {
        // Free-list is empty, need to allocate more objects from arena
        // Use a simple spin-lock for arena expansion (rare operation)
        std::atomic<bool> lock(false);
        while (lock.exchange(true, std::memory_order_acquire)) {
            #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                #ifdef _MSC_VER
                    _mm_pause();
                #else
                    __builtin_ia32_pause();
                #endif
            #endif
        }
        
        // Double-check after acquiring lock
        if (free_list_.empty()) {
            size_t new_count = total_objects_ > 0 ? total_objects_ : 128;
            
            if (arena_ && arena_offset_ + new_count * sizeof(PoolBlock) <= arena_size_) {
                char* block_ptr = static_cast<char*>(arena_) + arena_offset_;
                
                for (size_t i = 0; i < new_count; ++i) {
                    PoolBlock* pool_block = reinterpret_cast<PoolBlock*>(block_ptr);
                    T* obj_addr = reinterpret_cast<T*>(pool_block->data);
                    free_list_.release(obj_addr);
                    total_objects_++;
                    block_ptr += sizeof(PoolBlock);
                }
                
                arena_offset_ += new_count * sizeof(PoolBlock);
            }
        }
        
        lock.store(false, std::memory_order_release);
        
        // Try to acquire again
        obj = free_list_.acquire();
    }
    
    if (!obj) {
        return nullptr; // No memory available
    }

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

    // Release back to lock-free free-list - O(1)
    free_list_.release(obj);

    // Increment deallocation counter
    deallocation_count_.fetch_add(1, std::memory_order_relaxed);
}

template<typename T>
void ThreadLocalObjectPool<T>::preallocate(size_t count) {
    if (!arena_) {
        return; // Arena not allocated
    }

    // Calculate how much memory we need
    size_t total_size = count * sizeof(PoolBlock);

    // Check if we have enough space in the arena
    if (arena_offset_ + total_size > arena_size_) {
        return; // Not enough space in arena
    }

    // Initialize each PoolBlock and add to free list from the monolithic arena
    char* block_ptr = static_cast<char*>(arena_) + arena_offset_;

    for (size_t i = 0; i < count; ++i) {
        PoolBlock* pool_block = reinterpret_cast<PoolBlock*>(block_ptr);

        // Get the address where the T object will be constructed
        T* obj_addr = reinterpret_cast<T*>(pool_block->data);

        // Add to lock-free free-list - O(1) release
        free_list_.release(obj_addr);
        total_objects_++;

        // Move to next PoolBlock
        block_ptr += sizeof(PoolBlock);
    }

    arena_offset_ += total_size;
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