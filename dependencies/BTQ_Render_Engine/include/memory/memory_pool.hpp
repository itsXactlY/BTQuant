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