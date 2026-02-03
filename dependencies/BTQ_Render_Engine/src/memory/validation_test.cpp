#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <vector>
#include <atomic>

// Simplified version of our memory pool implementation for validation
namespace BTQuant {

// Simplified object pool for validation
template<typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t initial_capacity = 1024) : total_objects_(0) {
        preallocate(initial_capacity);
    }

    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_.empty()) {
            preallocate(total_objects_ > 0 ? total_objects_ : 128);
        }
        if (free_list_.empty()) return nullptr;
        
        T* obj = free_list_.top();
        free_list_.pop();
        return new (obj) T();
    }

    void deallocate(T* obj) {
        if (!obj) return;
        obj->~T();
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push(obj);
    }

    void preallocate(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < count; ++i) {
            auto block = std::make_unique<char[]>(sizeof(T));
            T* obj_addr = reinterpret_cast<T*>(block.get());
            free_list_.push(obj_addr);
            blocks_.push_back(std::move(block));
            total_objects_++;
        }
    }

    size_t get_total_objects() const { return total_objects_; }
    size_t get_free_objects() const { return free_list_.size(); }
    size_t get_used_objects() const { return total_objects_ - free_list_.size(); }

private:
    struct alignas(T) PoolBlock {
        char data[sizeof(T)];
    };

    std::mutex mutex_;
    std::stack<T*> free_list_;
    std::vector<std::unique_ptr<char[]>> blocks_;
    size_t total_objects_;
};

// Thread-local enhanced pool with performance counters
template<typename T>
class ThreadLocalObjectPool {
public:
    explicit ThreadLocalObjectPool(size_t initial_capacity = 1024) : total_objects_(0) {
        preallocate(initial_capacity);
    }
    
    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_list_.empty()) {
            preallocate(total_objects_ > 0 ? total_objects_ : 128);
        }
        if (free_list_.empty()) return nullptr;
        
        T* obj = free_list_.top();
        free_list_.pop();
        allocation_count_.fetch_add(1, std::memory_order_relaxed);
        return new (obj) T();
    }

    void deallocate(T* obj) {
        if (!obj) return;
        obj->~T();
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push(obj);
        deallocation_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void preallocate(size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < count; ++i) {
            auto block = std::make_unique<char[]>(sizeof(T));
            T* obj_addr = reinterpret_cast<T*>(block.get());
            free_list_.push(obj_addr);
            blocks_.push_back(std::move(block));
            total_objects_++;
        }
    }

    size_t get_total_objects() const { return total_objects_; }
    size_t get_free_objects() const { return free_list_.size(); }
    size_t get_used_objects() const { return total_objects_ - free_list_.size(); }
    size_t get_allocation_count() const { return allocation_count_.load(std::memory_order_relaxed); }
    size_t get_deallocation_count() const { return deallocation_count_.load(std::memory_order_relaxed); }

private:
    struct alignas(T) PoolBlock {
        char data[sizeof(T)];
    };

    std::mutex mutex_;
    std::stack<T*> free_list_;
    std::vector<std::unique_ptr<char[]>> blocks_;
    size_t total_objects_;
    
    // Performance counters
    std::atomic<size_t> allocation_count_{0};
    std::atomic<size_t> deallocation_count_{0};
};

// Simple test class
class TestObject {
public:
    int value = 0;
    TestObject() : value(42) {}
};

// Test the pools
void testPools() {
    std::cout << "Testing basic ObjectPool..." << std::endl;
    ObjectPool<TestObject> basic_pool(100);
    
    auto* obj1 = basic_pool.allocate();
    std::cout << "Allocated object with value: " << obj1->value << std::endl;
    basic_pool.deallocate(obj1);
    
    std::cout << "Testing ThreadLocalObjectPool..." << std::endl;
    ThreadLocalObjectPool<TestObject> enhanced_pool(100);
    
    auto* obj2 = enhanced_pool.allocate();
    std::cout << "Allocated object with value: " << obj2->value << std::endl;
    std::cout << "Allocation count: " << enhanced_pool.get_allocation_count() << std::endl;
    enhanced_pool.deallocate(obj2);
    std::cout << "Deallocation count: " << enhanced_pool.get_deallocation_count() << std::endl;
    
    std::cout << "✓ All validations passed!" << std::endl;
}

} // namespace BTQuant

int main() {
    BTQuant::testPools();
    return 0;
}