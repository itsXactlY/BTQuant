#pragma once

/**
 * @file hazard_pointer.hpp
 * @brief Lock-free Hazard Pointer Memory Reclamation for Trading Terminal
 * 
 * This implementation provides safe memory reclamation for lock-free data structures.
 * It prevents use-after-free bugs when multiple threads access shared data.
 * 
 * Key features:
 * - O(1) acquire/release operations
 * - Thread-local hazard pointer records
 * - Batch retirement for efficiency
 * - Automatic memory reclamation when safe
 * 
 * Reference: "Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"
 * by Maged M. Michael (2004)
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <array>
#include <cassert>

namespace btq {
namespace memory {

// Configuration constants
constexpr size_t HAZARD_POINTER_COUNT = 64;        // Max concurrent hazard pointers
constexpr size_t RETIREMENT_LIST_SIZE = 1024;      // Max retired pointers before scan
constexpr size_t HAZARD_POINTER_THRESHOLD = 2;     // Scan when retired > HP count * threshold

/**
 * @brief Hazard Pointer Record - one per thread that uses lock-free structures
 */
struct HazardPointerRecord {
    std::atomic<void*> pointer{nullptr};           // The protected pointer
    std::atomic<bool> active{false};               // Whether this record is in use
    std::thread::id owner_thread{};                // Thread that owns this record
    
    HazardPointerRecord() = default;
    
    // Non-copyable, non-movable
    HazardPointerRecord(const HazardPointerRecord&) = delete;
    HazardPointerRecord& operator=(const HazardPointerRecord&) = delete;
};

/**
 * @brief Global hazard pointer domain for the trading terminal
 * 
 * This class manages all hazard pointers across threads and handles
 * safe memory reclamation for lock-free data structures.
 */
class HazardPointerDomain {
public:
    static HazardPointerDomain& getInstance() {
        static HazardPointerDomain instance;
        return instance;
    }
    
    /**
     * @brief Acquire a hazard pointer for the current thread
     * @return Index of the acquired hazard pointer record
     */
    size_t acquireHazardPointer() {
        std::thread::id tid = std::this_thread::get_id();
        
        // First, try to find an existing record for this thread
        for (size_t i = 0; i < HAZARD_POINTER_COUNT; ++i) {
            bool expected = false;
            if (records_[i].active.compare_exchange_strong(expected, true,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) {
                records_[i].owner_thread = tid;
                return i;
            }
        }
        
        // All records are in use - this is a critical error in production
        // In a real system, we might want to spin-wait or allocate more
        assert(false && "No available hazard pointer records");
        return SIZE_MAX;
    }
    
    /**
     * @brief Release a hazard pointer
     * @param index The hazard pointer index to release
     */
    void releaseHazardPointer(size_t index) {
        if (index < HAZARD_POINTER_COUNT) {
            records_[index].pointer.store(nullptr, std::memory_order_release);
            records_[index].active.store(false, std::memory_order_release);
        }
    }
    
    /**
     * @brief Protect a pointer with a hazard pointer
     * @param index The hazard pointer index
     * @param ptr The pointer to protect
     */
    void protectPointer(size_t index, void* ptr) {
        if (index < HAZARD_POINTER_COUNT) {
            records_[index].pointer.store(ptr, std::memory_order_release);
        }
    }
    
    /**
     * @brief Clear the hazard pointer (stop protecting)
     * @param index The hazard pointer index
     */
    void clearHazardPointer(size_t index) {
        if (index < HAZARD_POINTER_COUNT) {
            records_[index].pointer.store(nullptr, std::memory_order_release);
        }
    }
    
    /**
     * @brief Check if a pointer is protected by any hazard pointer
     * @param ptr The pointer to check
     * @return true if the pointer is protected
     */
    bool isProtected(void* ptr) const {
        for (size_t i = 0; i < HAZARD_POINTER_COUNT; ++i) {
            if (records_[i].pointer.load(std::memory_order_acquire) == ptr) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Retire a pointer for later reclamation
     * @param ptr The pointer to retire
     * @param deleter The deleter function to call when safe
     */
    void retirePointer(void* ptr, std::function<void(void*)> deleter) {
        std::lock_guard<std::mutex> lock(retirement_mutex_);
        
        retired_pointers_.push_back({ptr, deleter});
        
        // Scan if we have too many retired pointers
        if (retired_pointers_.size() > HAZARD_POINTER_COUNT * HAZARD_POINTER_THRESHOLD) {
            scanRetiredPointers();
        }
    }
    
    /**
     * @brief Force a scan of retired pointers
     */
    void scanRetiredPointers() {
        // Collect all currently protected pointers
        std::vector<void*> protected_ptrs;
        protected_ptrs.reserve(HAZARD_POINTER_COUNT);
        
        for (size_t i = 0; i < HAZARD_POINTER_COUNT; ++i) {
            void* ptr = records_[i].pointer.load(std::memory_order_acquire);
            if (ptr != nullptr) {
                protected_ptrs.push_back(ptr);
            }
        }
        
        // Sort for binary search
        std::sort(protected_ptrs.begin(), protected_ptrs.end());
        
        // Reclaim unprotected retired pointers
        auto it = retired_pointers_.begin();
        while (it != retired_pointers_.end()) {
            if (!std::binary_search(protected_ptrs.begin(), protected_ptrs.end(), it->ptr)) {
                // Safe to delete
                if (it->deleter) {
                    it->deleter(it->ptr);
                }
                it = retired_pointers_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    /**
     * @brief Get statistics about hazard pointer usage
     */
    struct Stats {
        size_t active_count;
        size_t retired_count;
    };
    
    Stats getStats() const {
        Stats stats{0, 0};
        
        for (size_t i = 0; i < HAZARD_POINTER_COUNT; ++i) {
            if (records_[i].active.load(std::memory_order_relaxed)) {
                ++stats.active_count;
            }
        }
        
        stats.retired_count = retired_pointers_.size();
        return stats;
    }

private:
    HazardPointerDomain() = default;
    ~HazardPointerDomain() {
        // Clean up any remaining retired pointers
        for (auto& rp : retired_pointers_) {
            if (rp.deleter) {
                rp.deleter(rp.ptr);
            }
        }
    }
    
    // Non-copyable, non-movable
    HazardPointerDomain(const HazardPointerDomain&) = delete;
    HazardPointerDomain& operator=(const HazardPointerDomain&) = delete;
    
    // Hazard pointer records
    std::array<HazardPointerRecord, HAZARD_POINTER_COUNT> records_;
    
    // Retired pointers waiting for reclamation
    struct RetiredPointer {
        void* ptr;
        std::function<void(void*)> deleter;
    };
    
    std::vector<RetiredPointer> retired_pointers_;
    mutable std::mutex retirement_mutex_;
};

/**
 * @brief RAII wrapper for hazard pointer usage
 * 
 * Usage:
 *   HazardPointerGuard guard;
 *   guard.protect(some_lockfree_node);
 *   // ... access the node safely ...
 *   guard.clear();  // Optional - automatic on destruction
 */
class HazardPointerGuard {
public:
    HazardPointerGuard()
        : domain_(HazardPointerDomain::getInstance())
        , index_(domain_.acquireHazardPointer())
        , protected_ptr_(nullptr)
    {}
    
    ~HazardPointerGuard() {
        if (index_ != SIZE_MAX) {
            domain_.clearHazardPointer(index_);
            domain_.releaseHazardPointer(index_);
        }
    }
    
    // Non-copyable, movable
    HazardPointerGuard(const HazardPointerGuard&) = delete;
    HazardPointerGuard& operator=(const HazardPointerGuard&) = delete;
    
    HazardPointerGuard(HazardPointerGuard&& other) noexcept
        : domain_(other.domain_)
        , index_(other.index_)
        , protected_ptr_(other.protected_ptr_)
    {
        other.index_ = SIZE_MAX;
        other.protected_ptr_ = nullptr;
    }
    
    HazardPointerGuard& operator=(HazardPointerGuard&& other) noexcept {
        if (this != &other) {
            if (index_ != SIZE_MAX) {
                domain_.releaseHazardPointer(index_);
            }
            index_ = other.index_;
            protected_ptr_ = other.protected_ptr_;
            other.index_ = SIZE_MAX;
            other.protected_ptr_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Protect a pointer
     * @tparam T The type of the pointer
     * @param ptr The pointer to protect
     * @return The same pointer for convenience
     */
    template<typename T>
    T* protect(T* ptr) {
        protected_ptr_ = static_cast<void*>(ptr);
        domain_.protectPointer(index_, protected_ptr_);
        return ptr;
    }
    
    /**
     * @brief Clear the protection
     */
    void clear() {
        domain_.clearHazardPointer(index_);
        protected_ptr_ = nullptr;
    }
    
    /**
     * @brief Check if the guard is valid
     */
    bool valid() const {
        return index_ != SIZE_MAX;
    }

private:
    HazardPointerDomain& domain_;
    size_t index_;
    void* protected_ptr_;
};

/**
 * @brief Multi-pointer hazard pointer guard (for protecting multiple pointers)
 * 
 * Use this when you need to protect multiple pointers simultaneously,
 * such as when traversing a linked list or tree.
 */
template<size_t N>
class MultiHazardPointerGuard {
public:
    MultiHazardPointerGuard()
        : domain_(HazardPointerDomain::getInstance())
    {
        for (size_t i = 0; i < N; ++i) {
            indices_[i] = domain_.acquireHazardPointer();
        }
    }
    
    ~MultiHazardPointerGuard() {
        for (size_t i = 0; i < N; ++i) {
            if (indices_[i] != SIZE_MAX) {
                domain_.clearHazardPointer(indices_[i]);
                domain_.releaseHazardPointer(indices_[i]);
            }
        }
    }
    
    // Non-copyable, non-movable
    MultiHazardPointerGuard(const MultiHazardPointerGuard&) = delete;
    MultiHazardPointerGuard& operator=(const MultiHazardPointerGuard&) = delete;
    MultiHazardPointerGuard(MultiHazardPointerGuard&&) = delete;
    MultiHazardPointerGuard& operator=(MultiHazardPointerGuard&&) = delete;
    
    /**
     * @brief Protect a pointer at the given slot
     * @tparam T The type of the pointer
     * @param slot The slot index (0 to N-1)
     * @param ptr The pointer to protect
     * @return The same pointer for convenience
     */
    template<typename T>
    T* protect(size_t slot, T* ptr) {
        if (slot < N && indices_[slot] != SIZE_MAX) {
            domain_.protectPointer(indices_[slot], static_cast<void*>(ptr));
        }
        return ptr;
    }
    
    /**
     * @brief Clear protection at the given slot
     * @param slot The slot index (0 to N-1)
     */
    void clear(size_t slot) {
        if (slot < N && indices_[slot] != SIZE_MAX) {
            domain_.clearHazardPointer(indices_[slot]);
        }
    }
    
    /**
     * @brief Clear all protections
     */
    void clearAll() {
        for (size_t i = 0; i < N; ++i) {
            clear(i);
        }
    }

private:
    HazardPointerDomain& domain_;
    std::array<size_t, N> indices_;
};

/**
 * @brief Retire a pointer with automatic type deduction
 * @tparam T The type of the pointer
 * @param ptr The pointer to retire
 */
template<typename T>
void retirePointer(T* ptr) {
    HazardPointerDomain::getInstance().retirePointer(
        static_cast<void*>(ptr),
        [](void* p) { delete static_cast<T*>(p); }
    );
}

/**
 * @brief Retire a pointer with custom deleter
 * @tparam T The type of the pointer
 * @tparam Deleter The deleter type
 * @param ptr The pointer to retire
 * @param deleter The custom deleter
 */
template<typename T, typename Deleter>
void retirePointer(T* ptr, Deleter deleter) {
    HazardPointerDomain::getInstance().retirePointer(
        static_cast<void*>(ptr),
        [deleter](void* p) { deleter(static_cast<T*>(p)); }
    );
}

} // namespace memory
} // namespace btq
