#ifndef BTQ_HAZARD_POINTER_HPP
#define BTQ_HAZARD_POINTER_HPP

#include <atomic>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>
#include <functional>
#include <type_traits>

namespace btq {
namespace threading {

// Forward declaration
class HazardPointerManager;

/**
 * @brief A hazard pointer guard that protects a pointer from being deleted
 * 
 * This RAII class ensures that the protected pointer remains valid for the
 * duration of the guard's lifetime by registering it as a "hazard" that
 * prevents reclamation by other threads.
 */
template<typename T>
class HazardPointerGuard {
private:
    T* ptr_{nullptr};
    HazardPointerManager* manager_{nullptr};
    size_t index_{static_cast<size_t>(-1)};  // Index in the thread's hazard pointer array

public:
    /**
     * @brief Construct a hazard pointer guard
     * @param ptr The pointer to protect
     * @param manager The hazard pointer manager to register with
     */
    HazardPointerGuard(T* ptr, HazardPointerManager& manager);
    
    /**
     * @brief Destructor - releases the hazard pointer
     */
    ~HazardPointerGuard();
    
    /**
     * @brief Deleted copy constructor
     */
    HazardPointerGuard(const HazardPointerGuard&) = delete;
    
    /**
     * @brief Deleted assignment operator
     */
    HazardPointerGuard& operator=(const HazardPointerGuard&) = delete;
    
    /**
     * @brief Move constructor
     */
    HazardPointerGuard(HazardPointerGuard&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    HazardPointerGuard& operator=(HazardPointerGuard&& other) noexcept;
    
    /**
     * @brief Get the protected pointer
     * @return The protected pointer
     */
    T* get() const noexcept { return ptr_; }
    
    /**
     * @brief Dereference operator
     * @return Reference to the pointed-to object
     */
    T& operator*() const noexcept { return *ptr_; }
    
    /**
     * @brief Arrow operator
     * @return The protected pointer
     */
    T* operator->() const noexcept { return ptr_; }
    
    /**
     * @brief Check if the guard is protecting a valid pointer
     * @return True if protecting a valid pointer, false otherwise
     */
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
    
    /**
     * @brief Release the hazard pointer early
     */
    void release() noexcept;
};

/**
 * @brief Manager for hazard pointers - handles allocation and reclamation
 * 
 * This singleton-like class manages the pool of hazard pointers and the
 * reclamation of retired objects. Each thread should have its own set of
 * hazard pointers managed by this class.
 */
class HazardPointerManager {
private:
    // Maximum number of hazard pointers per thread
    static constexpr size_t MAX_HAZARD_POINTERS_PER_THREAD = 64;
    
    // Maximum number of retired objects before cleanup is forced
    static constexpr size_t MAX_RETIRED_BEFORE_CLEANUP = 1000;
    
    // Structure to hold a hazard pointer record for a thread
    struct HazardPointerRecord {
        std::atomic<void*> pointer{nullptr};
        std::atomic<bool> active{false};
        std::thread::id owner_thread_id{std::thread::id{}};
        
        HazardPointerRecord() = default;
        
        // Define move constructor and assignment operator since atomic members are not copyable
        HazardPointerRecord(HazardPointerRecord&& other) noexcept
            : pointer(other.pointer.load()), active(other.active.load()), owner_thread_id(other.owner_thread_id) {}
        
        HazardPointerRecord& operator=(HazardPointerRecord&& other) noexcept {
            if (this != &other) {
                pointer.store(other.pointer.load());
                active.store(other.active.load());
                owner_thread_id = other.owner_thread_id;
            }
            return *this;
        }
        
        // Delete copy constructor and assignment operator
        HazardPointerRecord(const HazardPointerRecord&) = delete;
        HazardPointerRecord& operator=(const HazardPointerRecord&) = delete;
    };
    
    // Structure to hold a retired object
    struct RetiredNode {
        void* ptr;
        std::function<void()> deleter;
        std::thread::id retiring_thread_id;
        RetiredNode* next;
        
        RetiredNode(void* p, std::function<void()> d, std::thread::id tid)
            : ptr(p), deleter(std::move(d)), retiring_thread_id(tid), next(nullptr) {}
    };
    
    // Global array of hazard pointer records
    alignas(64) std::vector<HazardPointerRecord> global_hazard_pointers_;
    
    // Head of the retired objects list
    alignas(64) std::atomic<RetiredNode*> retired_list_head_{nullptr};
    
    // Count of retired objects
    alignas(64) std::atomic<size_t> retired_count_{0};
    
    // Mutex for managing the global hazard pointer array
    mutable std::mutex global_mutex_;
    
    // Thread-local storage for this thread's hazard pointer indices
    static thread_local std::vector<size_t> thread_hazard_indices_;
    
    // Thread-local storage for this thread's hazard pointer records
    static thread_local std::vector<HazardPointerRecord*> thread_hazard_records_;
    
    /**
     * @brief Get or allocate a hazard pointer for this thread
     * @return Index of the allocated hazard pointer
     */
    size_t acquire_hazard_pointer_index();
    
    /**
     * @brief Release a hazard pointer back to the pool
     * @param index Index of the hazard pointer to release
     */
    void release_hazard_pointer_index(size_t index);
    
    /**
     * @brief Check if a pointer is currently protected by any hazard pointer
     * @param ptr Pointer to check
     * @return True if the pointer is protected, false otherwise
     */
    bool is_protected(void* ptr) const;
    
public:
    
    /**
     * @brief Clean up retired objects that are safe to delete
     */
    void cleanup_retired_objects();
    
    /**
     * @brief Helper function to delete a retired node
     * @param node Node to delete
     */
    void delete_retired_node(RetiredNode* node);

public:
    /**
     * @brief Constructor
     */
    HazardPointerManager();
    
    /**
     * @brief Destructor
     */
    ~HazardPointerManager();
    
    /**
     * @brief Deleted copy constructor
     */
    HazardPointerManager(const HazardPointerManager&) = delete;
    
    /**
     * @brief Deleted assignment operator
     */
    HazardPointerManager& operator=(const HazardPointerManager&) = delete;
    
    /**
     * @brief Get the singleton instance of the hazard pointer manager
     * @return Reference to the singleton instance
     */
    static HazardPointerManager& instance();
    
    /**
     * @brief Acquire a hazard pointer for the given pointer
     * @param ptr Pointer to protect
     * @return A hazard pointer guard that protects the pointer
     */
    template<typename T>
    HazardPointerGuard<T> acquire_hazard_pointer(T* ptr);
    
    /**
     * @brief Retire an object for deletion when safe
     * @param ptr Pointer to the object to retire
     * @param deleter Function to delete the object
     */
    template<typename T>
    void retire(T* ptr, std::function<void()> deleter = []() {});
    
    /**
     * @brief Force cleanup of retired objects
     */
    void force_cleanup();
    
    /**
     * @brief Get the number of retired objects waiting for cleanup
     * @return Number of retired objects
     */
    size_t get_retired_count() const;
    
    /**
     * @brief Get the number of active hazard pointers for this thread
     * @return Number of active hazard pointers
     */
    size_t get_active_hazard_count() const;
    
private:
    // Make HazardPointerGuard a friend class so it can access private members
    template<typename T>
    friend class HazardPointerGuard;
};

// Static thread-local definitions
thread_local std::vector<size_t> HazardPointerManager::thread_hazard_indices_;
thread_local std::vector<HazardPointerManager::HazardPointerRecord*> HazardPointerManager::thread_hazard_records_;

// Implementation of HazardPointerGuard methods
template<typename T>
HazardPointerGuard<T>::HazardPointerGuard(T* ptr, HazardPointerManager& manager)
    : ptr_(ptr), manager_(&manager), index_(static_cast<size_t>(-1)) {
    if (ptr != nullptr) {
        index_ = manager_->acquire_hazard_pointer_index();
        manager_->global_hazard_pointers_[index_].pointer.store(ptr, std::memory_order_release);
        manager_->global_hazard_pointers_[index_].active.store(true, std::memory_order_release);
    }
}

template<typename T>
HazardPointerGuard<T>::~HazardPointerGuard() {
    if (ptr_ != nullptr && manager_ != nullptr && index_ != static_cast<size_t>(-1)) {
        manager_->global_hazard_pointers_[index_].pointer.store(nullptr, std::memory_order_release);
        manager_->global_hazard_pointers_[index_].active.store(false, std::memory_order_release);
        manager_->release_hazard_pointer_index(index_);
    }
}

template<typename T>
HazardPointerGuard<T>::HazardPointerGuard(HazardPointerGuard&& other) noexcept
    : ptr_(other.ptr_), manager_(other.manager_), index_(other.index_) {
    other.ptr_ = nullptr;
    other.manager_ = nullptr;
    other.index_ = static_cast<size_t>(-1);
}

template<typename T>
HazardPointerGuard<T>& HazardPointerGuard<T>::operator=(HazardPointerGuard&& other) noexcept {
    if (this != &other) {
        // Release current hazard pointer if active
        if (ptr_ != nullptr && manager_ != nullptr && index_ != static_cast<size_t>(-1)) {
            manager_->global_hazard_pointers_[index_].pointer.store(nullptr, std::memory_order_release);
            manager_->global_hazard_pointers_[index_].active.store(false, std::memory_order_release);
            manager_->release_hazard_pointer_index(index_);
        }
        
        // Transfer ownership
        ptr_ = other.ptr_;
        manager_ = other.manager_;
        index_ = other.index_;
        
        // Reset other
        other.ptr_ = nullptr;
        other.manager_ = nullptr;
        other.index_ = static_cast<size_t>(-1);
    }
    return *this;
}

template<typename T>
void HazardPointerGuard<T>::release() noexcept {
    if (ptr_ != nullptr && manager_ != nullptr && index_ != static_cast<size_t>(-1)) {
        manager_->global_hazard_pointers_[index_].pointer.store(nullptr, std::memory_order_release);
        manager_->global_hazard_pointers_[index_].active.store(false, std::memory_order_release);
        manager_->release_hazard_pointer_index(index_);
        
        ptr_ = nullptr;
        manager_ = nullptr;
        index_ = static_cast<size_t>(-1);
    }
}

// Implementation of HazardPointerManager methods
inline HazardPointerManager::HazardPointerManager() {
    // Pre-allocate a reasonable number of hazard pointer records
    global_hazard_pointers_.resize(1024);  // Start with 1024, will grow as needed
}

inline HazardPointerManager::~HazardPointerManager() {
    // Clean up any remaining retired objects
    force_cleanup();
    
    // Ensure all hazard pointers are released
    for (auto& record : global_hazard_pointers_) {
        if (record.active.load(std::memory_order_acquire)) {
            // This indicates a programming error - hazard pointers not properly released
            // In a real implementation, you might want to log this
        }
    }
}

inline HazardPointerManager& HazardPointerManager::instance() {
    static HazardPointerManager instance;
    return instance;
}

inline size_t HazardPointerManager::acquire_hazard_pointer_index() {
    // First, check if we have a free hazard pointer in our thread-local cache
    for (size_t i = 0; i < thread_hazard_records_.size(); ++i) {
        if (!thread_hazard_records_[i]->active.load(std::memory_order_acquire)) {
            thread_hazard_records_[i]->active.store(true, std::memory_order_release);
            return thread_hazard_indices_[i];
        }
    }
    
    // No free hazard pointer in our cache, need to find one globally
    std::lock_guard<std::mutex> lock(global_mutex_);
    
    // Look for an inactive hazard pointer
    for (size_t i = 0; i < global_hazard_pointers_.size(); ++i) {
        auto& record = global_hazard_pointers_[i];
        if (!record.active.load(std::memory_order_acquire)) {
            record.owner_thread_id = std::this_thread::get_id();
            record.active.store(true, std::memory_order_release);
            
            // Add to thread-local cache
            thread_hazard_indices_.push_back(i);
            thread_hazard_records_.push_back(&record);
            
            return i;
        }
    }
    
    // No free hazard pointer found, expand the array
    size_t new_index = global_hazard_pointers_.size();
    global_hazard_pointers_.emplace_back();
    auto& new_record = global_hazard_pointers_[new_index];
    
    new_record.owner_thread_id = std::this_thread::get_id();
    new_record.active.store(true, std::memory_order_release);
    
    // Add to thread-local cache
    thread_hazard_indices_.push_back(new_index);
    thread_hazard_records_.push_back(&new_record);
    
    return new_index;
}

inline void HazardPointerManager::release_hazard_pointer_index(size_t index) {
    if (index >= global_hazard_pointers_.size()) {
        return; // Invalid index
    }
    
    auto& record = global_hazard_pointers_[index];
    record.active.store(false, std::memory_order_release);
    record.pointer.store(nullptr, std::memory_order_release);
    
    // Remove from thread-local cache if present
    for (size_t i = 0; i < thread_hazard_indices_.size(); ++i) {
        if (thread_hazard_indices_[i] == index) {
            thread_hazard_indices_.erase(thread_hazard_indices_.begin() + i);
            thread_hazard_records_.erase(thread_hazard_records_.begin() + i);
            break;
        }
    }
}

inline bool HazardPointerManager::is_protected(void* ptr) const {
    for (const auto& record : global_hazard_pointers_) {
        if (record.active.load(std::memory_order_acquire) && 
            record.pointer.load(std::memory_order_acquire) == ptr) {
            return true;
        }
    }
    return false;
}

template<typename T>
HazardPointerGuard<T> HazardPointerManager::acquire_hazard_pointer(T* ptr) {
    return HazardPointerGuard<T>(ptr, *this);
}

template<typename T>
void HazardPointerManager::retire(T* ptr, std::function<void()> deleter) {
    if (ptr == nullptr) {
        return; // Nothing to retire
    }
    
    // Create a deleter that properly deletes the object
    std::function<void()> actual_deleter = [ptr, del = std::move(deleter)]() {
        if (del) {
            del();
        } else {
            delete ptr;
        }
    };
    
    // Create a new retired node
    RetiredNode* new_node = new RetiredNode(ptr, std::move(actual_deleter), std::this_thread::get_id());
    
    // Atomically add to the retired list
    RetiredNode* old_head = retired_list_head_.load(std::memory_order_acquire);
    do {
        new_node->next = old_head;
    } while (!retired_list_head_.compare_exchange_weak(old_head, new_node, 
                                                       std::memory_order_acq_rel, 
                                                       std::memory_order_acquire));
                                                       
    // Increment retired count
    retired_count_.fetch_add(1, std::memory_order_acq_rel);
    
    // If we have too many retired objects, try to clean up
    if (retired_count_.load(std::memory_order_acquire) >= MAX_RETIRED_BEFORE_CLEANUP) {
        cleanup_retired_objects();
    }
}

inline void HazardPointerManager::cleanup_retired_objects() {
    // Get the current retired list
    RetiredNode* current_list = retired_list_head_.load(std::memory_order_acquire);
    if (current_list == nullptr) {
        return; // Nothing to clean up
    }
    
    // Atomically take the entire list by setting head to null
    RetiredNode* taken_list = retired_list_head_.exchange(nullptr, std::memory_order_acq_rel);
    if (taken_list == nullptr) {
        return; // Another thread took the list
    }
    
    // Count how many nodes we took
    size_t taken_count = 0;
    RetiredNode* temp = taken_list;
    while (temp) {
        temp = temp->next;
        ++taken_count;
    }
    
    // Update the retired count
    retired_count_.fetch_sub(taken_count, std::memory_order_acq_rel);
    
    // Separate nodes that can be deleted from those that still need protection
    RetiredNode* deletable_list = nullptr;
    RetiredNode* protected_list = nullptr;
    RetiredNode* remaining_list = taken_list;
    
    while (remaining_list != nullptr) {
        RetiredNode* current = remaining_list;
        remaining_list = remaining_list->next;
        current->next = nullptr;
        
        // Check if this pointer is currently protected by any hazard pointer
        if (is_protected(current->ptr)) {
            // Still protected, add to protected list
            current->next = protected_list;
            protected_list = current;
        } else {
            // Safe to delete, add to deletable list
            current->next = deletable_list;
            deletable_list = current;
        }
    }
    
    // Re-add protected nodes to the main retired list
    if (protected_list != nullptr) {
        RetiredNode* old_head = retired_list_head_.load(std::memory_order_acquire);
        RetiredNode* protected_tail = protected_list;
        while (protected_tail->next) {
            protected_tail = protected_tail->next;
        }
        protected_tail->next = old_head;
        
        // Try to add back to the main list
        RetiredNode* expected = nullptr;
        if (!retired_list_head_.compare_exchange_strong(expected, protected_list, 
                                                       std::memory_order_acq_rel, 
                                                       std::memory_order_acquire)) {
            // Another thread added to the list, merge with theirs
            protected_tail->next = expected;
            retired_list_head_.store(protected_list, std::memory_order_release);
        }
    }
    
    // Actually delete the safe nodes
    while (deletable_list != nullptr) {
        RetiredNode* to_delete = deletable_list;
        deletable_list = deletable_list->next;
        
        // Call the deleter function
        to_delete->deleter();
        delete to_delete;
    }
}

inline void HazardPointerManager::delete_retired_node(RetiredNode* node) {
    if (node) {
        node->deleter();
        delete node;
    }
}

inline void HazardPointerManager::force_cleanup() {
    // Keep cleaning up until the retired list is empty
    size_t attempts = 0;
    const size_t max_attempts = 10; // Prevent infinite loops
    
    while (retired_count_.load(std::memory_order_acquire) > 0 && attempts < max_attempts) {
        cleanup_retired_objects();
        ++attempts;
    }
}

inline size_t HazardPointerManager::get_retired_count() const {
    return retired_count_.load(std::memory_order_acquire);
}

inline size_t HazardPointerManager::get_active_hazard_count() const {
    size_t count = 0;
    for (const auto& record : global_hazard_pointers_) {
        if (record.active.load(std::memory_order_acquire)) {
            ++count;
        }
    }
    return count;
}

} // namespace threading
} // namespace btq

#endif // BTQ_HAZARD_POINTER_HPP