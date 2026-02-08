#ifndef BTQ_HAZARD_POINTER_HPP
#define BTQ_HAZARD_POINTER_HPP

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>

namespace btq {

// Forward declarations
class hazard_pointer;
template<class T, class D = std::default_delete<T>>
class hazard_pointer_obj_base;

namespace detail {
    // Internal implementation details for hazard pointers
    class hazard_pointer_manager {
    private:
        static constexpr size_t MAX_HAZARD_POINTERS = 1024;
        static constexpr size_t MAX_RETIRED_OBJECTS = 1000;

        struct hazard_record {
            std::atomic<void*> ptr{nullptr};
            std::atomic<bool> active{false};
            std::thread::id owner_tid{};

            hazard_record() = default;
        };

        struct retired_node {
            void* ptr;
            std::function<void()> deleter;
            std::thread::id retiring_tid;
            retired_node* next;

            retired_node(void* p, std::function<void()> d, std::thread::id tid)
                : ptr(p), deleter(std::move(d)), retiring_tid(tid), next(nullptr) {}
        };

        alignas(64) std::vector<hazard_record> global_hazard_ptrs_;
        alignas(64) std::atomic<retired_node*> retired_list_head_{nullptr};
        alignas(64) std::atomic<size_t> retired_count_{0};
        
        mutable std::mutex global_mtx_;

        // Thread-local storage
        static thread_local std::vector<size_t> thread_hazard_indices_;

    public:
        hazard_pointer_manager() {
            global_hazard_ptrs_.resize(MAX_HAZARD_POINTERS);
        }

        static hazard_pointer_manager& instance() {
            static hazard_pointer_manager mgr;
            return mgr;
        }

        size_t acquire_hazard_ptr_idx() {
            std::lock_guard<std::mutex> lock(global_mtx_);
            
            for (size_t i = 0; i < global_hazard_ptrs_.size(); ++i) {
                auto& rec = global_hazard_ptrs_[i];
                if (!rec.active.exchange(true, std::memory_order_acquire)) {
                    rec.owner_tid = std::this_thread::get_id();
                    thread_hazard_indices_.push_back(i);
                    return i;
                }
            }
            
            // Expand if needed
            size_t new_idx = global_hazard_ptrs_.size();
            if (new_idx < MAX_HAZARD_POINTERS) {
                global_hazard_ptrs_.emplace_back();
                auto& new_rec = global_hazard_ptrs_[new_idx];
                new_rec.active.store(true, std::memory_order_release);
                new_rec.owner_tid = std::this_thread::get_id();
                thread_hazard_indices_.push_back(new_idx);
                return new_idx;
            }
            
            throw std::runtime_error("Maximum hazard pointers exceeded");
        }

        void release_hazard_ptr_idx(size_t idx) {
            if (idx < global_hazard_ptrs_.size()) {
                auto& rec = global_hazard_ptrs_[idx];
                rec.active.store(false, std::memory_order_release);
                
                // Remove from thread-local cache
                for (auto it = thread_hazard_indices_.begin(); it != thread_hazard_indices_.end(); ++it) {
                    if (*it == idx) {
                        thread_hazard_indices_.erase(it);
                        break;
                    }
                }
            }
        }

        bool is_protected(void* ptr) const {
            for (const auto& rec : global_hazard_ptrs_) {
                if (rec.active.load(std::memory_order_acquire) && 
                    rec.ptr.load(std::memory_order_acquire) == ptr) {
                    return true;
                }
            }
            return false;
        }

        template<typename T, typename Deleter = std::default_delete<T>>
        void retire_object(T* ptr, Deleter d = Deleter{}) {
            if (!ptr) return;

            auto deleter = [ptr, d]() { d(ptr); };
            auto* node = new retired_node(ptr, std::move(deleter), std::this_thread::get_id());

            retired_node* old_head = retired_list_head_.load(std::memory_order_acquire);
            do {
                node->next = old_head;
            } while (!retired_list_head_.compare_exchange_weak(
                         old_head, node, std::memory_order_acq_rel, std::memory_order_acquire));

            retired_count_.fetch_add(1, std::memory_order_acq_rel);

            if (retired_count_.load(std::memory_order_acquire) >= MAX_RETIRED_OBJECTS) {
                cleanup_retired();
            }
        }

        void cleanup_retired() {
            retired_node* current_list = retired_list_head_.load(std::memory_order_acquire);
            if (!current_list) return;

            retired_node* taken_list = retired_list_head_.exchange(nullptr, std::memory_order_acq_rel);
            if (!taken_list) return;

            size_t taken_count = 0;
            for (auto* temp = taken_list; temp; temp = temp->next) {
                ++taken_count;
            }
            retired_count_.fetch_sub(taken_count, std::memory_order_acq_rel);

            retired_node* deletable_list = nullptr;
            retired_node* protected_list = nullptr;
            retired_node* remaining = taken_list;

            while (remaining) {
                auto* curr = remaining;
                remaining = remaining->next;
                curr->next = nullptr;

                if (is_protected(curr->ptr)) {
                    curr->next = protected_list;
                    protected_list = curr;
                } else {
                    curr->next = deletable_list;
                    deletable_list = curr;
                }
            }

            if (protected_list) {
                retired_node* old_head = retired_list_head_.load(std::memory_order_acquire);
                retired_node* tail = protected_list;
                while (tail->next) tail = tail->next;
                tail->next = old_head;

                retired_list_head_.store(protected_list, std::memory_order_release);
            }

            while (deletable_list) {
                auto* to_del = deletable_list;
                deletable_list = deletable_list->next;
                to_del->deleter();
                delete to_del;
            }
        }
    };

    thread_local std::vector<size_t> hazard_pointer_manager::thread_hazard_indices_;
}

/**
 * @brief Base class for objects that can be protected by hazard pointers
 * 
 * This class provides the retire() method to mark objects for deferred deletion.
 * 
 * @tparam T Type of the object
 * @tparam D Deleter type (defaults to std::default_delete<T>)
 */
template<class T, class D>
class hazard_pointer_obj_base {
public:
    /**
     * @brief Mark this object for retirement (deferred deletion)
     * 
     * @param d Custom deleter (defaults to D{})
     */
    void retire(D d = D{}) noexcept {
        auto* mgr = &detail::hazard_pointer_manager::instance();
        mgr->retire_object(static_cast<T*>(this), std::move(d));
    }

protected:
    hazard_pointer_obj_base() = default;
    hazard_pointer_obj_base(const hazard_pointer_obj_base&) = default;
    hazard_pointer_obj_base(hazard_pointer_obj_base&&) = default;
    hazard_pointer_obj_base& operator=(const hazard_pointer_obj_base&) = default;
    hazard_pointer_obj_base& operator=(hazard_pointer_obj_base&&) = default;
    ~hazard_pointer_obj_base() = default;
};

// Specialization for default deleter
template<class T>
class hazard_pointer_obj_base<T, std::default_delete<T>> {
public:
    /**
     * @brief Mark this object for retirement (deferred deletion)
     */
    void retire() noexcept {
        auto* mgr = &detail::hazard_pointer_manager::instance();
        mgr->retire_object(static_cast<T*>(this));
    }

protected:
    hazard_pointer_obj_base() = default;
    hazard_pointer_obj_base(const hazard_pointer_obj_base&) = default;
    hazard_pointer_obj_base(hazard_pointer_obj_base&&) = default;
    hazard_pointer_obj_base& operator=(const hazard_pointer_obj_base&) = default;
    hazard_pointer_obj_base& operator=(hazard_pointer_obj_base&&) = default;
    ~hazard_pointer_obj_base() = default;
};

/**
 * @brief A hazard pointer that can protect objects from deletion
 * 
 * A hazard pointer is a single-writer multi-reader pointer that can be owned 
 * by at most one thread at any time. It provides methods to protect objects
 * from being reclaimed while they are in use.
 */
class hazard_pointer {
private:
    size_t index_;
    bool active_;

public:
    /**
     * @brief Construct a hazard pointer
     */
    hazard_pointer() : index_(static_cast<size_t>(-1)), active_(false) {
        acquire();
    }

    /**
     * @brief Move constructor
     */
    hazard_pointer(hazard_pointer&& other) noexcept 
        : index_(other.index_), active_(other.active_) {
        other.active_ = false;
    }

    /**
     * @brief Move assignment operator
     */
    hazard_pointer& operator=(hazard_pointer&& other) noexcept {
        if (this != &other) {
            if (active_) {
                release();
            }
            index_ = other.index_;
            active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }

    /**
     * @brief Destructor
     */
    ~hazard_pointer() {
        if (active_) {
            release();
        }
    }

    /**
     * @brief Check if the hazard pointer is empty (not protecting anything)
     * @return true if not protecting any object, false otherwise
     */
    bool empty() const noexcept {
        return !active_ || get_current_ptr() == nullptr;
    }

    /**
     * @brief Protect the object pointed to by src
     * @tparam T Type of the object
     * @param src Atomic pointer to protect
     * @return Pointer to the protected object
     */
    template<class T>
    T* protect(const std::atomic<T*>& src) noexcept {
        if (!active_) {
            acquire();
        }
        
        T* ptr = src.load(std::memory_order_acquire);
        set_ptr(ptr);
        // Double-check that the pointer is still valid after setting the hazard
        T* current = src.load(std::memory_order_acquire);
        if (ptr != current) {
            ptr = current;
            set_ptr(ptr);
        }
        return ptr;
    }

    /**
     * @brief Try to protect the object pointed to by src
     * @tparam T Type of the object
     * @param[out] ptr Reference to store the protected pointer
     * @param src Atomic pointer to protect
     * @return true if protection was successful, false otherwise
     */
    template<class T>
    bool try_protect(T*& ptr, const std::atomic<T*>& src) noexcept {
        if (!active_) {
            acquire();
        }
        
        T* current = src.load(std::memory_order_acquire);
        set_ptr(current);
        // Double-check
        T* reloaded = src.load(std::memory_order_acquire);
        if (current == reloaded) {
            ptr = current;
            return true;
        } else {
            set_ptr(reloaded);
            ptr = reloaded;
            return false;
        }
    }

    /**
     * @brief Reset protection for the current object
     * @tparam T Type of the object
     * @param ptr Pointer to the object to stop protecting (optional)
     */
    template<class T>
    void reset_protection(const T* ptr = nullptr) noexcept {
        if (active_ && (ptr == nullptr || get_current_ptr() == ptr)) {
            set_ptr(nullptr);
        }
    }

    /**
     * @brief Reset protection (clear the hazard pointer)
     */
    void reset_protection(std::nullptr_t = nullptr) noexcept {
        if (active_) {
            set_ptr(nullptr);
        }
    }

    /**
     * @brief Swap this hazard pointer with another
     * @param other The other hazard pointer to swap with
     */
    void swap(hazard_pointer& other) noexcept {
        std::swap(this->index_, other.index_);
        std::swap(this->active_, other.active_);
    }

private:
    void acquire() {
        auto& mgr = detail::hazard_pointer_manager::instance();
        index_ = mgr.acquire_hazard_ptr_idx();
        active_ = true;
    }

    void release() {
        if (active_ && index_ != static_cast<size_t>(-1)) {
            auto& mgr = detail::hazard_pointer_manager::instance();
            mgr.release_hazard_ptr_idx(index_);
            active_ = false;
        }
    }

    void* get_current_ptr() const {
        if (!active_ || index_ >= detail::hazard_pointer_manager::instance().global_hazard_ptrs_.size()) {
            return nullptr;
        }
        return detail::hazard_pointer_manager::instance().global_hazard_ptrs_[index_].ptr.load(std::memory_order_acquire);
    }

    void set_ptr(void* ptr) {
        if (active_ && index_ < detail::hazard_pointer_manager::instance().global_hazard_ptrs_.size()) {
            detail::hazard_pointer_manager::instance().global_hazard_ptrs_[index_].ptr.store(ptr, std::memory_order_release);
        }
    }
};

/**
 * @brief Create a new hazard pointer
 * @return A newly constructed hazard pointer
 */
inline hazard_pointer make_hazard_pointer() {
    return hazard_pointer{};
}

/**
 * @brief Swap two hazard pointers
 * @param lhs First hazard pointer
 * @param rhs Second hazard pointer
 */
inline void swap(hazard_pointer& lhs, hazard_pointer& rhs) noexcept {
    lhs.swap(rhs);
}

} // namespace btq

#endif // BTQ_HAZARD_POINTER_HPP