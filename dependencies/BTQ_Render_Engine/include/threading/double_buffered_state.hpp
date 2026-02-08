#pragma once

#include <atomic>
#include <memory>
#include <utility>

namespace btq {
namespace threading {

/**
 * DoubleBufferedState - Implements a lock-free double buffering mechanism
 * 
 * This class provides a thread-safe mechanism where:
 * - Writer thread fills the "back" buffer
 * - Reader thread accesses the "front" buffer
 * - When writer completes, buffers are atomically swapped
 * 
 * This eliminates contention between reader and writer threads.
 */
template<typename T>
class DoubleBufferedState {
public:
    /**
     * Constructor
     * @param initial_state Initial state for both front and back buffers
     */
    explicit DoubleBufferedState(const T& initial_state = T{})
        : front_buffer_(std::make_unique<T>(initial_state))
        , back_buffer_(std::make_unique<T>(initial_state))
        , front_ptr_(&front_buffer_)
        , back_ptr_(&back_buffer_)
    {
    }

    /**
     * Constructor with move semantics
     * @param initial_state Initial state for both front and back buffers
     */
    explicit DoubleBufferedState(T&& initial_state)
        : front_buffer_(std::make_unique<T>(std::forward<T>(initial_state)))
        , back_buffer_(std::make_unique<T>(*front_buffer_))
        , front_ptr_(&front_buffer_)
        , back_ptr_(&back_buffer_)
    {
    }

    // Copy and move operations are allowed for the class to be usable in structs
    // The internal state is read-only for copies, so it's safe
    DoubleBufferedState(const DoubleBufferedState& other)
        : front_buffer_(std::make_unique<T>(*(other.front_buffer_)))
        , back_buffer_(std::make_unique<T>(*(other.back_buffer_)))
        , front_ptr_(&front_buffer_)
        , back_ptr_(&back_buffer_)
    {
    }

    DoubleBufferedState& operator=(const DoubleBufferedState& other) {
        if (this != &other) {
            *front_buffer_ = *(other.front_buffer_);
            *back_buffer_ = *(other.back_buffer_);
        }
        return *this;
    }

    DoubleBufferedState(DoubleBufferedState&& other) noexcept
        : front_buffer_(std::move(other.front_buffer_))
        , back_buffer_(std::move(other.back_buffer_))
        , front_ptr_(&front_buffer_)
        , back_ptr_(&back_buffer_)
    {
        // Update pointers to point to our own buffers after the move
        other.front_buffer_ = std::make_unique<T>();
        other.back_buffer_ = std::make_unique<T>();
        other.front_ptr_ = &(other.front_buffer_);
        other.back_ptr_ = &(other.back_buffer_);
    }

    DoubleBufferedState& operator=(DoubleBufferedState&& other) noexcept {
        if (this != &other) {
            front_buffer_ = std::move(other.front_buffer_);
            back_buffer_ = std::move(other.back_buffer_);
            
            // Update pointers to point to our own buffers after the move
            other.front_buffer_ = std::make_unique<T>();
            other.back_buffer_ = std::make_unique<T>();
            other.front_ptr_ = &(other.front_buffer_);
            other.back_ptr_ = &(other.back_buffer_);
        }
        return *this;
    }

    /**
     * Get current front buffer for reading (non-modifying)
     * Safe to call from reader thread
     * @return Const reference to the front buffer
     */
    const T& read() const noexcept {
        auto ptr = front_ptr_.load(std::memory_order_acquire);
        return *(*ptr);
    }

    /**
     * Get back buffer for writing
     * Safe to call from writer thread
     * @return Reference to the back buffer
     */
    T& write() noexcept {
        auto ptr = back_ptr_.load(std::memory_order_acquire);
        return *(*ptr);
    }

    /**
     * Atomically swap front and back buffers
     * Should be called by writer thread after completing writes
     */
    void swap() noexcept {
        // Swap the pointers atomically
        std::unique_ptr<T>* temp = front_ptr_.exchange(back_ptr_.load(std::memory_order_acquire), 
                                                      std::memory_order_acq_rel);
        back_ptr_.store(temp, std::memory_order_release);
    }

    /**
     * Update the back buffer with a new value and swap
     * Combines write and swap operations
     * @param new_value The new value to set
     */
    void update_and_swap(const T& new_value) {
        auto back_buffer = back_ptr_.load(std::memory_order_acquire);
        *(*back_buffer) = new_value;
        swap();
    }

    /**
     * Update the back buffer with a new value using move semantics and swap
     * @param new_value The new value to set
     */
    void update_and_swap(T&& new_value) {
        auto back_buffer = back_ptr_.load(std::memory_order_acquire);
        *(*back_buffer) = std::forward<T>(new_value);
        swap();
    }

private:
    // Buffers to hold the state
    std::unique_ptr<T> front_buffer_;
    std::unique_ptr<T> back_buffer_;
    
    // Atomic pointers to the buffers to avoid ABA problems
    std::atomic<std::unique_ptr<T>*> front_ptr_;
    std::atomic<std::unique_ptr<T>*> back_ptr_;
};

} // namespace threading
} // namespace btq