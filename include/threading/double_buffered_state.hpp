#pragma once

#include <atomic>
#include <memory>
#include <utility>
#include <functional>

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
        , update_counter_(0)
    {
        // Initialize the atomic pointers to point to our buffers
        front_ptr_.store(front_buffer_.get(), std::memory_order_relaxed);
        back_ptr_.store(back_buffer_.get(), std::memory_order_relaxed);
    }

    /**
     * Constructor with move semantics
     * @param initial_state Initial state for both front and back buffers
     */
    explicit DoubleBufferedState(T&& initial_state)
        : front_buffer_(std::make_unique<T>(std::forward<T>(initial_state)))
        , back_buffer_(std::make_unique<T>(*front_buffer_))
        , update_counter_(0)
    {
        // Initialize the atomic pointers to point to our buffers
        front_ptr_.store(front_buffer_.get(), std::memory_order_relaxed);
        back_ptr_.store(back_buffer_.get(), std::memory_order_relaxed);
    }

    // Copy operations - note that these create a new double-buffered state
    // with the same logical value as the original's current read value
    DoubleBufferedState(const DoubleBufferedState& other)
        : front_buffer_(std::make_unique<T>(*(other.front_ptr_.load(std::memory_order_acquire))))  // Copy the current read value to our front
        , back_buffer_(std::make_unique<T>(*(other.front_ptr_.load(std::memory_order_acquire))))  // Also copy to our back initially
        , update_counter_(other.update_counter_.load(std::memory_order_acquire))  // Copy the update count too
    {
        // Initialize the atomic pointers to point to our buffers
        front_ptr_.store(front_buffer_.get(), std::memory_order_relaxed);
        back_ptr_.store(back_buffer_.get(), std::memory_order_relaxed);
    }

    DoubleBufferedState& operator=(const DoubleBufferedState& other) {
        if (this != &other) {
            // Copy the current state from other's current front buffer to both of our buffers
            T temp_value = *(other.front_ptr_.load(std::memory_order_acquire));
            *front_buffer_ = temp_value;
            *back_buffer_ = temp_value;
            update_counter_.store(other.update_counter_.load(std::memory_order_acquire), std::memory_order_release);
        }
        return *this;
    }

    DoubleBufferedState(DoubleBufferedState&& other) noexcept
        : front_buffer_(std::move(other.front_buffer_))
        , back_buffer_(std::move(other.back_buffer_))
        , front_ptr_(other.front_ptr_.load(std::memory_order_acquire))  // Copy the pointer value
        , back_ptr_(other.back_ptr_.load(std::memory_order_acquire))    // Copy the pointer value
        , update_counter_(other.update_counter_.load())
    {
        // Update the moved-from object to have valid pointers to default-constructed values
        other.front_buffer_ = std::make_unique<T>();
        other.back_buffer_ = std::make_unique<T>();
        other.front_ptr_.store(other.front_buffer_.get(), std::memory_order_relaxed);
        other.back_ptr_.store(other.back_buffer_.get(), std::memory_order_relaxed);
        other.update_counter_.store(0);
    }

    DoubleBufferedState& operator=(DoubleBufferedState&& other) noexcept {
        if (this != &other) {
            front_buffer_ = std::move(other.front_buffer_);
            back_buffer_ = std::move(other.back_buffer_);
            front_ptr_.store(front_buffer_.get(), std::memory_order_relaxed);
            back_ptr_.store(back_buffer_.get(), std::memory_order_relaxed);
            update_counter_.store(other.update_counter_.load());

            // Update the moved-from object to have valid pointers to default-constructed values
            other.front_buffer_ = std::make_unique<T>();
            other.back_buffer_ = std::make_unique<T>();
            other.front_ptr_.store(other.front_buffer_.get(), std::memory_order_relaxed);
            other.back_ptr_.store(other.back_buffer_.get(), std::memory_order_relaxed);
            other.update_counter_.store(0);
        }
        return *this;
    }

    /**
     * Get current front buffer for reading (non-modifying)
     * Safe to call from reader thread
     * @return Const reference to the front buffer
     */
    const T& read() const noexcept {
        return *(front_ptr_.load(std::memory_order_acquire));
    }

    /**
     * Get back buffer for writing
     * Safe to call from writer thread
     * @return Reference to the back buffer
     */
    T& write() noexcept {
        return *(back_ptr_.load(std::memory_order_acquire));
    }

    /**
     * Atomically swap front and back buffers
     * Should be called by writer thread after completing writes
     */
    void swap() noexcept {
        // Swap the pointers atomically
        T* temp = front_ptr_.exchange(back_ptr_.load(std::memory_order_acquire),
                                     std::memory_order_acq_rel);
        back_ptr_.store(temp, std::memory_order_release);
        
        // Increment update counter to track number of swaps
        update_counter_.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * Update the back buffer with a new value and swap
     * Combines write and swap operations
     * @param new_value The new value to set
     */
    void update_and_swap(const T& new_value) {
        *(back_ptr_.load(std::memory_order_acquire)) = new_value;
        swap();
    }

    /**
     * Update the back buffer with a new value using move semantics and swap
     * @param new_value The new value to set
     */
    void update_and_swap(T&& new_value) {
        *(back_ptr_.load(std::memory_order_acquire)) = std::forward<T>(new_value);
        swap();
    }

    /**
     * Apply a function to modify the back buffer and then swap
     * Useful for atomic updates that depend on current state
     * @param func Function that takes a reference to T and modifies it
     */
    template<typename Func>
    void modify_and_swap(Func&& func) {
        func(*(back_ptr_.load(std::memory_order_acquire)));
        swap();
    }

    /**
     * Get the number of times the buffers have been swapped
     * Useful for tracking update frequency
     * @return Number of swaps performed
     */
    size_t get_update_count() const noexcept {
        return update_counter_.load(std::memory_order_acquire);
    }

    /**
     * Reset the update counter to zero
     */
    void reset_update_count() noexcept {
        update_counter_.store(0, std::memory_order_release);
    }

    /**
     * Perform a read operation with a callback function
     * This ensures the read is consistent and atomic
     * @param func Function that takes a const reference to T
     */
    template<typename Func>
    auto read_with(Func&& func) const -> decltype(func(std::declval<const T&>())) {
        return func(*(front_ptr_.load(std::memory_order_acquire)));
    }

private:
    // Buffers to hold the state
    std::unique_ptr<T> front_buffer_;
    std::unique_ptr<T> back_buffer_;

    // Atomic pointers to the buffer data to enable lock-free access
    std::atomic<T*> front_ptr_;
    std::atomic<T*> back_ptr_;

    // Counter to track number of updates/swaps
    mutable std::atomic<size_t> update_counter_;
};

} // namespace threading
} // namespace btq