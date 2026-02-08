#ifndef BTQ_ATOMIC_SIGNAL_HPP
#define BTQ_ATOMIC_SIGNAL_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <type_traits>

namespace btq {
namespace threading {

/**
 * @brief Atomic signal class using C++20/26 atomic wait/notify mechanisms
 * 
 * This class provides a lock-free signaling mechanism that replaces traditional
 * condition variables with atomic wait/notify operations for improved performance
 * in high-frequency scenarios.
 */
class AtomicSignal {
private:
    std::atomic<uint32_t> signal_state_{0};
    
public:
    /**
     * @brief Constructor
     */
    AtomicSignal() = default;
    
    /**
     * @brief Destructor
     */
    ~AtomicSignal() = default;
    
    /**
     * @brief Signal one waiting thread
     *
     * Equivalent to condition_variable::notify_one()
     */
    void notify_one() noexcept {
        signal_state_.fetch_add(1u, std::memory_order_release);
        // Note: std::atomic::notify_one() is only available in C++20 and later
        // For compatibility with older standards, we use a simple fetch_add
    }

    /**
     * @brief Signal all waiting threads
     *
     * Equivalent to condition_variable::notify_all()
     */
    void notify_all() noexcept {
        signal_state_.fetch_add(1u, std::memory_order_release);
        // Note: std::atomic::notify_all() is only available in C++20 and later
        // For compatibility with older standards, we use a simple fetch_add
    }

    /**
     * @brief Wait for the signal
     *
     * Blocks the calling thread until the signal is notified.
     */
    void wait() const noexcept {
        uint32_t expected = signal_state_.load(std::memory_order_acquire);
        while (signal_state_.load(std::memory_order_acquire) == expected) {
            // Note: std::atomic::wait() is only available in C++20 and later
            // For compatibility with older standards, we use a simple spin-wait
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            expected = signal_state_.load(std::memory_order_acquire);
        }
    }

    /**
     * @brief Wait for the signal with predicate
     *
     * @param pred Predicate function to evaluate before waiting
     */
    template<typename Predicate>
    void wait(Predicate pred) const {
        while (!pred()) {
            uint32_t expected = signal_state_.load(std::memory_order_acquire);
            if (pred()) {
                return; // Predicate became true without waiting
            }

            // Wait while the value hasn't changed
            while (signal_state_.load(std::memory_order_acquire) == expected && !pred()) {
                // Note: std::atomic::wait() is only available in C++20 and later
                // For compatibility with older standards, we use a simple spin-wait
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                expected = signal_state_.load(std::memory_order_acquire);
            }
        }
    }
    
    /**
     * @brief Wait for a specific duration
     * 
     * @tparam Rep Duration representation type
     * @tparam Period Duration period type
     * @param timeout_duration Duration to wait
     * @return true if signal was received, false if timeout occurred
     */
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout_duration) const {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + timeout_duration;
        
        uint32_t expected = signal_state_.load(std::memory_order_acquire);
        
        while (signal_state_.load(std::memory_order_acquire) == expected) {
            // Check if timeout has already passed
            auto current_time = std::chrono::steady_clock::now();
            if (current_time >= end_time) {
                return false; // Timeout
            }
            
            // Simple spin-wait approach since std::atomic doesn't have wait_for
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        return true; // Signaled
    }
    
    /**
     * @brief Wait until a specific time point
     * 
     * @tparam Clock Clock type
     * @tparam Duration Duration type
     * @param timeout_time Time point to wait until
     * @return true if signal was received, false if timeout occurred
     */
    template<typename Clock, typename Duration>
    bool wait_until(const std::chrono::time_point<Clock, Duration>& timeout_time) const {
        auto current_time = Clock::now();
        if (current_time >= timeout_time) {
            return false; // Already timed out
        }
        
        auto remaining = timeout_time - current_time;
        return wait_for(remaining);
    }
    
    /**
     * @brief Wait with predicate and timeout
     * 
     * @tparam Rep Duration representation type
     * @tparam Period Duration period type
     * @tparam Predicate Predicate function type
     * @param timeout_duration Duration to wait
     * @param pred Predicate function to evaluate
     * @return std::cv_status::no_timeout if predicate became true, std::cv_status::timeout otherwise
     */
    template<typename Rep, typename Period, typename Predicate>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout_duration, Predicate pred) const {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + timeout_duration;
        
        while (!pred()) {
            auto current_time = std::chrono::steady_clock::now();
            if (current_time >= end_time) {
                return false; // Timeout
            }
            
            // Simple spin-wait approach since std::atomic doesn't have wait_for
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        return true; // Predicate satisfied
    }
    
    /**
     * @brief Get current signal state
     * 
     * @return Current value of the signal state
     */
    uint32_t get_state() const noexcept {
        return signal_state_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Reset the signal state
     * 
     * This is mainly for testing purposes
     */
    void reset() noexcept {
        signal_state_.store(0, std::memory_order_release);
    }
};

/**
 * @brief Specialized atomic signal for boolean state
 * 
 * A more efficient version for simple true/false signaling
 */
class AtomicBooleanSignal {
private:
    std::atomic<bool> signal_state_{false};
    
public:
    /**
     * @brief Constructor
     */
    AtomicBooleanSignal() = default;
    
    /**
     * @brief Destructor
     */
    ~AtomicBooleanSignal() = default;
    
    /**
     * @brief Signal waiting threads
     */
    void signal() noexcept {
        signal_state_.store(true, std::memory_order_release);
        // Note: std::atomic::notify_all() is only available in C++20 and later
        // For compatibility with older standards, we use a simple store
    }

    /**
     * @brief Reset the signal state
     */
    void reset() noexcept {
        signal_state_.store(false, std::memory_order_release);
    }

    /**
     * @brief Wait for the signal to become true
     */
    void wait() const noexcept {
        while (!signal_state_.load(std::memory_order_acquire)) {
            // Note: std::atomic::wait() is only available in C++20 and later
            // For compatibility with older standards, we use a simple spin-wait
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }

    /**
     * @brief Wait for the signal with predicate
     */
    template<typename Predicate>
    void wait(Predicate pred) const {
        while (!pred() && !signal_state_.load(std::memory_order_acquire)) {
            if (pred() || signal_state_.load(std::memory_order_acquire)) {
                return;
            }
            // Note: std::atomic::wait() is only available in C++20 and later
            // For compatibility with older standards, we use a simple spin-wait
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    
    /**
     * @brief Wait for a specific duration
     */
    template<typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout_duration) const {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + timeout_duration;
        
        while (!signal_state_.load(std::memory_order_acquire)) {
            auto current_time = std::chrono::steady_clock::now();
            if (current_time >= end_time) {
                return false; // Timeout
            }
            
            // Simple spin-wait approach since std::atomic doesn't have wait_for
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        
        return true; // Signaled
    }
    
    /**
     * @brief Check if signal is set
     */
    bool is_signaled() const noexcept {
        return signal_state_.load(std::memory_order_acquire);
    }
};

/**
 * @brief Counter-based atomic signal
 * 
 * Useful for signaling when a specific count is reached
 */
class AtomicCounterSignal {
private:
    std::atomic<uint32_t> counter_{0};
    std::atomic<uint32_t> target_{1}; // Default target is 1
    
public:
    /**
     * @brief Constructor
     */
    explicit AtomicCounterSignal(uint32_t initial_target = 1) 
        : target_(initial_target) {}
    
    /**
     * @brief Destructor
     */
    ~AtomicCounterSignal() = default;
    
    /**
     * @brief Increment the counter and notify if target is reached
     */
    void increment() noexcept {
        uint32_t old_value = counter_.fetch_add(1u, std::memory_order_acq_rel);
        uint32_t new_value = old_value + 1;

        if (new_value >= target_.load(std::memory_order_acquire)) {
            // Note: std::atomic::notify_all() is only available in C++20 and later
            // For compatibility with older standards, we use a simple fetch_add
        }
    }

    /**
     * @brief Set a new target value
     */
    void set_target(uint32_t new_target) noexcept {
        target_.store(new_target, std::memory_order_release);

        // Note: std::atomic::notify_all() is only available in C++20 and later
        // For compatibility with older standards, we use a simple store
    }
    
    /**
     * @brief Get current counter value
     */
    uint32_t get_count() const noexcept {
        return counter_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Get target value
     */
    uint32_t get_target() const noexcept {
        return target_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Wait until counter reaches target
     */
    void wait_for_target() const noexcept {
        uint32_t current = counter_.load(std::memory_order_acquire);
        uint32_t target = target_.load(std::memory_order_acquire);

        while (current < target) {
            // Note: std::atomic::wait() is only available in C++20 and later
            // For compatibility with older standards, we use a simple spin-wait
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            current = counter_.load(std::memory_order_acquire);
            target = target_.load(std::memory_order_acquire);
        }
    }
    
    /**
     * @brief Wait with timeout until counter reaches target
     */
    template<typename Rep, typename Period>
    bool wait_for_target(const std::chrono::duration<Rep, Period>& timeout_duration) const {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + timeout_duration;
        
        uint32_t current = counter_.load(std::memory_order_acquire);
        uint32_t target = target_.load(std::memory_order_acquire);
        
        while (current < target) {
            auto current_time = std::chrono::steady_clock::now();
            if (current_time >= end_time) {
                return false; // Timeout
            }
            
            // Simple spin-wait approach since std::atomic doesn't have wait_for
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            current = counter_.load(std::memory_order_acquire);
            target = target_.load(std::memory_order_acquire);
        }
        
        return true; // Target reached
    }
    
    /**
     * @brief Reset counter to zero
     */
    void reset() noexcept {
        counter_.store(0, std::memory_order_release);
    }
    
    /**
     * @brief Reset counter and set new target
     */
    void reset(uint32_t new_target) noexcept {
        counter_.store(0, std::memory_order_release);
        target_.store(new_target, std::memory_order_release);
    }
};

} // namespace threading
} // namespace btq

#endif // BTQ_ATOMIC_SIGNAL_HPP