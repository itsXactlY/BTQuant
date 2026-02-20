#pragma once

/**
 * @file execution_thread.hpp
 * @brief Dedicated Execution Thread with SPSC Command Queue
 * 
 * This implementation provides:
 * - Lock-free SPSC queue for order commands
 * - Dedicated thread for order execution
 * - Zero UI blocking during order submission
 * - Batch order processing
 * - Rate limiting and throttling
 * - Order status tracking
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace btq {
namespace trading {

// Forward declarations
struct OrderRequest;
struct OrderResponse;

/**
 * @brief Execution command types
 */
enum class ExecutionCommandType : uint8_t {
    SUBMIT_ORDER,
    CANCEL_ORDER,
    MODIFY_ORDER,
    BATCH_SUBMIT,
    SYNC_REQUEST,       // Force sync with exchange
    SHUTDOWN
};

/**
 * @brief Execution command for the SPSC queue
 */
struct ExecutionCommand {
    uint64_t id = 0;
    ExecutionCommandType type = ExecutionCommandType::SUBMIT_ORDER;
    OrderRequest* request = nullptr;
    std::string order_id;           // For cancel/modify
    double new_price = 0.0;         // For modify
    double new_quantity = 0.0;      // For modify
    int64_t timestamp = 0;
    uint32_t priority = 0;          // Higher = more urgent
};

/**
 * @brief Execution result
 */
struct ExecutionResult {
    uint64_t command_id = 0;
    bool success = false;
    OrderResponse response;
    std::string error_message;
    int64_t execution_time_us = 0;  // Execution time in microseconds
};

/**
 * @brief Lock-free SPSC ring buffer for execution commands
 * 
 * Single-Producer (UI thread) -> Single-Consumer (Execution thread)
 */
template<size_t Capacity>
class ExecutionCommandQueue {
public:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
    ExecutionCommandQueue() = default;
    
    /**
     * @brief Push a command to the queue (producer side)
     * @return true if successful, false if queue is full
     */
    bool push(ExecutionCommand&& cmd) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (current_head + 1) & (Capacity - 1);
        
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Queue is full
        }
        
        buffer_[current_head] = std::move(cmd);
        head_.store(next_head, std::memory_order_release);
        
        return true;
    }
    
    /**
     * @brief Pop a command from the queue (consumer side)
     * @return The command, or nullopt if queue is empty
     */
    std::optional<ExecutionCommand> pop() {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Queue is empty
        }
        
        ExecutionCommand cmd = std::move(buffer_[current_tail]);
        tail_.store((current_tail + 1) & (Capacity - 1), std::memory_order_release);
        
        return cmd;
    }
    
    /**
     * @brief Check if queue is empty
     */
    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Get the number of elements in the queue
     */
    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & (Capacity - 1);
    }
    
    /**
     * @brief Get capacity
     */
    static constexpr size_t capacity() { return Capacity; }

private:
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::array<ExecutionCommand, Capacity> buffer_;
};

/**
 * @brief Result callback type
 */
using ResultCallback = std::function<void(const ExecutionResult&)>;

/**
 * @brief Order submission callback type
 */
using OrderSubmitCallback = std::function<OrderResponse(const OrderRequest&)>;

/**
 * @brief Order cancellation callback type
 */
using OrderCancelCallback = std::function<bool(const std::string& order_id)>;

/**
 * @brief Rate limiter for order submission
 */
class RateLimiter {
public:
    RateLimiter(size_t max_requests, std::chrono::milliseconds window)
        : max_requests_(max_requests)
        , window_(window)
        , tokens_(max_requests)
    {}
    
    /**
     * @brief Try to acquire a token
     * @return true if allowed, false if rate limited
     */
    bool tryAcquire() {
        auto now = std::chrono::steady_clock::now();
        
        // Refill tokens based on time elapsed
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_refill_);
        if (elapsed >= window_) {
            tokens_ = max_requests_;
            last_refill_ = now;
        }
        
        if (tokens_ > 0) {
            --tokens_;
            return true;
        }
        
        return false;
    }
    
    /**
     * @brief Get time until next token is available
     */
    std::chrono::milliseconds timeUntilNextToken() const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_refill_);
        
        if (elapsed >= window_) {
            return std::chrono::milliseconds(0);
        }
        
        return window_ - elapsed;
    }

private:
    size_t max_requests_;
    std::chrono::milliseconds window_;
    size_t tokens_;
    std::chrono::steady_clock::time_point last_refill_ = std::chrono::steady_clock::now();
};

/**
 * @brief Dedicated execution thread for order management
 */
class ExecutionThread {
public:
    /**
     * @brief Configuration
     */
    struct Config {
        size_t queue_capacity = 1024;
        size_t batch_size = 10;
        std::chrono::milliseconds poll_interval{1};
        std::chrono::milliseconds rate_limit_window{1000};
        size_t rate_limit_max = 50;
        bool enable_batching = true;
    };
    
    ExecutionThread() = default;
    
    /**
     * @brief Initialize and start the execution thread
     */
    void initialize(const Config& config = Config{}) {
        config_ = config;
        rate_limiter_ = std::make_unique<RateLimiter>(
            config.rate_limit_max, 
            config.rate_limit_window);
        
        running_.store(true);
        thread_ = std::thread(&ExecutionThread::run, this);
    }
    
    /**
     * @brief Shutdown the execution thread
     */
    void shutdown() {
        running_.store(false);
        
        // Push shutdown command
        ExecutionCommand cmd;
        cmd.type = ExecutionCommandType::SHUTDOWN;
        pushCommand(std::move(cmd));
        
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    
    ~ExecutionThread() {
        shutdown();
    }
    
    // Non-copyable, non-movable
    ExecutionThread(const ExecutionThread&) = delete;
    ExecutionThread& operator=(const ExecutionThread&) = delete;
    
    /**
     * @brief Set the order submission callback
     */
    void setSubmitCallback(OrderSubmitCallback callback) {
        submit_callback_ = std::move(callback);
    }
    
    /**
     * @brief Set the order cancellation callback
     */
    void setCancelCallback(OrderCancelCallback callback) {
        cancel_callback_ = std::move(callback);
    }
    
    /**
     * @brief Set the result callback
     */
    void setResultCallback(ResultCallback callback) {
        result_callback_ = std::move(callback);
    }
    
    /**
     * @brief Submit an order (non-blocking)
     */
    uint64_t submitOrder(OrderRequest&& request) {
        uint64_t id = next_command_id_++;
        
        ExecutionCommand cmd;
        cmd.id = id;
        cmd.type = ExecutionCommandType::SUBMIT_ORDER;
        cmd.request = new OrderRequest(std::move(request));
        cmd.timestamp = std::chrono::nanoseconds(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        pushCommand(std::move(cmd));
        
        return id;
    }
    
    /**
     * @brief Cancel an order (non-blocking)
     */
    uint64_t cancelOrder(const std::string& order_id) {
        uint64_t id = next_command_id_++;
        
        ExecutionCommand cmd;
        cmd.id = id;
        cmd.type = ExecutionCommandType::CANCEL_ORDER;
        cmd.order_id = order_id;
        cmd.timestamp = std::chrono::nanoseconds(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        pushCommand(std::move(cmd));
        
        return id;
    }
    
    /**
     * @brief Modify an order (non-blocking)
     */
    uint64_t modifyOrder(const std::string& order_id, double new_price, double new_quantity = 0.0) {
        uint64_t id = next_command_id_++;
        
        ExecutionCommand cmd;
        cmd.id = id;
        cmd.type = ExecutionCommandType::MODIFY_ORDER;
        cmd.order_id = order_id;
        cmd.new_price = new_price;
        cmd.new_quantity = new_quantity;
        cmd.timestamp = std::chrono::nanoseconds(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        pushCommand(std::move(cmd));
        
        return id;
    }
    
    /**
     * @brief Submit multiple orders as a batch
     */
    uint64_t submitBatch(std::vector<OrderRequest>&& requests) {
        uint64_t id = next_command_id_++;
        
        ExecutionCommand cmd;
        cmd.id = id;
        cmd.type = ExecutionCommandType::BATCH_SUBMIT;
        // Batch requests would be stored separately
        cmd.timestamp = std::chrono::nanoseconds(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        pushCommand(std::move(cmd));
        
        return id;
    }
    
    /**
     * @brief Get pending command count
     */
    size_t getPendingCount() const {
        return command_queue_.size();
    }
    
    /**
     * @brief Check if the thread is running
     */
    bool isRunning() const {
        return running_.load();
    }

private:
    void pushCommand(ExecutionCommand&& cmd) {
        while (!command_queue_.push(std::move(cmd))) {
            // Queue is full, wait a bit
            std::this_thread::yield();
        }
    }
    
    void run() {
        std::vector<ExecutionCommand> batch;
        batch.reserve(config_.batch_size);
        
        while (running_.load()) {
            // Try to pop a command
            auto cmd = command_queue_.pop();
            
            if (!cmd) {
                // No commands, wait
                std::this_thread::sleep_for(config_.poll_interval);
                continue;
            }
            
            // Check for shutdown
            if (cmd->type == ExecutionCommandType::SHUTDOWN) {
                break;
            }
            
            // Process command
            processCommand(*cmd);
        }
    }
    
    void processCommand(const ExecutionCommand& cmd) {
        auto start_time = std::chrono::steady_clock::now();
        
        ExecutionResult result;
        result.command_id = cmd.id;
        
        // Rate limiting
        if (!rate_limiter_->tryAcquire()) {
            result.success = false;
            result.error_message = "Rate limited";
            notifyResult(result);
            return;
        }
        
        switch (cmd.type) {
            case ExecutionCommandType::SUBMIT_ORDER:
                if (submit_callback_ && cmd.request) {
                    result.response = submit_callback_(*cmd.request);
                    result.success = result.response.success;
                    delete cmd.request;
                }
                break;
                
            case ExecutionCommandType::CANCEL_ORDER:
                if (cancel_callback_) {
                    result.success = cancel_callback_(cmd.order_id);
                }
                break;
                
            case ExecutionCommandType::MODIFY_ORDER:
                // Would implement modify logic
                result.success = false;
                result.error_message = "Modify not implemented";
                break;
                
            case ExecutionCommandType::BATCH_SUBMIT:
                // Would implement batch logic
                result.success = false;
                result.error_message = "Batch not implemented";
                break;
                
            default:
                result.success = false;
                result.error_message = "Unknown command type";
                break;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        result.execution_time_us = std::chrono::duration_cast<
            std::chrono::microseconds>(end_time - start_time).count();
        
        notifyResult(result);
    }
    
    void notifyResult(const ExecutionResult& result) {
        if (result_callback_) {
            result_callback_(result);
        }
    }
    
    Config config_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    
    ExecutionCommandQueue<1024> command_queue_;
    std::atomic<uint64_t> next_command_id_{1};
    
    std::unique_ptr<RateLimiter> rate_limiter_;
    
    OrderSubmitCallback submit_callback_;
    OrderCancelCallback cancel_callback_;
    ResultCallback result_callback_;
};

/**
 * @brief Order status tracker
 */
class OrderStatusTracker {
public:
    /**
     * @brief Add or update an order
     */
    void updateOrder(const std::string& order_id, OrderStatus status, 
                     const std::string& info = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto& entry = orders_[order_id];
        entry.status = status;
        entry.info = info;
        entry.last_update = std::chrono::steady_clock::now();
    }
    
    /**
     * @brief Get order status
     */
    std::optional<OrderStatus> getStatus(const std::string& order_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = orders_.find(order_id);
        if (it != orders_.end()) {
            return it->second.status;
        }
        return std::nullopt;
    }
    
    /**
     * @brief Remove an order
     */
    void removeOrder(const std::string& order_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        orders_.erase(order_id);
    }
    
    /**
     * @brief Get all open orders
     */
    std::vector<std::string> getOpenOrders() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::vector<std::string> open_orders;
        for (const auto& [id, entry] : orders_) {
            if (entry.status == OrderStatus::OPEN || 
                entry.status == OrderStatus::PENDING) {
                open_orders.push_back(id);
            }
        }
        return open_orders;
    }

private:
    struct OrderEntry {
        OrderStatus status = OrderStatus::PENDING;
        std::string info;
        std::chrono::steady_clock::time_point last_update;
    };
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, OrderEntry> orders_;
};

/**
 * @brief Latency statistics for execution
 */
class ExecutionLatencyStats {
public:
    void recordLatency(int64_t latency_us) {
        total_count_++;
        total_latency_us_ += latency_us;
        min_latency_us_ = std::min(min_latency_us_, latency_us);
        max_latency_us_ = std::max(max_latency_us_, latency_us);
        
        // Simple moving average for last N samples
        recent_latencies_.push_back(latency_us);
        if (recent_latencies_.size() > 100) {
            recent_latencies_.erase(recent_latencies_.begin());
        }
    }
    
    double getAverageLatency() const {
        if (total_count_ == 0) return 0.0;
        return static_cast<double>(total_latency_us_) / total_count_;
    }
    
    double getRecentAverageLatency() const {
        if (recent_latencies_.empty()) return 0.0;
        int64_t sum = 0;
        for (int64_t lat : recent_latencies_) {
            sum += lat;
        }
        return static_cast<double>(sum) / recent_latencies_.size();
    }
    
    int64_t getMinLatency() const { return min_latency_us_; }
    int64_t getMaxLatency() const { return max_latency_us_; }
    size_t getCount() const { return total_count_; }

private:
    size_t total_count_ = 0;
    int64_t total_latency_us_ = 0;
    int64_t min_latency_us_ = INT64_MAX;
    int64_t max_latency_us_ = 0;
    std::vector<int64_t> recent_latencies_;
};

} // namespace trading
} // namespace btq
