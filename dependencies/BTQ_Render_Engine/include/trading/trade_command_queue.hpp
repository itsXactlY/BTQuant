#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>

// Lock-free SPSC queue from moodycamel (concurrentqueue)
// We'll use a simple ring buffer implementation for SPSC
#include "readerwriterqueue.h"

namespace BTQuant {
namespace RenderEngine {

/**
 * Order side enumeration
 */
enum class OrderSide {
    BUY,
    SELL
};

/**
 * Order type enumeration
 */
enum class OrderType {
    MARKET,
    LIMIT,
    STOP_MARKET,
    STOP_LIMIT,
    TRAILING_STOP
};

/**
 * Time in Force enumeration
 */
enum class TimeInForce {
    GTC,  // Good Till Cancel
    IOC,  // Immediate Or Cancel
    FOK,  // Fill Or Kill
    DAY   // Day Order
};

/**
 * Trade command structure for SPSC queue
 * This is pushed by the UI thread and consumed by the execution thread
 */
struct TradeCommand {
    uint64_t command_id = 0;          // Unique command ID
    uint32_t symbol_id = 0;           // Symbol ID
    std::string symbol;               // Symbol string (e.g., "BTC-USDT")
    std::string exchange;             // Exchange name (e.g., "Binance")
    OrderSide side = OrderSide::BUY;  // Buy or Sell
    OrderType type = OrderType::MARKET;  // Order type
    TimeInForce tif = TimeInForce::GTC;  // Time in Force
    
    double quantity = 0.0;            // Order quantity
    double price = 0.0;               // Limit price (for limit orders)
    double stop_price = 0.0;          // Stop price (for stop orders)
    
    uint64_t timestamp = 0;           // Command creation timestamp
    uint64_t client_order_id = 0;     // Client-generated order ID
    
    // Optional fields
    std::string strategy_id;          // Strategy that generated this order
    std::string notes;                // Additional notes
    
    TradeCommand() = default;
    
    TradeCommand(uint32_t sym_id, const std::string& sym, const std::string& exch,
                 OrderSide s, OrderType t, double qty, TimeInForce tif_val = TimeInForce::GTC)
        : symbol_id(sym_id), symbol(sym), exchange(exch), side(s), type(t), 
          tif(tif_val), quantity(qty) {
        timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }
};

/**
 * Trade result structure for async callbacks
 */
struct TradeResult {
    uint64_t command_id = 0;
    uint64_t order_id = 0;            // Exchange order ID
    bool success = false;
    std::string error_message;
    double filled_quantity = 0.0;
    double average_price = 0.0;
    uint64_t timestamp = 0;
};

/**
 * Callback type for trade result notifications
 */
using TradeResultCallback = std::function<void(const TradeResult&)>;

/**
 * Lock-free SPSC queue for trading commands
 * 
 * Design principles:
 * - UI thread (producer) pushes commands without blocking
 * - Execution thread (consumer) processes commands asynchronously
 * - No locks on the hot path - uses moodycamel::ReaderWriterQueue
 * - Results are delivered via callbacks on the execution thread
 */
class TradeCommandQueue {
public:
    static constexpr size_t DEFAULT_CAPACITY = 1024;
    
    TradeCommandQueue(size_t capacity = DEFAULT_CAPACITY);
    ~TradeCommandQueue();
    
    // Non-copyable, non-movable
    TradeCommandQueue(const TradeCommandQueue&) = delete;
    TradeCommandQueue& operator=(const TradeCommandQueue&) = delete;
    TradeCommandQueue(TradeCommandQueue&&) = delete;
    TradeCommandQueue& operator=(TradeCommandQueue&&) = delete;
    
    /**
     * Push a trade command to the queue (producer thread only)
     * @param command The trade command to push
     * @return true if successfully pushed, false if queue is full
     */
    bool push(const TradeCommand& command);
    
    /**
     * Push a trade command to the queue (move version, producer thread only)
     * @param command The trade command to push (moved)
     * @return true if successfully pushed, false if queue is full
     */
    bool push(TradeCommand&& command);
    
    /**
     * Try to pop a command from the queue (consumer thread only)
     * @param command Output parameter for the popped command
     * @return true if a command was popped, false if queue is empty
     */
    bool try_pop(TradeCommand& command);
    
    /**
     * Get the approximate size of the queue
     * Note: This is approximate in a concurrent context
     */
    size_t size_approx() const;
    
    /**
     * Check if the queue is empty
     * Note: This is approximate in a concurrent context
     */
    bool empty() const;
    
    /**
     * Set callback for trade results
     * The callback will be invoked on the consumer thread
     */
    void set_result_callback(TradeResultCallback callback);
    
    /**
     * Report a trade result (called by consumer thread)
     */
    void report_result(const TradeResult& result);
    
    /**
     * Get the next command ID (thread-safe)
     */
    uint64_t next_command_id();
    
    /**
     * Start the execution thread
     * @param executor Function that processes trade commands
     */
    void start_execution_thread(std::function<void(TradeCommand&, TradeCommandQueue&)> executor);
    
    /**
     * Stop the execution thread
     */
    void stop_execution_thread();
    
    /**
     * Check if the execution thread is running
     */
    bool is_running() const { return running_.load(std::memory_order_acquire); }

private:
    moodycamel::ReaderWriterQueue<TradeCommand> queue_;
    std::atomic<uint64_t> next_command_id_{1};
    std::atomic<bool> running_{false};
    std::thread execution_thread_;
    TradeResultCallback result_callback_;
};

/**
 * Global trade command queue instance
 * This is the main queue used by the UI for order entry
 */
class GlobalTradeQueue {
public:
    static TradeCommandQueue& instance() {
        static TradeCommandQueue queue;
        return queue;
    }
    
    // Convenience method to push a command
    static bool push_command(const TradeCommand& cmd) {
        return instance().push(cmd);
    }
    
    // Convenience method to push a command (move)
    static bool push_command(TradeCommand&& cmd) {
        return instance().push(std::move(cmd));
    }
    
private:
    GlobalTradeQueue() = delete;
};

}  // namespace RenderEngine
}  // namespace BTQuant
