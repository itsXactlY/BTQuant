#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdint>

// Forward declaration of data types to avoid circular includes
namespace BTQuant {
    struct OrderBookSnapshot;
}

/**
 * @brief Manages a rolling buffer of orderbook snapshots for historical depth visualization
 * Implements a lock-free circular buffer of the top 100 Bid/Ask levels, snapping state every 100ms
 */
class OrderbookSnapshotManager {
public:
    // Maximum number of price levels to store per snapshot (top 100 levels)
    static constexpr size_t MAX_LEVELS = 100;
    
    // Default buffer size for historical snapshots
    static constexpr size_t DEFAULT_BUFFER_SIZE = 10000; // 1000 snapshots = ~1000 seconds at 100ms intervals
    
    // Snapshot interval in milliseconds
    static constexpr auto SNAPSHOT_INTERVAL = std::chrono::milliseconds(100);

    /**
     * @brief Represents a single orderbook snapshot with bid/ask levels
     */
    struct OrderbookSnapshot {
        uint64_t timestamp;           // Nanosecond precision timestamp
        uint32_t symbol_id;          // Symbol identifier
        uint32_t bids_count;         // Number of valid bid levels
        uint32_t asks_count;         // Number of valid ask levels
        
        // Bid and ask levels (top 100 each)
        struct Level {
            double price;
            double size;
            
            Level() : price(0.0), size(0.0) {}
            Level(double p, double s) : price(p), size(s) {}
        };
        
        Level bids[MAX_LEVELS];
        Level asks[MAX_LEVELS];
        
        OrderbookSnapshot() : timestamp(0), symbol_id(0), bids_count(0), asks_count(0) {
            std::memset(bids, 0, sizeof(bids));
            std::memset(asks, 0, sizeof(asks));
        }
    };

private:
    // Circular buffer for storing snapshots
    std::unique_ptr<OrderbookSnapshot[]> buffer_;
    
    // Buffer metadata
    const size_t buffer_size_;
    std::atomic<size_t> write_index_{0};
    std::atomic<size_t> read_index_{0};
    std::atomic<size_t> count_{0};
    
    // Control variables for the snapshot thread
    std::atomic<bool> running_{false};
    std::thread snapshot_thread_;
    
    // Pointer to the data bridge for reading current orderbook state
    // Using void* to avoid circular dependency; cast to HotSpineDataBridge* when needed
    void* data_bridge_ptr_;
    
    // Last snapshot time to control interval
    std::atomic<int64_t> last_snapshot_time_{0};

public:
    /**
     * @brief Constructor
     * @param data_bridge_ptr Pointer to the hotspine data bridge
     * @param buffer_size Size of the circular buffer (default 10000)
     */
    explicit OrderbookSnapshotManager(void* data_bridge_ptr, 
                                     size_t buffer_size = DEFAULT_BUFFER_SIZE)
        : buffer_size_(buffer_size), data_bridge_ptr_(data_bridge_ptr) {
        buffer_ = std::make_unique<OrderbookSnapshot[]>(buffer_size_);
    }

    /**
     * @brief Destructor - stops the snapshot thread if running
     */
    ~OrderbookSnapshotManager() {
        stop();
        if (snapshot_thread_.joinable()) {
            snapshot_thread_.join();
        }
    }

    /**
     * @brief Start the snapshot collection thread
     */
    void start() {
        if (!running_.exchange(true)) {
            snapshot_thread_ = std::thread(&OrderbookSnapshotManager::snapshot_loop, this);
        }
    }

    /**
     * @brief Stop the snapshot collection thread
     */
    void stop() {
        running_.store(false);
    }

    /**
     * @brief Take a manual snapshot of the current orderbook state
     * @param symbol_id The symbol to snapshot
     */
    void take_snapshot(uint32_t symbol_id) {
        if (!running_.load()) return;

        // Get current time
        auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        // Check if enough time has passed since last snapshot
        int64_t last_time = last_snapshot_time_.load();
        if (now - last_time < SNAPSHOT_INTERVAL.count() * 1000000) { // Convert ms to ns
            return;
        }

        // Attempt to update last snapshot time atomically
        if (!last_snapshot_time_.compare_exchange_strong(last_time, now)) {
            return; // Another thread took a snapshot
        }

        // In a real implementation, we would cast data_bridge_ptr_ to HotSpineDataBridge*
        // and get the current orderbook state from the data bridge.
        // For now, we'll create a dummy snapshot - this would be replaced with actual data
        OrderbookSnapshot snapshot;
        snapshot.timestamp = now;
        snapshot.symbol_id = symbol_id;
        
        // In a real implementation, we would populate the snapshot with actual orderbook data
        // from the data bridge. For now, we'll leave it empty as a placeholder.
        
        // Add snapshot to the buffer
        add_snapshot(std::move(snapshot));
    }

    /**
     * @brief Add a snapshot to the circular buffer
     * @param snapshot The snapshot to add
     */
    void add_snapshot(OrderbookSnapshot&& snapshot) {
        size_t current_write;
        size_t next_write;
        
        // Loop until we successfully update the write index
        do {
            current_write = write_index_.load(std::memory_order_acquire);
            next_write = (current_write + 1) % buffer_size_;
            
            // Check if buffer is full (would overwrite unread data)
            if (next_write == read_index_.load(std::memory_order_acquire)) {
                // Advance read index to prevent overwrite (oldest data is lost)
                size_t current_read = read_index_.load(std::memory_order_relaxed);
                size_t next_read = (current_read + 1) % buffer_size_;
                
                if (read_index_.compare_exchange_weak(current_read, next_read, 
                                                     std::memory_order_release, 
                                                     std::memory_order_relaxed)) {
                    count_.fetch_sub(1, std::memory_order_relaxed);
                }
            }
            
            // Try to advance the write index
        } while (!write_index_.compare_exchange_weak(current_write, next_write,
                                                    std::memory_order_release,
                                                    std::memory_order_acquire));
        
        // Store the snapshot at the position we secured
        buffer_[current_write] = std::move(snapshot);
        
        // Increment the count
        count_.fetch_add(1, std::memory_order_release);
    }

    /**
     * @brief Get the most recent snapshot
     * @return Pointer to the most recent snapshot, or nullptr if none available
     */
    const OrderbookSnapshot* get_latest_snapshot() const {
        size_t current_write = write_index_.load(std::memory_order_acquire);
        if (count_.load(std::memory_order_relaxed) == 0) {
            return nullptr;
        }
        
        size_t latest_index = (current_write == 0) ? buffer_size_ - 1 : current_write - 1;
        return &buffer_[latest_index];
    }

    /**
     * @brief Get a range of snapshots
     * @param start_index Starting index (relative to read position)
     * @param count Number of snapshots to retrieve
     * @return Vector of snapshot pointers
     */
    std::vector<const OrderbookSnapshot*> get_snapshots(size_t start_index, size_t count) const {
        std::vector<const OrderbookSnapshot*> result;
        size_t current_count = count_.load(std::memory_order_relaxed);
        
        if (current_count == 0) {
            return result;
        }
        
        size_t actual_start = std::min(start_index, current_count - 1);
        size_t actual_count = std::min(count, current_count - actual_start);
        
        result.reserve(actual_count);
        
        for (size_t i = 0; i < actual_count; ++i) {
            size_t idx = (read_index_.load(std::memory_order_acquire) + actual_start + i) % buffer_size_;
            result.push_back(&buffer_[idx]);
        }
        
        return result;
    }

    /**
     * @brief Get the number of snapshots currently in the buffer
     */
    size_t size() const {
        return count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get the maximum capacity of the buffer
     */
    size_t capacity() const {
        return buffer_size_;
    }

    /**
     * @brief Clear all snapshots from the buffer
     */
    void clear() {
        write_index_.store(0, std::memory_order_release);
        read_index_.store(0, std::memory_order_release);
        count_.store(0, std::memory_order_relaxed);
    }

private:
    /**
     * @brief Main loop for automatic snapshot taking
     */
    void snapshot_loop() {
        while (running_.load()) {
            // In a real implementation, we would iterate through symbols and take snapshots
            // For now, we'll just sleep to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
};