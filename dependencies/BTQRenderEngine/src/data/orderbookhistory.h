#ifndef ORDERBOOK_HISTORY_H
#define ORDERBOOK_HISTORY_H

#include <vector>
#include <memory>
#include <mutex>
#include <chrono>

// Define structures for order book data
struct OrderLevel {
    double price;
    double quantity;
    int64_t timestamp; // Unix timestamp in microseconds
    
    OrderLevel() : price(0.0), quantity(0.0), timestamp(0) {}
    OrderLevel(double p, double q, int64_t ts) : price(p), quantity(q), timestamp(ts) {}
};

struct OrderBookSnapshot {
    std::vector<OrderLevel> bids; // Top N bid levels
    std::vector<OrderLevel> asks; // Top N ask levels
    int64_t snapshotTime;         // Time when snapshot was taken
    
    OrderBookSnapshot() : snapshotTime(0) {
        bids.reserve(100); // Reserve space for top 100 levels
        asks.reserve(100);
    }
    
    explicit OrderBookSnapshot(int maxLevels) : snapshotTime(0) {
        bids.reserve(maxLevels);
        asks.reserve(maxLevels);
    }
};

class OrderBookHistory {
private:
    std::vector<std::unique_ptr<OrderBookSnapshot>> buffer_;
    size_t capacity_;
    size_t head_;           // Index of the oldest element
    size_t size_;           // Current number of elements in buffer
    mutable std::mutex mutex_;
    
public:
    explicit OrderBookHistory(size_t capacity = 1000);
    
    // Add a new snapshot to the circular buffer
    void addSnapshot(std::unique_ptr<OrderBookSnapshot> snapshot);
    
    // Retrieve a snapshot by index (0 = oldest, size_-1 = newest)
    std::unique_ptr<OrderBookSnapshot> getSnapshotByIndex(size_t index) const;
    
    // Retrieve a snapshot by timestamp (closest match)
    std::unique_ptr<OrderBookSnapshot> getSnapshotByTime(int64_t targetTime) const;
    
    // Get the most recent snapshot
    std::unique_ptr<OrderBookSnapshot> getLatestSnapshot() const;
    
    // Get snapshot count
    size_t size() const;
    
    // Get buffer capacity
    size_t capacity() const;
    
    // Check if buffer is empty
    bool empty() const;
    
    // Clear the buffer
    void clear();
    
    // Get all timestamps for UI timeline controls
    std::vector<int64_t> getAllTimestamps() const;

private:
    // Helper to get timestamp at internal index (for searching)
    int64_t getSnapshotTimeAtInternalIndex(size_t internalIdx) const;
};

// Convenience functions for creating snapshots
std::unique_ptr<OrderBookSnapshot> createSnapshot(
    const std::vector<OrderLevel>& bids,
    const std::vector<OrderLevel>& asks,
    int64_t timestamp,
    int maxLevels = 100);

#endif // ORDERBOOK_HISTORY_H