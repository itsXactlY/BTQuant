#ifndef ORDERBOOK_HISTORY_H
#define ORDERBOOK_HISTORY_H

#include <vector>
#include <memory>
#include <mutex>
#include <chrono>

// Structure representing a single level in the order book
struct OrderLevel {
    double price;
    double quantity;
    
    OrderLevel() : price(0.0), quantity(0.0) {}
    OrderLevel(double p, double q) : price(p), quantity(q) {}
};

// Structure representing an L2 snapshot (top 100 levels)
struct L2Snapshot {
    std::vector<OrderLevel> bids;  // Top 100 bid levels
    std::vector<OrderLevel> asks;  // Top 100 ask levels
    int64_t timestamp;             // Unix timestamp in microseconds
    
    L2Snapshot() : timestamp(0) {
        bids.reserve(100);  // Reserve space for top 100 levels
        asks.reserve(100);
    }
    
    explicit L2Snapshot(int maxLevels) : timestamp(0) {
        bids.reserve(maxLevels);
        asks.reserve(maxLevels);
    }
};

// Circular buffer to maintain historical L2 snapshots for heatmap playback
class OrderBookHistory {
private:
    std::vector<std::unique_ptr<L2Snapshot>> buffer_;
    size_t capacity_;
    size_t head_;           // Index of the oldest element
    size_t size_;           // Current number of elements in buffer
    mutable std::mutex mutex_;

public:
    explicit OrderBookHistory(size_t capacity = 10000);  // Default to 10k snapshots

    // Add a new L2 snapshot to the circular buffer (top 100 levels)
    void addSnapshot(std::unique_ptr<L2Snapshot> snapshot);

    // Retrieve a snapshot by index (0 = oldest, size_-1 = newest)
    std::unique_ptr<L2Snapshot> getSnapshotByIndex(size_t index) const;

    // Retrieve a snapshot by timestamp (closest match for heatmap playback)
    std::unique_ptr<L2Snapshot> getSnapshotByTime(int64_t targetTime) const;

    // Get the most recent snapshot
    std::unique_ptr<L2Snapshot> getLatestSnapshot() const;

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

    // Get a range of snapshots for heatmap rendering
    std::vector<std::unique_ptr<L2Snapshot>> getSnapshotsInRange(int64_t startTime, int64_t endTime) const;

private:
    // Helper to get timestamp at internal index (for searching)
    int64_t getSnapshotTimeAtInternalIndex(size_t internalIdx) const;
};

#endif // ORDERBOOK_HISTORY_H