#include "orderbookhistory.h"
#include <algorithm>
#include <cmath>

// Constructor implementation
OrderBookHistory::OrderBookHistory(size_t capacity)
    : capacity_(capacity), head_(0), size_(0) {
    buffer_.resize(capacity_);
}

// Add a new L2 snapshot to the circular buffer (top 100 levels)
void OrderBookHistory::addSnapshot(std::unique_ptr<L2Snapshot> snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (size_ < capacity_) {
        // Buffer is not full, add at current size position
        buffer_[size_] = std::move(snapshot);
        size_++;
    } else {
        // Buffer is full, overwrite the oldest entry (head_)
        buffer_[head_] = std::move(snapshot);
        head_ = (head_ + 1) % capacity_; // Move head forward in circular fashion
    }
}

// Retrieve a snapshot by index (0 = oldest, size_-1 = newest)
std::unique_ptr<L2Snapshot> OrderBookHistory::getSnapshotByIndex(size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (index >= size_) {
        return nullptr; // Index out of bounds
    }

    size_t actualIndex;
    if (size_ < capacity_) {
        // Buffer not full, direct indexing
        actualIndex = index;
    } else {
        // Buffer is full, adjust for circular nature
        actualIndex = (head_ + index) % capacity_;
    }

    // Return a copy of the snapshot to avoid shared ownership issues
    auto original = buffer_[actualIndex].get();
    auto snapshot = std::make_unique<L2Snapshot>(100); // Use 100 levels

    snapshot->timestamp = original->timestamp;
    snapshot->bids = original->bids;  // Copy the vectors
    snapshot->asks = original->asks;

    return snapshot;
}

// Retrieve a snapshot by timestamp (closest match for heatmap playback)
std::unique_ptr<L2Snapshot> OrderBookHistory::getSnapshotByTime(int64_t targetTime) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (size_ == 0) {
        return nullptr;
    }

    // Find the closest timestamp match using binary search approach
    size_t bestIndex = 0;
    int64_t minDiff = std::abs(getSnapshotTimeAtInternalIndex(0) - targetTime);
    bool found = buffer_[0] != nullptr;

    // Search through all stored snapshots to find the closest match
    for (size_t i = 0; i < size_; ++i) {
        size_t actualIndex = (head_ + i) % capacity_;
        if (buffer_[actualIndex]) {
            int64_t currentTime = buffer_[actualIndex]->timestamp;
            int64_t diff = std::abs(currentTime - targetTime);

            if (diff < minDiff) {
                minDiff = diff;
                bestIndex = actualIndex;
                found = true;
            }
        }
    }

    if (!found || !buffer_[bestIndex]) {
        return nullptr;
    }

    // Return a copy of the best matching snapshot
    auto original = buffer_[bestIndex].get();
    auto snapshot = std::make_unique<L2Snapshot>(100);

    snapshot->timestamp = original->timestamp;
    snapshot->bids = original->bids;
    snapshot->asks = original->asks;

    return snapshot;
}

// Get the most recent snapshot
std::unique_ptr<L2Snapshot> OrderBookHistory::getLatestSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (size_ == 0) {
        return nullptr;
    }

    size_t latestIndex;
    if (size_ < capacity_) {
        // Buffer not full, latest is at size_ - 1
        latestIndex = size_ - 1;
    } else {
        // Buffer is full, latest is at (head_ + size_ - 1) % capacity_
        latestIndex = (head_ + size_ - 1) % capacity_;
    }

    if (!buffer_[latestIndex]) {
        return nullptr;
    }

    auto original = buffer_[latestIndex].get();
    auto snapshot = std::make_unique<L2Snapshot>(100);

    snapshot->timestamp = original->timestamp;
    snapshot->bids = original->bids;
    snapshot->asks = original->asks;

    return snapshot;
}

// Get snapshot count
size_t OrderBookHistory::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
}

// Get buffer capacity
size_t OrderBookHistory::capacity() const {
    return capacity_;
}

// Check if buffer is empty
bool OrderBookHistory::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_ == 0;
}

// Clear the buffer
void OrderBookHistory::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& snapshot : buffer_) {
        snapshot.reset();
    }
    head_ = 0;
    size_ = 0;
}

// Get all timestamps for UI timeline controls
std::vector<int64_t> OrderBookHistory::getAllTimestamps() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int64_t> timestamps;
    timestamps.reserve(size_);

    for (size_t i = 0; i < size_; ++i) {
        size_t actualIndex = (head_ + i) % capacity_;
        if (buffer_[actualIndex]) {
            timestamps.push_back(buffer_[actualIndex]->timestamp);
        }
    }

    return timestamps;
}

// Get a range of snapshots for heatmap rendering
std::vector<std::unique_ptr<L2Snapshot>> OrderBookHistory::getSnapshotsInRange(int64_t startTime, int64_t endTime) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::unique_ptr<L2Snapshot>> result;
    
    for (size_t i = 0; i < size_; ++i) {
        size_t actualIndex = (head_ + i) % capacity_;
        if (buffer_[actualIndex]) {
            int64_t currentTimestamp = buffer_[actualIndex]->timestamp;
            if (currentTimestamp >= startTime && currentTimestamp <= endTime) {
                // Add a copy of the snapshot to the result
                auto original = buffer_[actualIndex].get();
                auto snapshot = std::make_unique<L2Snapshot>(100);
                
                snapshot->timestamp = original->timestamp;
                snapshot->bids = original->bids;
                snapshot->asks = original->asks;
                
                result.push_back(std::move(snapshot));
            }
        }
    }
    
    return result;
}

// Helper to get timestamp at internal index (for searching)
int64_t OrderBookHistory::getSnapshotTimeAtInternalIndex(size_t internalIdx) const {
    if (buffer_[internalIdx] != nullptr) {
        return buffer_[internalIdx]->timestamp;
    }
    return 0;
}