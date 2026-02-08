#include "orderbookhistory.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>

OrderBookHistory::OrderBookHistory(size_t capacity)
    : capacity_(capacity), head_(0), size_(0) {
    buffer_.resize(capacity_);  // Pre-allocate the vector with null pointers
}

void OrderBookHistory::addSnapshot(std::unique_ptr<L2Snapshot> snapshot) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (size_ < capacity_) {
        // Buffer is not full, just append
        buffer_[size_] = std::move(snapshot);
        size_++;
    } else {
        // Buffer is full, overwrite the oldest element
        buffer_[head_] = std::move(snapshot);
        head_ = (head_ + 1) % capacity_;
    }
}

std::unique_ptr<L2Snapshot> OrderBookHistory::getSnapshotByIndex(size_t index) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (index >= size_) {
        return nullptr;
    }

    // Calculate the actual position in the circular buffer
    size_t actual_pos;
    if (size_ < capacity_) {
        // Buffer hasn't wrapped around yet
        actual_pos = index;
    } else {
        // Buffer has wrapped around
        actual_pos = (head_ + index) % capacity_;
    }
    
    if (buffer_[actual_pos]) {
        return std::make_unique<L2Snapshot>(*buffer_[actual_pos]);
    }
    return nullptr;
}

std::unique_ptr<L2Snapshot> OrderBookHistory::getSnapshotByTime(int64_t targetTime) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (size_ == 0) {
        return nullptr;
    }

    // Find the closest snapshot to the target time
    // In a circular buffer with chronological insertion, older items are overwritten,
    // so we need to search through all valid entries
    std::unique_ptr<L2Snapshot> closest_snapshot = nullptr;
    int64_t min_diff = std::numeric_limits<int64_t>::max();

    for (size_t i = 0; i < size_; ++i) {
        size_t actual_pos;
        if (size_ < capacity_) {
            // Buffer hasn't wrapped around yet
            actual_pos = i;
        } else {
            // Buffer has wrapped around
            actual_pos = (head_ + i) % capacity_;
        }
        
        if (buffer_[actual_pos]) {
            int64_t current_time = buffer_[actual_pos]->timestamp;
            int64_t diff = std::abs(current_time - targetTime);

            if (diff < min_diff) {
                min_diff = diff;
                closest_snapshot = std::make_unique<L2Snapshot>(*buffer_[actual_pos]);
            }
        }
    }

    return closest_snapshot;
}

std::unique_ptr<L2Snapshot> OrderBookHistory::getLatestSnapshot() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (size_ == 0) {
        return nullptr;
    }

    size_t latest_pos;
    if (size_ < capacity_) {
        // Buffer hasn't wrapped around yet
        latest_pos = size_ - 1;
    } else {
        // Buffer has wrapped around
        latest_pos = (head_ + size_ - 1) % capacity_;
    }
    
    if (buffer_[latest_pos]) {
        return std::make_unique<L2Snapshot>(*buffer_[latest_pos]);
    }
    return nullptr;
}

size_t OrderBookHistory::size() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return size_;
}

size_t OrderBookHistory::capacity() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return capacity_;
}

bool OrderBookHistory::empty() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return size_ == 0;
}

void OrderBookHistory::clear() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (auto& ptr : buffer_) {
        ptr.reset();  // Explicitly reset each unique_ptr
    }
    head_ = 0;
    size_ = 0;
}

std::vector<int64_t> OrderBookHistory::getAllTimestamps() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<int64_t> timestamps;
    timestamps.reserve(size_);

    for (size_t i = 0; i < size_; ++i) {
        size_t actual_pos;
        if (size_ < capacity_) {
            // Buffer hasn't wrapped around yet
            actual_pos = i;
        } else {
            // Buffer has wrapped around
            actual_pos = (head_ + i) % capacity_;
        }
        
        if (buffer_[actual_pos]) {
            timestamps.push_back(buffer_[actual_pos]->timestamp);
        }
    }

    return timestamps;
}

std::vector<std::unique_ptr<L2Snapshot>> OrderBookHistory::getSnapshotsInRange(int64_t startTime, int64_t endTime) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<std::unique_ptr<L2Snapshot>> result;

    for (size_t i = 0; i < size_; ++i) {
        size_t actual_pos;
        if (size_ < capacity_) {
            // Buffer hasn't wrapped around yet
            actual_pos = i;
        } else {
            // Buffer has wrapped around
            actual_pos = (head_ + i) % capacity_;
        }
        
        if (buffer_[actual_pos] && 
            buffer_[actual_pos]->timestamp >= startTime && 
            buffer_[actual_pos]->timestamp <= endTime) {
            result.push_back(std::make_unique<L2Snapshot>(*buffer_[actual_pos]));
        }
    }

    return result;
}

int64_t OrderBookHistory::getSnapshotTimeAtInternalIndex(size_t internalIdx) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (internalIdx >= size_) {
        throw std::out_of_range("Index out of range");
    }

    size_t actual_pos;
    if (size_ < capacity_) {
        // Buffer hasn't wrapped around yet
        actual_pos = internalIdx;
    } else {
        // Buffer has wrapped around
        actual_pos = (head_ + internalIdx) % capacity_;
    }
    
    if (!buffer_[actual_pos]) {
        throw std::runtime_error("Invalid snapshot at index");
    }
    
    return buffer_[actual_pos]->timestamp;
}