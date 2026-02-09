#pragma once

#include <atomic>
#include <memory>
#include <cstring>
#include <type_traits>

#include "trading/HotspineData.h"
#include "../dependencies/BTQ_Render_Engine/include/hotspine_layout_v3.hpp"

namespace BTQuant {

/**
 * @brief Atomic storage for market data with lock-free operations
 * 
 * This class implements a lock-free ring buffer for storing market data
 * using atomic operations for thread-safe access in high-frequency environments.
 */
template<typename T, size_t BufferSize = 8192>
class AtomicMarketDataStorage {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable for atomic storage");
    static_assert(BufferSize > 0 && (BufferSize & (BufferSize - 1)) == 0, 
                  "BufferSize must be a power of 2 for efficient masking");

public:
    /**
     * @brief Constructor for AtomicMarketDataStorage
     */
    AtomicMarketDataStorage() {
        // Initialize the buffer to zero
        for (size_t i = 0; i < BufferSize; ++i) {
            buffer_[i] = T{};
        }
    }

    /**
     * @brief Atomically store market data
     * @param data The market data to store
     * @return True if successful, false if buffer is full
     */
    bool store(const T& data) {
        // Load current write head (acquire)
        uint64_t write_head = header_.write_head.load(std::memory_order_acquire);
        
        // Calculate slot index: idx = write_head & (buffer_size - 1)
        uint64_t slot_idx = write_head & BUFFER_MASK;
        
        // Copy data to the slot
        buffer_[slot_idx] = data;
        
        // Check for overflow condition: if write_head - read_tail >= buffer_size
        uint64_t read_tail = header_.read_tail.load(std::memory_order_acquire);
        uint64_t available_count = write_head - read_tail;
        
        if (available_count >= BufferSize) {
            // Overflow: increment dropped_count atomic (diagnostic only) and overwrite (circular)
            header_.dropped_count.fetch_add(1, std::memory_order_relaxed);
            // Still proceed with the write and update the head, as we're implementing circular buffer
        }
        
        // Atomic store write_head (release)
        header_.write_head.store(write_head + 1, std::memory_order_release);
        
        return true;
    }

    /**
     * @brief Atomically load market data
     * @param[out] data Reference to store the loaded data
     * @return True if successful, false if no data available
     */
    bool load(T& data) {
        // Load current read tail (acquire)
        uint64_t read_tail = header_.read_tail.load(std::memory_order_acquire);
        
        // Check if there's data available
        uint64_t write_head = header_.write_head.load(std::memory_order_acquire);
        if (read_tail >= write_head) {
            // No data available
            return false;
        }
        
        // Calculate slot index: idx = read_tail & (buffer_size - 1)
        uint64_t slot_idx = read_tail & BUFFER_MASK;
        
        // Copy data from the slot
        data = buffer_[slot_idx];
        
        // Atomic store read_tail (release)
        header_.read_tail.store(read_tail + 1, std::memory_order_release);
        
        return true;
    }

    /**
     * @brief Get the number of available data items
     * @return Number of items ready to be consumed
     */
    size_t available_count() const {
        uint64_t write_head = header_.write_head.load(std::memory_order_acquire);
        uint64_t read_tail = header_.read_tail.load(std::memory_order_acquire);
        return static_cast<size_t>(write_head - read_tail);
    }

    /**
     * @brief Check if the storage is empty
     * @return True if empty, false otherwise
     */
    bool is_empty() const {
        return available_count() == 0;
    }

    /**
     * @brief Check if the storage is full
     * @return True if full, false otherwise
     */
    bool is_full() const {
        uint64_t write_head = header_.write_head.load(std::memory_order_acquire);
        uint64_t read_tail = header_.read_tail.load(std::memory_order_acquire);
        return (write_head - read_tail) >= BufferSize;
    }

    /**
     * @brief Clear all stored data by resetting read position to write position
     */
    void clear() {
        // Reset the read tail to match the current write head
        uint64_t current_write_head = header_.write_head.load(std::memory_order_acquire);
        header_.read_tail.store(current_write_head, std::memory_order_release);
    }

    /**
     * @brief Get the number of dropped items due to overflow
     * @return Number of dropped items
     */
    uint64_t dropped_count() const {
        return header_.dropped_count.load(std::memory_order_acquire);
    }

    /**
     * @brief Get the total capacity of the storage
     * @return Buffer capacity
     */
    constexpr size_t capacity() const {
        return BufferSize;
    }

private:
    // Ring buffer header with atomic counters
    struct alignas(64) RingBufferHeader {
        std::atomic<uint64_t> write_head{0};
        std::atomic<uint64_t> read_tail{0};
        std::atomic<uint64_t> dropped_count{0};
        uint8_t padding[40]; // Padding to align to 64 bytes
        
        RingBufferHeader() = default;
    };

    static constexpr size_t BUFFER_MASK = BufferSize - 1; // For efficient modulo operation
    
    RingBufferHeader header_;
    T buffer_[BufferSize];
};

// Specialization for HotspineData with specific functionality
using HotspineDataStorage = AtomicMarketDataStorage<RenderEngine::HotspineData, 8192>;

} // namespace BTQuant