#ifndef PUBBTQUANT_TRADERINGBUFFER_H
#define PUBBTQUANT_TRADERINGBUFFER_H

#include "rawtradetable.h"
#include <atomic>
#include <memory>
#include <vector>
#include <chrono>
#include <cstring>
#include <thread>
#include <functional>
#include <algorithm>

// Structure to hold ring buffer entries for trades
struct alignas(64) TradeRingBufferEntry {
    std::atomic<bool> valid;
    RawTrade trade_data;

    TradeRingBufferEntry() : valid(false) {}
};

// Lock-free ring buffer specifically for trade data
class TradeRingBuffer {
private:
    static constexpr size_t TRADE_RING_BUFFER_SIZE = 1048576; // 2^20
    static constexpr size_t TRADE_RING_BUFFER_MASK = TRADE_RING_BUFFER_SIZE - 1;
    std::unique_ptr<TradeRingBufferEntry[]> trade_ring_buffer_;
    
    std::atomic<uint64_t> write_head_{0};
    std::atomic<uint64_t> read_tail_{0};

    // Statistics
    std::atomic<uint64_t> total_writes_{0};
    std::atomic<uint64_t> dropped_writes_{0};

public:
    TradeRingBuffer()
        : trade_ring_buffer_(std::make_unique<TradeRingBufferEntry[]>(TRADE_RING_BUFFER_SIZE)) {
    }

    ~TradeRingBuffer() = default;

    // Write a trade to the ring buffer (non-blocking)
    bool write_trade(const RawTrade& trade) {
        uint64_t current_write = write_head_.load(std::memory_order_acquire);
        uint64_t current_read = read_tail_.load(std::memory_order_acquire);

        // Check if buffer is full
        if ((current_write - current_read) >= TRADE_RING_BUFFER_SIZE) {
            dropped_writes_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        // Attempt to advance write head
        uint64_t new_write;
        do {
            new_write = current_write + 1;
        } while (!write_head_.compare_exchange_weak(current_write, new_write,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire));

        // Successfully acquired slot, write data to ring buffer
        size_t index = new_write & TRADE_RING_BUFFER_MASK;
        trade_ring_buffer_[index].trade_data = trade;
        trade_ring_buffer_[index].valid.store(true, std::memory_order_release);

        total_writes_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Read a single trade from the atomic tail (non-blocking, no vector copy)
    bool read_from_tail(RawTrade& out_trade) const {
        uint64_t current_write = write_head_.load(std::memory_order_acquire);
        uint64_t current_read = read_tail_.load(std::memory_order_acquire);

        // If read tail has caught up to write head, there's nothing to read
        if (current_read >= current_write) {
            return false;
        }

        // Read from the current tail position
        size_t index = current_read & TRADE_RING_BUFFER_MASK;
        
        // Check if the entry is valid
        if (!trade_ring_buffer_[index].valid.load(std::memory_order_acquire)) {
            return false;
        }

        // Copy the trade data atomically
        out_trade = trade_ring_buffer_[index].trade_data;
        
        return true;
    }

    // Read multiple trades from the tail without copying vectors (batch read)
    size_t read_batch_from_tail(RawTrade* out_trades, size_t max_count) const {
        if (!out_trades || max_count == 0) {
            return 0;
        }

        uint64_t current_write = write_head_.load(std::memory_order_acquire);
        uint64_t current_read = read_tail_.load(std::memory_order_acquire);

        size_t count = 0;
        uint64_t pos = current_read;
        
        // Read up to max_count trades starting from the current read position
        while (pos < current_write && count < max_count) {
            size_t index = pos & TRADE_RING_BUFFER_MASK;
            
            // Check if the entry is valid
            if (trade_ring_buffer_[index].valid.load(std::memory_order_acquire)) {
                // Copy the trade data directly to the output array
                out_trades[count] = trade_ring_buffer_[index].trade_data;
                count++;
            }
            pos++;
        }

        return count;
    }

    // Atomically advance the read tail by a specified number of positions
    void advance_read_tail(size_t num_advances) {
        uint64_t current_read = read_tail_.load(std::memory_order_acquire);
        read_tail_.store(current_read + num_advances, std::memory_order_release);
    }

    // Get the current read tail position
    uint64_t get_read_tail() const {
        return read_tail_.load(std::memory_order_acquire);
    }

    // Get the current write head position
    uint64_t get_write_head() const {
        return write_head_.load(std::memory_order_acquire);
    }

    // Get number of unread items in the buffer
    uint64_t get_unread_count() const {
        uint64_t current_write = write_head_.load(std::memory_order_acquire);
        uint64_t current_read = read_tail_.load(std::memory_order_acquire);
        return (current_write > current_read) ? (current_write - current_read) : 0;
    }

    // Get statistics about the ring buffer
    struct RingBufferStats {
        uint64_t total_writes;
        uint64_t dropped_writes;
        uint64_t write_head;
        uint64_t read_tail;
        uint64_t unread_count;
    };

    RingBufferStats get_stats() const {
        RingBufferStats stats;
        stats.total_writes = total_writes_.load(std::memory_order_acquire);
        stats.dropped_writes = dropped_writes_.load(std::memory_order_acquire);
        stats.write_head = write_head_.load(std::memory_order_acquire);
        stats.read_tail = read_tail_.load(std::memory_order_acquire);
        stats.unread_count = get_unread_count();
        return stats;
    }

    // Reset statistics
    void reset_stats() {
        total_writes_.store(0, std::memory_order_release);
        dropped_writes_.store(0, std::memory_order_release);
    }
};

#endif // PUBBTQUANT_TRADERINGBUFFER_H